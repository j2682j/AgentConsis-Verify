from __future__ import annotations

"""Fetch full page content for filtered search sources."""

import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

try:
    from markdownify import markdownify
except Exception:
    markdownify = None  # type: ignore

from ...config import SearchSourceCandidate
from ..content_requirement import ContentRequirementVerifier

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf;q=0.8,text/plain;q=0.7,*/*;q=0.5"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
LOW_TRUST_DOMAINS = (
    "quora.com",
    "youtube.com",
    "youtu.be",
    "pinterest.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
)
JS_RENDER_MARKERS = (
    "enable javascript",
    "requires javascript",
    "please enable js",
    "id=\"__next\"",
    "id=\"root\"",
    "__next_data__",
    "window.__",
)


@dataclass(frozen=True)
class PageFetchResult:
    """
    記錄一次 full-page fetch 的文字內容與抽取方式。
    Args:
        - content: 抽取並截斷後的頁面文字。
        - method: 使用的抽取策略，例如 trafilatura、beautifulsoup、pdf_pymupdf。
        - content_type: HTTP content-type。
        - status_code: HTTP status code。
        - quality_status: 內容品質檢查結果。
        - trace: fetch/extraction 步驟摘要。
    Returns:
        - PageFetchResult: 可寫回 source.raw_content 的抓取結果。
    """

    content: str
    method: str
    raw_html: str = ""
    content_type: str = ""
    status_code: int = 0
    quality_status: str = "unknown"
    trace: tuple[str, ...] = ()
    is_complete: bool = False
    truncated: bool = False
    original_char_count: int = 0
    final_url: str = ""
    transport_ok: bool = False
    content_extracted: bool = False


@dataclass(frozen=True)
class _ExtractionResult:
    text: str
    method: str
    trace: tuple[str, ...]


def _limit_text(text: str, token_limit: int) -> str:
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    return text[:char_limit].rstrip() + "... [truncated]"


def _content_scope(text: str, token_limit: int, quality_status: str) -> dict[str, object]:
    original_char_count = len(str(text or ""))
    lowered = str(text or "").casefold()
    truncated = (
        original_char_count > token_limit * CHARS_PER_TOKEN
        or "[section truncated]" in lowered
        or "[truncated]" in lowered
    )
    return {
        "is_complete": quality_status == "ok" and not truncated,
        "truncated": truncated,
        "original_char_count": original_char_count,
    }


def _page_fetch_result(
    *,
    text: str,
    method: str,
    max_tokens: int,
    quality_status: str,
    raw_html: str = "",
    content_type: str = "",
    status_code: int = 0,
    trace: tuple[str, ...] = (),
    final_url: str = "",
) -> PageFetchResult:
    scope = _content_scope(text, max_tokens, quality_status)
    return PageFetchResult(
        content=_limit_text(text, max_tokens),
        method=method,
        raw_html=raw_html[:2_000_000],
        content_type=content_type,
        status_code=status_code,
        quality_status=quality_status,
        trace=trace,
        is_complete=bool(scope["is_complete"]),
        truncated=bool(scope["truncated"]),
        original_char_count=int(scope["original_char_count"]),
        final_url=str(final_url or "").strip(),
        transport_ok=bool((status_code and status_code < 400) or text),
        content_extracted=bool(str(text or "").strip()),
    )


def _normalize_extracted_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\x00", " ")
    cleaned = re.sub(r"\|{3,}", " | ", cleaned)
    cleaned = re.sub(r"(?:\|\s*){8,}", " | ", cleaned)
    cleaned = re.sub(r"(?:-+\|){3,}-*", "", cleaned)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _section(title: str, lines: list[str], *, max_chars: int = 6000) -> str:
    cleaned_lines = _dedupe_lines(lines)
    if not cleaned_lines:
        return ""
    body = "\n".join(cleaned_lines)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "... [section truncated]"
    return f"{title}:\n{body}"


def _dedupe_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = _normalize_extracted_text(line)
        if not cleaned:
            continue
        key = re.sub(r"\W+", " ", cleaned.casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _load_soup(html: str):
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return None
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _clean_soup_copy(html: str):
    soup = _load_soup(html)
    if soup is None:
        return None
    for selector in (
        "script:not([type='application/ld+json'])",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "iframe",
        "button",
    ):
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def _extract_metadata(soup) -> list[str]:
    lines: list[str] = []
    if soup is None:
        return lines
    title = soup.find("title")
    if title and title.get_text(strip=True):
        lines.append(f"Title: {title.get_text(' ', strip=True)}")
    for attrs in (
        {"name": "description"},
        {"property": "og:title"},
        {"property": "og:description"},
        {"name": "twitter:title"},
        {"name": "twitter:description"},
    ):
        tag = soup.find("meta", attrs=attrs)
        content = tag.get("content", "") if tag else ""
        if content:
            label = attrs.get("name") or attrs.get("property") or "meta"
            lines.append(f"{label}: {content}")
    return lines


def _extract_headings(soup) -> list[str]:
    if soup is None:
        return []
    lines: list[str] = []
    for tag in soup.select("h1, h2, h3"):
        text = tag.get_text(" ", strip=True)
        if text:
            lines.append(text)
    return lines[:20]


def _extract_tables(soup, *, max_tables: int = 8, max_rows: int = 40, max_cols: int = 12) -> list[str]:
    if soup is None:
        return []
    tables: list[str] = []
    for table_index, table in enumerate(soup.find_all("table"), start=1):
        if len(tables) >= max_tables:
            break
        rows: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells[:max_cols])
            if len(rows) >= max_rows:
                break
        if not rows:
            continue
        width = max(len(row) for row in rows)
        normalized_rows = [
            row + [""] * (width - len(row))
            for row in rows
        ]
        markdown_rows = [" | ".join(row) for row in normalized_rows]
        tables.append(f"Table {table_index}\n" + "\n".join(markdown_rows))
    return tables


def _extract_lists(soup, *, max_lists: int = 6, max_items: int = 12) -> list[str]:
    if soup is None:
        return []
    lists: list[str] = []
    for list_index, tag in enumerate(soup.find_all(["ul", "ol"]), start=1):
        if len(lists) >= max_lists:
            break
        items = [
            item.get_text(" ", strip=True)
            for item in tag.find_all("li", recursive=False)
        ]
        items = [item for item in items if len(item) >= 8][:max_items]
        if len(items) < 2:
            continue
        lists.append(f"List {list_index}\n" + "\n".join(f"- {item}" for item in items))
    return lists


def _extract_captions_and_alt(soup, *, max_items: int = 20) -> list[str]:
    if soup is None:
        return []
    lines: list[str] = []
    for tag in soup.select("figcaption, caption"):
        text = tag.get_text(" ", strip=True)
        if text:
            lines.append(text)
    for image in soup.find_all("img"):
        alt = image.get("alt", "")
        if alt and len(alt) >= 6:
            lines.append(f"Image alt: {alt}")
    return lines[:max_items]


def _extract_json_ld(soup, *, max_items: int = 18) -> list[str]:
    if soup is None:
        return []
    lines: list[str] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        for line in _json_ld_lines(payload):
            lines.append(line)
            if len(lines) >= max_items:
                return lines
    return lines


def _json_ld_lines(value: object, *, prefix: str = "") -> list[str]:
    keys = {
        "name",
        "headline",
        "description",
        "datePublished",
        "dateModified",
        "author",
        "creator",
        "publisher",
        "about",
        "location",
        "address",
    }
    lines: list[str] = []
    if isinstance(value, list):
        for item in value:
            lines.extend(_json_ld_lines(item, prefix=prefix))
        return lines
    if not isinstance(value, dict):
        return lines
    for key, item in value.items():
        if key == "@graph":
            lines.extend(_json_ld_lines(item, prefix=prefix))
            continue
        if key not in keys:
            continue
        label = f"{prefix}{key}"
        if isinstance(item, str) and item.strip():
            lines.append(f"{label}: {item.strip()}")
        elif isinstance(item, dict):
            for nested_key in ("name", "headline", "description"):
                nested = item.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    lines.append(f"{label}.{nested_key}: {nested.strip()}")
        elif isinstance(item, list):
            joined = ", ".join(
                entry.strip()
                for entry in item
                if isinstance(entry, str) and entry.strip()
            )
            if joined:
                lines.append(f"{label}: {joined}")
    return lines


def _looks_like_pdf(url: str, content_type: str, content: bytes) -> bool:
    if "pdf" in content_type.lower():
        return True
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    return content[:4] == b"%PDF"


def _extract_pdf_text(content: bytes) -> tuple[str, str] | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    try:
        document = fitz.open(stream=content, filetype="pdf")
        pages = [
            f"[PDF Page {index}]\n{page.get_text('text')}"
            for index, page in enumerate(document, start=1)
        ]
        return _normalize_extracted_text("\n\n".join(pages)), "pdf_pymupdf"
    except Exception as exc:
        logger.debug("PyMuPDF PDF extraction failed: %s", exc)
        return None


def _extract_with_trafilatura(html: str, url: str) -> tuple[str, str] | None:
    try:
        import trafilatura  # type: ignore
    except Exception:
        return None

    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:
        logger.debug("trafilatura extraction failed for %s: %s", url, exc)
        return None
    text = _normalize_extracted_text(extracted or "")
    if text:
        return text, "trafilatura"
    return None


def _extract_with_readability(html: str, url: str) -> tuple[str, str] | None:
    try:
        from readability import Document  # type: ignore
    except Exception:
        return None

    try:
        summary = Document(html).summary()
    except Exception as exc:
        logger.debug("readability extraction failed for %s: %s", url, exc)
        return None
    return _extract_with_beautifulsoup(summary, method="readability")


def _extract_with_beautifulsoup(
    html: str,
    *,
    method: str = "beautifulsoup",
) -> tuple[str, str] | None:
    soup = _clean_soup_copy(html)
    if soup is None:
        return None
    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    text = _normalize_extracted_text(text)
    if text:
        return text, method
    return None


def _extract_with_markdownify(html: str, url: str) -> tuple[str, str] | None:
    if markdownify is None:
        return None
    try:
        text = markdownify(html)  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("markdownify failed for %s: %s", url, exc)
        return None
    text = _normalize_extracted_text(text)
    if text:
        return text, "markdownify"
    return None


def _is_probably_js_shell(html: str, extracted_text: str) -> bool:
    lower = html[:5000].lower()
    if any(marker in lower for marker in JS_RENDER_MARKERS):
        return len(extracted_text) < 800
    return False


def _content_quality_status(text: str, *, min_chars: int = 500) -> str:
    cleaned = _normalize_extracted_text(text)
    if not cleaned:
        return "empty"
    if len(cleaned) < min_chars:
        return "too_short"
    alphanumeric = sum(character.isalnum() for character in cleaned)
    if alphanumeric / max(1, len(cleaned)) < 0.35:
        return "low_alphanumeric_ratio"
    words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
    if len(words) >= 30:
        unique_ratio = len({word.casefold() for word in words}) / len(words)
        if unique_ratio < 0.16:
            return "repetitive"
    boilerplate_markers = (
        "enable javascript",
        "subscribe to continue",
        "please log in",
        "access denied",
        "checking your browser",
    )
    lower = cleaned[:2000].casefold()
    if any(marker in lower for marker in boilerplate_markers):
        return "boilerplate_or_access_wall"
    return "ok"


def _should_try_playwright(
    *,
    html: str = "",
    extracted_text: str = "",
    quality_status: str = "",
    status_code: int = 0,
) -> bool:
    if status_code in {403, 406, 408, 409, 429}:
        return True
    if quality_status in {"empty", "boilerplate_or_access_wall"}:
        return True
    if quality_status == "too_short" and len(_normalize_extracted_text(extracted_text)) < 120:
        return True
    if html and _is_probably_js_shell(html, extracted_text):
        return True
    return False


def _fetch_with_playwright(url: str) -> tuple[str, str, str] | None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_(),
            )
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
            text = page.locator("body").inner_text(timeout=5000)
            html = page.content()
            browser.close()
    except Exception as exc:
        logger.debug("Playwright fetch failed for %s: %s", url, exc)
        return None

    text = _normalize_extracted_text(text)
    if text:
        return text, "playwright_body_text", html
    return None


def _browser_result_parts(value: object) -> tuple[str, str, str]:
    """接受測試替身的二元素結果與正式瀏覽器的三元素結果。"""
    if not isinstance(value, tuple) or len(value) < 2:
        return "", "", ""
    text = str(value[0] or "")
    method = str(value[1] or "")
    html = str(value[2] or "") if len(value) >= 3 else ""
    return text, method, html


def _extract_html_text(html: str, url: str) -> _ExtractionResult | None:
    soup = _load_soup(html)
    sections: list[str] = []
    trace: list[str] = []

    metadata = _extract_metadata(soup)
    section = _section("Metadata", metadata, max_chars=1200)
    if section:
        sections.append(section)
        trace.append(f"metadata:{len(metadata)}")

    json_ld = _extract_json_ld(soup)
    section = _section("Structured Data", json_ld, max_chars=1600)
    if section:
        sections.append(section)
        trace.append(f"json_ld:{len(json_ld)}")

    headings = _extract_headings(soup)
    section = _section("Headings", headings, max_chars=1200)
    if section:
        sections.append(section)
        trace.append(f"headings:{len(headings)}")

    main_text = ""
    main_method = ""
    for candidate in (
        _extract_with_trafilatura(html, url),
        _extract_with_readability(html, url),
        _extract_with_beautifulsoup(html),
        _extract_with_markdownify(html, url),
    ):
        if candidate and candidate[0]:
            main_text, main_method = candidate
            break
    if main_text:
        sections.append(_section("Content", [main_text], max_chars=9000))
        trace.append(f"main:{main_method}:{len(main_text)}")

    tables = _extract_tables(soup)
    section = _section("Tables", tables, max_chars=12000)
    if section:
        sections.append(section)
        trace.append(f"tables:{len(tables)}")

    lists = _extract_lists(soup)
    section = _section("Lists", lists, max_chars=3500)
    if section:
        sections.append(section)
        trace.append(f"lists:{len(lists)}")

    captions = _extract_captions_and_alt(soup)
    section = _section("Captions", captions, max_chars=1800)
    if section:
        sections.append(section)
        trace.append(f"captions:{len(captions)}")

    text = _normalize_extracted_text("\n\n".join(section for section in sections if section))
    if not text:
        return None

    method = "structured_html"
    if main_method:
        method = f"structured_html+{main_method}"
    return _ExtractionResult(text=text, method=method, trace=tuple(trace))


def _fetch_raw_content(
    url: str,
    *,
    max_tokens: int,
    force_browser: bool = False,
) -> PageFetchResult | None:
    if force_browser:
        browser_extracted = _fetch_with_playwright(url)
        text, method, browser_html = _browser_result_parts(browser_extracted)
        if text:
            quality_status = _content_quality_status(text, min_chars=160)
            return _page_fetch_result(
                text=text,
                method=method,
                raw_html=browser_html[:2_000_000],
                max_tokens=max_tokens,
                content_type="text/html",
                status_code=200,
                quality_status=quality_status,
                trace=("playwright_forced", f"quality:{quality_status}"),
                final_url=url,
            )

    response = None
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=10,
            allow_redirects=True,
        )
    except Exception as exc:
        logger.debug("Failed to fetch raw content for %s: %s", url, exc)
        return None

    status_code = int(getattr(response, "status_code", 0) or 0)
    content_type = response.headers.get("content-type", "")
    raw_bytes = response.content or b""
    trace = [f"requests_status:{status_code}"]

    if status_code >= 400:
        trace.append(f"http_error:{status_code}")
        if _should_try_playwright(status_code=status_code):
            browser_extracted = _fetch_with_playwright(url)
            text, method, browser_html = _browser_result_parts(browser_extracted)
            if text:
                quality_status = _content_quality_status(text)
                return _page_fetch_result(
                    text=text,
                    method=method,
                    raw_html=browser_html[:2_000_000],
                    max_tokens=max_tokens,
                    content_type=content_type,
                    status_code=status_code,
                    quality_status=quality_status,
                    trace=tuple(trace + ["playwright_fallback:http_error", f"quality:{quality_status}"]),
                    final_url=url,
                )
        return None

    if _looks_like_pdf(url, content_type, raw_bytes):
        extracted = _extract_pdf_text(raw_bytes)
        if extracted is not None:
            text, method = extracted
            quality_status = _content_quality_status(text, min_chars=160)
            return _page_fetch_result(
                text=text,
                method=method,
                max_tokens=max_tokens,
                content_type=content_type,
                status_code=status_code,
                quality_status=quality_status,
                trace=tuple(trace + [f"pdf:{method}", f"quality:{quality_status}"]),
                final_url=str(getattr(response, "url", "") or url),
            )
        return None

    if "text/plain" in content_type.lower():
        text = _normalize_extracted_text(response.text)
        quality_status = _content_quality_status(text, min_chars=160)
        return _page_fetch_result(
            text=text,
            method="text_plain",
            max_tokens=max_tokens,
            content_type=content_type,
            status_code=status_code,
            quality_status=quality_status,
            trace=tuple(trace + ["text_plain", f"quality:{quality_status}"]),
            final_url=str(getattr(response, "url", "") or url),
        )

    html = response.text
    extracted = _extract_html_text(html, url)
    if extracted is None:
        trace.append("html_extraction_empty")
        text = ""
        method = "none"
        quality_status = "empty"
    else:
        text = extracted.text
        method = extracted.method
        quality_status = _content_quality_status(text, min_chars=160)
        trace.extend(extracted.trace)
        trace.append(f"quality:{quality_status}")

    if _should_try_playwright(
        html=html,
        extracted_text=text,
        quality_status=quality_status,
        status_code=status_code,
    ):
        browser_extracted = _fetch_with_playwright(url)
        browser_text, browser_method, browser_html = _browser_result_parts(browser_extracted)
        if browser_text:
            browser_quality = _content_quality_status(browser_text, min_chars=160)
            if (
                quality_status != "ok"
                or len(browser_text) > len(text) + 250
                or browser_quality == "ok"
            ):
                text, method = browser_text, browser_method
                quality_status = browser_quality
                html = browser_html or html
                trace.append("playwright_fallback:used")
                trace.append(f"quality:{quality_status}")
            else:
                trace.append("playwright_fallback:discarded_static_better")

    if not text:
        return None

    return _page_fetch_result(
        text=text,
        method=method,
        raw_html=html[:2_000_000],
        max_tokens=max_tokens,
        content_type=content_type,
        status_code=status_code,
        quality_status=quality_status,
        trace=tuple(trace),
        final_url=str(getattr(response, "url", "") or url),
    )


def fetch_page_content_result(
    url: str,
    *,
    max_tokens: int = 2000,
    force_browser: bool = False,
) -> PageFetchResult | None:
    return _fetch_raw_content(
        url,
        max_tokens=max_tokens,
        force_browser=force_browser,
    )


def fetch_page_content(url: str, *, max_tokens: int = 2000) -> str | None:
    result = fetch_page_content_result(url, max_tokens=max_tokens)
    if result is None or not result.content:
        return None
    return result.content


class PageContentFetcher:
    """
    根據 SourceFilter 的判斷，替高價值搜尋來源抓取完整網頁內容。

    Args:
        - max_workers: 同時抓取網頁的最大工作數。
        - min_content_chars: 判定抓取結果足夠像全文內容的最小字元數。

    Returns:
        - PageContentFetcher: 可重複使用的完整網頁抓取服務。
    """

    def __init__(self, *, max_workers: int = 4, min_content_chars: int = 160) -> None:
        self.max_workers = max(1, max_workers)
        self.min_content_chars = max(1, min_content_chars)
        self.requirement_verifier = ContentRequirementVerifier()

    def fetch_sources(
        self,
        sources: list[SearchSourceCandidate],
        *,
        max_pages: int,
        max_tokens_per_source: int = 2000,
    ) -> int:
        """
        抓取被標記為 should_fetch_full_page 的來源，並將內容寫回 source.raw_content。

        Args:
            - sources: 已通過 SourceFilter 的來源候選。
            - max_pages: 本輪最多抓取的來源數。
            - max_tokens_per_source: 每個來源保留的最大 token 估計值。

        Returns:
            - int: 成功抓到完整內容的來源數量。
        """
        candidates = self._fetch_candidates(sources, max_pages=max_pages)
        if not candidates:
            return 0

        fetched_count = 0
        worker_count = min(self.max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_source = {
                executor.submit(
                    fetch_page_content_result,
                    source.url,
                    max_tokens=max_tokens_per_source,
                    force_browser=source.access_mode == "browser",
                ): source
                for source in candidates
            }
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                except Exception as exc:
                    source.filter_reasons.append(f"full_page_fetch_error:{type(exc).__name__}")
                    source.acquisition_state = "failed"
                    source.missing_content = [source.required_content]
                    continue

                content = result.content if result is not None else None
                if result is None:
                    source.filter_reasons.append("full_page_fetch_failed")
                    source.acquisition_state = "failed"
                    source.missing_content = [source.required_content]
                    continue
                source.filter_reasons.append(f"fetch_status:{result.status_code}")
                if result.content_type:
                    source.filter_reasons.append(
                        f"fetch_content_type:{result.content_type.split(';', 1)[0].strip()}"
                    )
                source.filter_reasons.append(f"fetch_quality:{result.quality_status}")
                for trace_item in result.trace[:12]:
                    source.filter_reasons.append(f"fetch_trace:{trace_item}")

                state = self.requirement_verifier.verify(
                    required_content=source.required_content,
                    content=str(content or ""),
                    method=result.method,
                    content_type=result.content_type,
                    status_code=result.status_code,
                    content_complete=result.is_complete,
                    source_kind=source.source_kind,
                )
                source.transport_ok = state.transport_ok
                source.content_extracted = state.content_extracted
                source.requirement_met = state.requirement_met
                source.acquisition_state = state.state
                source.missing_content = list(state.missing_content)
                source.filter_reasons.append(f"required_content:{state.required_content}")
                source.filter_reasons.append(f"acquisition_state:{state.state}")

                if not self._is_usable_content(source, content, result=result):
                    source.requirement_met = False
                    source.acquisition_state = (
                        "content_extracted" if source.content_extracted else "failed"
                    )
                    source.missing_content = [source.required_content]
                    source.filter_reasons.append("low_quality_full_page")
                    continue

                source.raw_content = str(content).strip()
                source.raw_html = str(result.raw_html or "").strip()
                source.content_complete = bool(result.is_complete)
                source.content_truncated = bool(result.truncated)
                source.original_content_chars = int(result.original_char_count)
                source.final_url = str(result.final_url or source.url)
                source.fetched = True
                source.should_fetch_full_page = False
                source.filter_reasons.append("full_page_fetched")
                source.filter_reasons.append(f"fetch_method:{result.method}")
                source.filter_reasons.append(
                    f"fetch_complete:{str(result.is_complete).lower()}"
                )
                source.filter_reasons.append(
                    f"fetch_truncated:{str(result.truncated).lower()}"
                )
                source.filter_reasons.append(
                    f"requirement_met:{str(state.requirement_met).lower()}"
                )
                fetched_count += 1

        return fetched_count

    def _fetch_candidates(
        self,
        sources: list[SearchSourceCandidate],
        *,
        max_pages: int,
    ) -> list[SearchSourceCandidate]:
        selected: list[SearchSourceCandidate] = []
        for source in sources:
            if len(selected) >= max_pages:
                break
            if not self._should_fetch(source):
                continue
            selected.append(source)
        return selected

    def _should_fetch(self, source: SearchSourceCandidate) -> bool:
        if source.blocked or not source.should_fetch_full_page:
            return False
        if source.fetched or self._has_full_page_content(source):
            return False
        if not source.url:
            return False
        domain = urlparse(source.url).netloc.lower()
        if not domain or any(marker in domain for marker in LOW_TRUST_DOMAINS):
            return False
        return True

    def _has_full_page_content(self, source: SearchSourceCandidate) -> bool:
        raw = str(source.raw_content or "").strip()
        snippet = str(source.snippet or "").strip()
        return bool(raw and raw != snippet and len(raw) > len(snippet) + 120)

    def _is_usable_content(
        self,
        source: SearchSourceCandidate,
        content: str | None,
        *,
        result: PageFetchResult | None = None,
    ) -> bool:
        if not content:
            return False
        text = str(content).strip()
        if len(text) < self.min_content_chars:
            return False
        if result is not None and result.quality_status in {
            "empty",
            "low_alphanumeric_ratio",
            "boilerplate_or_access_wall",
            "repetitive",
        }:
            return False
        snippet = str(source.snippet or "").strip()
        if snippet and text == snippet:
            return False
        return True


__all__ = [
    "PageContentFetcher",
    "PageFetchResult",
    "fetch_page_content",
    "fetch_page_content_result",
]
