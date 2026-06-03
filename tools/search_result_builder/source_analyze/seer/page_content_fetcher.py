from __future__ import annotations

"""Fetch full page content for filtered search sources."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

try:
    from markdownify import markdownify
except Exception:
    markdownify = None  # type: ignore

from ...config import SearchSourceCandidate

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
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


def _limit_text(text: str, token_limit: int) -> str:
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "... [truncated]"


def _fetch_raw_content(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to fetch raw content for %s: %s", url, exc)
        return None

    if markdownify is not None:
        try:
            return markdownify(response.text)  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("markdownify failed for %s: %s", url, exc)
    return response.text


def fetch_page_content(url: str, *, max_tokens: int = 2000) -> str | None:
    content = _fetch_raw_content(url)
    if not content:
        return None
    return _limit_text(content, max_tokens)


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
                    fetch_page_content,
                    source.url,
                    max_tokens=max_tokens_per_source,
                ): source
                for source in candidates
            }
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    content = future.result()
                except Exception as exc:
                    source.filter_reasons.append(f"full_page_fetch_error:{type(exc).__name__}")
                    continue

                if not self._is_usable_content(source, content):
                    source.filter_reasons.append("full_page_fetch_empty_or_short")
                    continue

                source.raw_content = str(content).strip()
                source.fetched = True
                source.should_fetch_full_page = False
                source.filter_reasons.append("full_page_fetched")
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

    def _is_usable_content(self, source: SearchSourceCandidate, content: str | None) -> bool:
        if not content:
            return False
        text = str(content).strip()
        if len(text) < self.min_content_chars:
            return False
        snippet = str(source.snippet or "").strip()
        if snippet and text == snippet:
            return False
        return True
