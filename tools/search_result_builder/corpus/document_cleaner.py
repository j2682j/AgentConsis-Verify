from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from utils.network_utils import normalize_text


class _ReadableHTMLParser(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }
    SKIP_TAGS = {
        "button",
        "canvas",
        "footer",
        "form",
        "header",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


class DocumentCleaner:
    """
    將抓取的 HTML 或 Markdown 清理成適合建立檢索 corpus 的正文。

    Args:
        - min_line_chars: 一般文字行的最小字元數。

    Returns:
        - DocumentCleaner: 可移除網頁框架、Markdown 包裝與重複行的清理器。
    """

    BOILERPLATE_PATTERNS = (
        r"^(?:accept|manage)\s+(?:all\s+)?cookies",
        r"^cookie\s+(?:policy|preferences)",
        r"^privacy\s+policy$",
        r"^terms\s+(?:of\s+service|and\s+conditions)$",
        r"^(?:sign|log)\s+in$",
        r"^subscribe(?:\s+now)?$",
        r"^skip\s+to\s+(?:main\s+)?content$",
        r"^all\s+rights\s+reserved\.?$",
    )
    HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")
    MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
    MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\((?:[^()]|\([^)]*\))*\)")
    BARE_URL_RE = re.compile(r"https?://\S+")
    CODE_FENCE_RE = re.compile(r"^\s*```.*?$", flags=re.MULTILINE)
    MARKDOWN_PREFIX_RE = re.compile(
        r"^\s{0,3}(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)"
    )
    TABLE_SEPARATOR_RE = re.compile(
        r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$"
    )

    def __init__(self, *, min_line_chars: int = 2) -> None:
        self.min_line_chars = max(1, min_line_chars)
        self._boilerplate = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in self.BOILERPLATE_PATTERNS
        ]

    def clean(self, content: str) -> str:
        """
        清理單份網頁內容。

        Args:
            - content: HTML、Markdown 或純文字。

        Returns:
            - str: 保留段落結構的乾淨正文。
        """
        text = str(content or "")
        if not text.strip():
            return ""
        if self.HTML_TAG_RE.search(text):
            parser = _ReadableHTMLParser()
            parser.feed(text)
            text = parser.text()

        text = html.unescape(text)
        text = self.MARKDOWN_IMAGE_RE.sub(" ", text)
        text = self.MARKDOWN_LINK_RE.sub(r"\1", text)
        text = self.BARE_URL_RE.sub(" ", text)
        text = self.CODE_FENCE_RE.sub("", text)
        text = text.replace("\u00a0", " ").replace("\u200b", "")

        cleaned_lines: list[str] = []
        previous_key = ""
        for raw_line in text.splitlines():
            line = self.MARKDOWN_PREFIX_RE.sub("", raw_line)
            line = re.sub(r"\s+", " ", line).strip(" |")
            if not line or self.TABLE_SEPARATOR_RE.match(line):
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if len(line) < self.min_line_chars:
                continue
            if any(pattern.match(line) for pattern in self._boilerplate):
                continue
            key = normalize_text(line).lower()
            if key and key == previous_key:
                continue
            cleaned_lines.append(line)
            previous_key = key

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_title(self, title: str) -> str:
        """
        清理 corpus title。

        Args:
            - title: Search result 或網頁標題。

        Returns:
            - str: 單行標題。
        """
        return normalize_text(self.clean(title)).strip()


__all__ = ["DocumentCleaner"]
