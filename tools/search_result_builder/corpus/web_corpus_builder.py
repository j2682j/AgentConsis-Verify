from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

from utils.network_utils import normalize_text

from ..config import SearchSourceCandidate
from ..source_analyze.seer.ngram_deduplicate import NgramDeduplicator
from ..source_analyze.seer.page_content_fetcher import fetch_page_content
from .chunker import DocumentChunker
from .document_cleaner import DocumentCleaner
from .jsonl_exporter import JSONLExporter


@dataclass(frozen=True)
class CorpusRecord:
    """
    表示 JSONL corpus 中的一個網頁文字 chunk。

    Args:
        - id: 穩定的 page/chunk 識別碼。
        - title: 網頁標題。
        - text: 清理並切分後的正文。
        - url: 原始網頁 URL。
        - retrieved_at: 內容取得日期，格式為 YYYY-MM-DD。

    Returns:
        - CorpusRecord: 可直接匯出為指定 JSONL 格式的資料。
    """

    id: str
    title: str
    text: str
    url: str
    retrieved_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
        }


class WebCorpusBuilder:
    """
    將搜尋來源轉換為清理、切分、去重後的 corpus JSONL。

    Args:
        - cleaner: HTML/Markdown 正文清理器。
        - chunker: 文件切分器。
        - exporter: JSONL 匯出器。
        - deduplicator: 跨頁 chunk 去重器。
        - duplicate_threshold: n-gram 相似度去重門檻。
        - page_fetcher: URL 全文抓取函式，預設沿用 search pipeline fetcher。

    Returns:
        - WebCorpusBuilder: 可建立 records 或直接輸出 JSONL 的整合入口。
    """

    def __init__(
        self,
        *,
        cleaner: DocumentCleaner | None = None,
        chunker: DocumentChunker | None = None,
        exporter: JSONLExporter | None = None,
        deduplicator: NgramDeduplicator | None = None,
        duplicate_threshold: float = 0.9,
        page_fetcher: Callable[..., str | None] | None = None,
    ) -> None:
        self.cleaner = cleaner or DocumentCleaner()
        self.chunker = chunker or DocumentChunker()
        self.exporter = exporter or JSONLExporter()
        self.deduplicator = deduplicator or NgramDeduplicator(n=3)
        self.duplicate_threshold = max(0.0, min(duplicate_threshold, 1.0))
        self.page_fetcher = page_fetcher or fetch_page_content

    def build_records(
        self,
        sources: Iterable[SearchSourceCandidate | dict[str, Any]],
        *,
        retrieved_at: str | date | None = None,
        fetch_missing: bool = True,
        max_pages_to_fetch: int | None = None,
        max_fetch_tokens: int = 8000,
        max_chunks_per_url: int = 12,
        max_records: int | None = 300,
    ) -> list[CorpusRecord]:
        """
        將搜尋來源建立成 corpus records。

        Args:
            - sources: SearchSourceCandidate 或等價 dict。
            - retrieved_at: 取得日期；預設為執行當日。
            - fetch_missing: raw_content 為空時是否抓取 URL。
            - max_pages_to_fetch: 此次最多抓取的網頁數；None 表示不限制。
            - max_fetch_tokens: 單一網頁抓取內容的 token 上限估計。

        Returns:
            - list[CorpusRecord]: 去重後且具有穩定 ID 的 records。
        """
        retrieved_date = self._retrieved_date(retrieved_at)
        records: list[CorpusRecord] = []
        accepted_texts: list[str] = []
        accepted_hashes: set[str] = set()
        fetched_pages = 0
        page_index = 0
        per_url_limit = max(1, max_chunks_per_url)
        record_limit = None if max_records is None else max(1, max_records)

        for source in sources:
            if record_limit is not None and len(records) >= record_limit:
                break
            source_data = self._source_data(source)
            if source_data["blocked"]:
                continue
            content = source_data["raw_content"]
            can_fetch = (
                fetch_missing
                and not content
                and bool(source_data["url"])
                and (
                    max_pages_to_fetch is None
                    or fetched_pages < max_pages_to_fetch
                )
            )
            if can_fetch:
                content = (
                    self.page_fetcher(
                        source_data["url"],
                        max_tokens=max_fetch_tokens,
                    )
                    or ""
                )
                fetched_pages += 1
            if not content:
                content = source_data["snippet"]

            cleaned = self.cleaner.clean(content)
            chunks = self.chunker.chunk(cleaned)
            unique_chunks: list[str] = []
            for chunk in chunks:
                if len(unique_chunks) >= per_url_limit:
                    break
                if self._is_low_quality_chunk(chunk):
                    continue
                content_hash = self._content_hash(chunk)
                if not content_hash or content_hash in accepted_hashes:
                    continue
                if self._is_duplicate(chunk, accepted_texts):
                    continue
                unique_chunks.append(chunk)
                accepted_texts.append(chunk)
                accepted_hashes.add(content_hash)
            if not unique_chunks:
                continue

            page_index += 1
            title = self.cleaner.clean_title(source_data["title"])
            for chunk_index, chunk in enumerate(unique_chunks):
                if record_limit is not None and len(records) >= record_limit:
                    break
                records.append(
                    CorpusRecord(
                        id=f"page-{page_index:03d}-{chunk_index:03d}",
                        title=title,
                        text=chunk,
                        url=source_data["url"],
                        retrieved_at=retrieved_date,
                    )
                )
        return records

    def build_jsonl(
        self,
        sources: Iterable[SearchSourceCandidate | dict[str, Any]],
        output_path: str | Path,
        *,
        retrieved_at: str | date | None = None,
        fetch_missing: bool = True,
        max_pages_to_fetch: int | None = None,
        max_fetch_tokens: int = 8000,
        max_chunks_per_url: int = 12,
        max_records: int | None = 300,
        append: bool = False,
    ) -> int:
        """
        建立 records 並輸出 JSONL。

        Args:
            - sources: SearchSourceCandidate 或等價 dict。
            - output_path: JSONL 輸出路徑。
            - retrieved_at: 取得日期。
            - fetch_missing: 是否抓取缺少 raw_content 的來源。
            - max_pages_to_fetch: 最大抓取頁數。
            - max_fetch_tokens: 每頁最大抓取 token 數估計。
            - append: 是否附加至既有 JSONL。

        Returns:
            - int: 寫入的 JSONL record 數量。
        """
        records = self.build_records(
            sources,
            retrieved_at=retrieved_at,
            fetch_missing=fetch_missing,
            max_pages_to_fetch=max_pages_to_fetch,
            max_fetch_tokens=max_fetch_tokens,
            max_chunks_per_url=max_chunks_per_url,
            max_records=max_records,
        )
        return self.exporter.export(records, output_path, append=append)

    def _is_duplicate(self, text: str, accepted_texts: list[str]) -> bool:
        key = normalize_text(text)
        if not key:
            return True
        return any(
            self.deduplicator.is_duplicate(
                key,
                accepted,
                threshold=self.duplicate_threshold,
            )
            for accepted in accepted_texts
        )

    def _content_hash(self, text: str) -> str:
        normalized = normalize_text(text).casefold()
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_low_quality_chunk(self, text: str) -> bool:
        cleaned = normalize_text(text)
        if len(cleaned) < self.chunker.min_chars:
            return True
        alphanumeric = sum(character.isalnum() for character in cleaned)
        if alphanumeric / max(1, len(cleaned)) < 0.45:
            return True
        words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
        if not words:
            return True
        if len(words) < 12:
            return False
        unique_ratio = len({word.casefold() for word in words}) / len(words)
        return unique_ratio < 0.2

    def _source_data(
        self,
        source: SearchSourceCandidate | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(source, dict):
            getter = source.get
        else:
            getter = lambda key, default=None: getattr(source, key, default)
        return {
            "title": str(getter("title", "") or ""),
            "url": str(getter("url", "") or ""),
            "snippet": str(getter("snippet", getter("content", "")) or ""),
            "raw_content": str(getter("raw_content", "") or ""),
            "blocked": bool(getter("blocked", False)),
        }

    def _retrieved_date(self, value: str | date | None) -> str:
        if value is None:
            return date.today().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return date.fromisoformat(str(value)).isoformat()


__all__ = ["CorpusRecord", "WebCorpusBuilder"]
