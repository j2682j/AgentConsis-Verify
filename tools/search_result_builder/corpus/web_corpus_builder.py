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
from ..source_analyze.seer.page_content_fetcher import (
    PageFetchResult,
    fetch_page_content_result,
)
from .collection_record import CollectionRecord
from .collection_record_extractor import CollectionRecordExtractor
from .chunker import DocumentChunker
from .document_cleaner import DocumentCleaner
from .jsonl_exporter import JSONLExporter
from .record_assembler import RecordAssembler
from .record_text_serializer import RecordTextSerializer
from .structured_document_extractor import StructuredDocumentExtractor


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
    record_type: str = "passage"
    record_id: str = ""
    authors: tuple[str, ...] = ()
    date: str = ""
    source: str = ""
    content_url: str = ""
    language: str = ""
    country: str = ""
    content: str = ""
    parent_url: str = ""
    extra_fields: tuple[tuple[str, str], ...] = ()
    extraction_method: str = ""
    content_scope: str = "passage"
    content_complete: bool = False
    content_truncated: bool = False
    original_content_chars: int = 0
    required_content: str = "html_text"
    acquisition_state: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "authors": list(self.authors),
            "date": self.date,
            "source": self.source,
            "content_url": self.content_url,
            "language": self.language,
            "country": self.country,
            "content": self.content,
            "parent_url": self.parent_url,
            "extra_fields": dict(self.extra_fields),
            "extraction_method": self.extraction_method,
            "content_scope": self.content_scope,
            "content_complete": self.content_complete,
            "content_truncated": self.content_truncated,
            "original_content_chars": self.original_content_chars,
            "required_content": self.required_content,
            "acquisition_state": self.acquisition_state,
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
        structured_extractor: StructuredDocumentExtractor | None = None,
        collection_extractor: CollectionRecordExtractor | None = None,
        record_assembler: RecordAssembler | None = None,
        record_serializer: RecordTextSerializer | None = None,
    ) -> None:
        self.cleaner = cleaner or DocumentCleaner()
        self.chunker = chunker or DocumentChunker()
        self.exporter = exporter or JSONLExporter()
        self.deduplicator = deduplicator or NgramDeduplicator(n=3)
        self.duplicate_threshold = max(0.0, min(duplicate_threshold, 1.0))
        self.page_fetcher = page_fetcher or fetch_page_content_result
        self.structured_extractor = structured_extractor or StructuredDocumentExtractor()
        self.record_assembler = record_assembler or RecordAssembler()
        self.collection_extractor = collection_extractor or CollectionRecordExtractor(
            assembler=self.record_assembler,
        )
        self.record_serializer = record_serializer or RecordTextSerializer()

    def build_records(
        self,
        sources: Iterable[SearchSourceCandidate | dict[str, Any]],
        *,
        retrieved_at: str | date | None = None,
        fetch_missing: bool = True,
        max_pages_to_fetch: int | None = None,
        max_fetch_tokens: int = 8000,
        max_chunks_per_url: int = 20,
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
        accepted_record_keys: set[str] = set()
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
            raw_html = source_data["raw_html"]
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
                fetch_result = self._fetch_payload(
                    source_data["url"],
                    max_tokens=max_fetch_tokens,
                )
                content = fetch_result.content
                raw_html = fetch_result.raw_html
                source_data["content_complete"] = fetch_result.is_complete
                source_data["content_truncated"] = fetch_result.truncated
                source_data["original_content_chars"] = (
                    fetch_result.original_char_count
                )
                source_data["final_url"] = fetch_result.final_url
                fetched_pages += 1
            if not content:
                content = source_data["snippet"]

            collection_result = self.collection_extractor.extract(
                raw_html,
                parent_url=source_data["url"],
                source_title=source_data["title"],
                source_kind=source_data["source_kind"],
            )
            if collection_result.records:
                page_index += 1
                page_records = self._collection_records(
                    collection_result.records,
                    page_index=page_index,
                    retrieved_at=retrieved_date,
                    max_records=per_url_limit,
                    accepted_record_keys=accepted_record_keys,
                    record_limit=(
                        None
                        if record_limit is None
                        else max(0, record_limit - len(records))
                    ),
                )
                records.extend(page_records)
                continue

            chunks: list[str] = []
            for unit in self.structured_extractor.extract(content):
                cleaned = self.cleaner.clean(unit.text)
                if not cleaned:
                    continue
                if unit.unit_type == "table_row":
                    chunks.append(cleaned)
                else:
                    chunks.extend(self.chunker.chunk(cleaned))
            unique_chunks: list[str] = []
            chunk_limit_reached = False
            for chunk in chunks:
                if len(unique_chunks) >= per_url_limit:
                    chunk_limit_reached = True
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
            remaining_record_capacity = (
                len(unique_chunks)
                if record_limit is None
                else max(0, record_limit - len(records))
            )
            source_content_complete = bool(
                source_data["content_complete"]
                and not chunk_limit_reached
                and remaining_record_capacity >= len(unique_chunks)
            )
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
                        record_type=self._record_type_for_source(source_data),
                        parent_url=source_data["url"],
                        content_scope=(
                            "full_document"
                            if source_data["content_complete"]
                            else "passage"
                        ),
                        content_complete=source_content_complete,
                        content_truncated=(
                            source_data["content_truncated"]
                            or not source_content_complete
                        ),
                        original_content_chars=source_data["original_content_chars"],
                        required_content=source_data["required_content"],
                        acquisition_state=source_data["acquisition_state"],
                    )
                )
        return records

    def build_enriched_records(
        self,
        record: CorpusRecord | dict[str, Any],
        *,
        max_tokens: int = 5000,
        max_chunks: int = 6,
    ) -> list[CorpusRecord]:
        """選擇性抓取結構化記錄的內容連結並建立關聯 passage。"""
        data = record.to_dict() if isinstance(record, CorpusRecord) else dict(record)
        content_url = str(data.get("content_url") or "")
        if not content_url:
            return []
        fetch_result = self._fetch_payload(content_url, max_tokens=max_tokens)
        cleaned = self.cleaner.clean(fetch_result.content)
        if not cleaned:
            return []
        all_chunks = self.chunker.chunk(cleaned)
        if not all_chunks:
            all_chunks = [cleaned]
        chunks = all_chunks[: max(1, max_chunks)]
        all_content_preserved = len(chunks) == len(all_chunks)
        collection_record = self._collection_from_mapping(data)
        record_id = str(data.get("record_id") or data.get("id") or "record")
        retrieved_at = str(data.get("retrieved_at") or self._retrieved_date(None))
        results: list[CorpusRecord] = []
        for index, chunk in enumerate(chunks, start=1):
            text = self.record_serializer.serialize(
                collection_record,
                content_override=chunk,
            )
            results.append(
                self._to_corpus_record(
                    collection_record,
                    record_id=record_id,
                    passage_id=f"{record_id}-content-{index:03d}",
                    text=text,
                    content=chunk,
                    retrieved_at=retrieved_at,
                    extraction_method=(
                        f"{collection_record.extraction_method}+linked_content"
                    ).strip("+"),
                    content_scope="full_document",
                    content_complete=(
                        fetch_result.is_complete and all_content_preserved
                    ),
                    content_truncated=(
                        fetch_result.truncated or not all_content_preserved
                    ),
                    original_content_chars=fetch_result.original_char_count,
                )
            )
        return results

    def build_jsonl(
        self,
        sources: Iterable[SearchSourceCandidate | dict[str, Any]],
        output_path: str | Path,
        *,
        retrieved_at: str | date | None = None,
        fetch_missing: bool = True,
        max_pages_to_fetch: int | None = None,
        max_fetch_tokens: int = 8000,
        max_chunks_per_url: int = 20,
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
            "raw_html": str(getter("raw_html", "") or ""),
            "blocked": bool(getter("blocked", False)),
            "source_kind": str(getter("source_kind", "web") or "web"),
            "content_complete": bool(getter("content_complete", False)),
            "content_truncated": bool(getter("content_truncated", False)),
            "original_content_chars": int(
                getter("original_content_chars", 0) or 0
            ),
            "final_url": str(getter("final_url", "") or ""),
            "required_content": str(
                getter("required_content", "html_text") or "html_text"
            ),
            "acquisition_state": str(
                getter("acquisition_state", "pending") or "pending"
            ),
        }

    def _record_type_for_source(self, source_data: dict[str, Any]) -> str:
        required = str(source_data.get("required_content") or "").casefold()
        source_kind = str(source_data.get("source_kind") or "").casefold()
        if required in {"pdf_text", "pdf_figure"}:
            return "pdf_page"
        if required == "transcript":
            return "transcript_segment"
        if required in {"temporal_video", "visual"} or source_kind == "video":
            return "visual_observation"
        if required == "collection_records":
            return "collection_record"
        return "passage"

    def _fetch_payload(self, url: str, *, max_tokens: int) -> PageFetchResult:
        result = self.page_fetcher(url, max_tokens=max_tokens)
        if isinstance(result, PageFetchResult):
            return result
        if result is None:
            return PageFetchResult(content="", method="none", final_url=url)
        content = str(result)
        return PageFetchResult(
            content=content,
            method="custom",
            quality_status="ok" if content else "empty",
            is_complete=bool(content),
            original_char_count=len(content),
            final_url=url,
        )

    def _collection_records(
        self,
        collection_records: list[CollectionRecord],
        *,
        page_index: int,
        retrieved_at: str,
        max_records: int,
        accepted_record_keys: set[str],
        record_limit: int | None,
    ) -> list[CorpusRecord]:
        results: list[CorpusRecord] = []
        for item_index, record in enumerate(collection_records, start=1):
            if len(results) >= max_records:
                break
            if record_limit is not None and len(results) >= record_limit:
                break
            key = self.record_assembler.identity_key(record)
            if not key or key in accepted_record_keys:
                continue
            accepted_record_keys.add(key)
            record_id = f"record-{page_index:03d}-{item_index:03d}"
            texts = self._serialized_record_texts(record)
            for chunk_index, (text, content) in enumerate(texts, start=1):
                if len(results) >= max_records:
                    break
                if record_limit is not None and len(results) >= record_limit:
                    break
                results.append(
                    self._to_corpus_record(
                        record,
                        record_id=record_id,
                        passage_id=f"{record_id}-{chunk_index:03d}",
                        text=text,
                        content=content,
                        retrieved_at=retrieved_at,
                    )
                )
        return results

    def _serialized_record_texts(
        self,
        record: CollectionRecord,
    ) -> list[tuple[str, str]]:
        full_text = self.record_serializer.serialize(record)
        if not record.content or len(full_text) <= self.chunker.max_chars:
            return [(full_text, record.content)]
        chunks = self.chunker.chunk(record.content)
        if not chunks:
            return [(full_text, record.content)]
        return [
            (self.record_serializer.serialize(record, content_override=chunk), chunk)
            for chunk in chunks
        ]

    def _to_corpus_record(
        self,
        record: CollectionRecord,
        *,
        record_id: str,
        passage_id: str,
        text: str,
        content: str,
        retrieved_at: str,
        extraction_method: str | None = None,
        content_scope: str = "collection_record",
        content_complete: bool = False,
        content_truncated: bool = False,
        original_content_chars: int = 0,
    ) -> CorpusRecord:
        return CorpusRecord(
            id=passage_id,
            title=record.title,
            text=text,
            url=record.content_url or record.parent_url,
            retrieved_at=retrieved_at,
            record_type=record.record_type,
            record_id=record_id,
            authors=record.authors,
            date=record.date,
            source=record.source,
            content_url=record.content_url,
            language=record.language,
            country=record.country,
            content=content,
            parent_url=record.parent_url,
            extra_fields=record.extra_fields,
            extraction_method=extraction_method or record.extraction_method,
            content_scope=content_scope,
            content_complete=content_complete,
            content_truncated=content_truncated,
            original_content_chars=original_content_chars,
        )

    def _collection_from_mapping(self, data: dict[str, Any]) -> CollectionRecord:
        authors = data.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        extra_fields = data.get("extra_fields") or {}
        if isinstance(extra_fields, dict):
            extra_fields = list(extra_fields.items())
        return CollectionRecord(
            record_type=str(data.get("record_type") or "article"),
            title=str(data.get("title") or ""),
            authors=tuple(str(item) for item in authors),
            date=str(data.get("date") or ""),
            source=str(data.get("source") or ""),
            content_url=str(data.get("content_url") or ""),
            language=str(data.get("language") or ""),
            country=str(data.get("country") or ""),
            content=str(data.get("content") or ""),
            parent_url=str(data.get("parent_url") or data.get("url") or ""),
            extra_fields=tuple((str(key), str(value)) for key, value in extra_fields),
            extraction_method=str(data.get("extraction_method") or ""),
            record_id=str(data.get("record_id") or ""),
            retrieved_at=str(data.get("retrieved_at") or ""),
        )

    def _retrieved_date(self, value: str | date | None) -> str:
        if value is None:
            return date.today().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return date.fromisoformat(str(value)).isoformat()


__all__ = ["CorpusRecord", "WebCorpusBuilder"]
