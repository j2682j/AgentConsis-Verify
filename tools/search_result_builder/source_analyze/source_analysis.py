from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, SearchSourceCandidate
from .rag_labeler import EfficientRAGLabelerAdapter
from .seer.ngram_deduplicate import NgramDeduplicator
from .seer.page_content_fetcher import PageContentFetcher
from .seer.source_filter import SourceFilter


@dataclass
class SourceChunk:
    """
    保存從單一 search source 切出的 evidence chunk。

    Args:
        - chunk_id: chunk id。
        - source_id: 對應的 source id。
        - query_id: 對應的 query id。
        - title: source title。
        - url: source URL。
        - text: chunk 文字。
        - source: 原始 SearchSourceCandidate。

    Returns:
        - SourceChunk: 可交給 labeler 判斷的 chunk。
    """

    chunk_id: str
    source_id: str
    query_id: str
    title: str
    url: str
    text: str
    source: SearchSourceCandidate | None = None


@dataclass
class UsefulLabeledChunk:
    """
    保存 EfficientRAG labeler 標記後的 useful / useless chunk。

    Args:
        - chunk: 原始 SourceChunk。
        - label: useful 或 useless。
        - kept_tokens: labeler 保留的 useful tokens。
        - dropped_tokens: labeler 捨棄的 tokens。

    Returns:
        - UsefulLabeledChunk: source analysis 的 labeled chunk。
    """

    chunk: SourceChunk
    label: str
    kept_tokens: list[str]
    dropped_tokens: list[str]


@dataclass
class SourceUsefulnessResult:
    """
    保存 SourceAnalysis 回傳給 EvidenceSearcher 的結果摘要。

    Args:
        - sources: 通過 hard filter 的 sources。
        - stop_query: 是否已有足夠 useful chunks 可進入 Retrieval Control。
        - best_helpfulness: 保留欄位；helpfulness 關閉時固定為 0。
        - threshold: 保留欄位；helpfulness 關閉時不參與判斷。
        - fetched_pages: full-page fetch 數量。

    Returns:
        - SourceUsefulnessResult: source analysis pipeline 結果。
    """

    sources: list[SearchSourceCandidate]
    stop_query: bool = False
    best_helpfulness: float = 0.0
    threshold: float = 0.0
    fetched_pages: int = 0


class SourceAnalysis:
    """
    執行 source analysis pipeline。

    Args:
        - source_filter: hard source filter。
        - page_content_fetcher: full-page fetcher。
        - labeler: EfficientRAG labeler adapter。
        - deduplicator: SEER n-gram deduplicator。
        - helpfulness_threshold: 保留欄位；目前 helpfulness scoring 關閉。
        - min_useful_chunks: 視為可停止 query 的最低 useful chunk 數量。
        - min_chunk_chars: chunk 最小字元數。
        - duplicate_threshold: useful chunks 去重門檻。

    Returns:
        - SourceAnalysis: source analysis pipeline。
    """

    def __init__(
        self,
        *,
        source_filter: SourceFilter | None = None,
        page_content_fetcher: PageContentFetcher | None = None,
        labeler: EfficientRAGLabelerAdapter | None = None,
        deduplicator: NgramDeduplicator | None = None,
        helpfulness_threshold: float = 0.6,
        min_useful_chunks: int = 1,
        min_chunk_chars: int = 40,
        duplicate_threshold: float = 0.82,
    ) -> None:
        self.source_filter = source_filter or SourceFilter()
        self.page_content_fetcher = page_content_fetcher or PageContentFetcher()
        self.labeler = labeler or EfficientRAGLabelerAdapter()
        self.deduplicator = deduplicator or NgramDeduplicator()
        self.helpfulness_threshold = helpfulness_threshold
        self.min_useful_chunks = max(1, min_useful_chunks)
        self.min_chunk_chars = min_chunk_chars
        self.duplicate_threshold = duplicate_threshold
        self.last_blocked_sources: list[SearchSourceCandidate] = []
        self.last_evidence_items: list[EvidenceItem] = []
        self.last_candidates: list[CandidateAnswer] = []
        self.last_rejected_candidates: list[dict[str, str]] = []
        self.last_useful_chunks: list[UsefulLabeledChunk] = []
        self.last_useless_chunks: list[UsefulLabeledChunk] = []
        self.last_diagnostics: dict[str, Any] = {}

    def analyze(
        self,
        *,
        question: str,
        sources: list[SearchSourceCandidate],
        query_text_by_id: dict[str, str],
        fetch_limit: int,
        max_pages: int,
        max_evidence_items: int = 8,
        max_chars_per_item: int = 420,
        max_candidates: int = 10,
    ) -> SourceUsefulnessResult:
        """
        對 search sources 執行 filter、fetch、chunk、label、dedup 與 evidence conversion。

        Args:
            - question: 原始問題。
            - sources: search tool 回傳 sources。
            - query_text_by_id: query id 到 query 文字的對應。
            - fetch_limit: 最多標記多少 source 抓全文。
            - max_pages: 最多抓取多少完整頁面。
            - max_evidence_items: 最多輸出多少 EvidenceItem。
            - max_chars_per_item: 每個 EvidenceItem 最大文字長度。
            - max_candidates: 保留相容欄位，目前不使用。

        Returns:
            - SourceUsefulnessResult: source analysis 結果。
        """
        del max_candidates
        filtered_sources = self.source_filter.filter_sources(
            sources,
            question=question,
            query_text_by_id=query_text_by_id,
            fetch_limit=fetch_limit,
        )
        fetched_pages = self.page_content_fetcher.fetch_sources(
            filtered_sources,
            max_pages=max_pages,
        )
        blocked_sources = [source for source in sources if source.blocked]

        chunks = self._chunk_sources(filtered_sources)
        labeled_chunks = self._label_chunks(question=question, chunks=chunks)
        useful_chunks = [item for item in labeled_chunks if item.label == "useful"]
        useless_chunks = [item for item in labeled_chunks if item.label != "useful"]
        deduped_useful = self._dedupe_labeled_chunks(useful_chunks)
        evidence_items = self._to_evidence_items(
            deduped_useful[:max_evidence_items],
            max_chars_per_item=max_chars_per_item,
        )
        stop_query = len(deduped_useful) >= self.min_useful_chunks
        diagnostics = {
            "source_pipeline": "hard_filter->fetch->chunk->efficientrag_labeler->seer_dedup->evidence_conversion",
            "source_count": len(sources),
            "filtered_source_count": len(filtered_sources),
            "blocked_source_count": len(blocked_sources),
            "chunk_count": len(chunks),
            "useful_chunk_count": len(deduped_useful),
            "useless_chunk_count": len(useless_chunks),
            "helpfulness_scoring": "disabled",
            "stop_query": stop_query,
        }

        self.last_blocked_sources = blocked_sources
        self.last_evidence_items = evidence_items
        self.last_candidates = []
        self.last_rejected_candidates = []
        self.last_useful_chunks = deduped_useful
        self.last_useless_chunks = useless_chunks
        self.last_diagnostics = diagnostics

        return SourceUsefulnessResult(
            sources=filtered_sources,
            stop_query=stop_query,
            best_helpfulness=0.0,
            threshold=0.0,
            fetched_pages=fetched_pages,
        )

    def build(self, **kwargs: Any) -> SourceUsefulnessResult:
        """
        執行 source analysis。

        Args:
            - kwargs: analyze() 參數。

        Returns:
            - SourceUsefulnessResult: source analysis 結果。
        """
        return self.analyze(**kwargs)

    def _label_chunks(
        self,
        *,
        question: str,
        chunks: list[SourceChunk],
    ) -> list[UsefulLabeledChunk]:
        labeled_chunks: list[UsefulLabeledChunk] = []
        for chunk in chunks:
            label_result = self.labeler.label_text(
                question=question,
                text=chunk.text,
            )
            labeled_chunks.append(
                UsefulLabeledChunk(
                    chunk=chunk,
                    label=label_result.label,
                    kept_tokens=label_result.kept_tokens,
                    dropped_tokens=label_result.dropped_tokens,
                )
            )
        return labeled_chunks

    def _chunk_sources(self, sources: list[SearchSourceCandidate]) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        for source in sources:
            text = source.raw_content or source.snippet
            for chunk_text in self._split_chunks(text):
                cleaned = normalize_text(chunk_text)
                if len(cleaned) < self.min_chunk_chars:
                    continue
                chunks.append(
                    SourceChunk(
                        chunk_id=f"CH{len(chunks) + 1}",
                        source_id=source.source_id,
                        query_id=source.query_id,
                        title=source.title,
                        url=source.url,
                        text=cleaned,
                        source=source,
                    )
                )
        return chunks

    def _split_chunks(self, text: str) -> list[str]:
        cleaned = str(text or "").replace("\\n", "\n")
        cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
        if paragraphs:
            return paragraphs

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            current.append(sentence)
            current_len += len(sentence)
            if current_len >= 420:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _dedupe_labeled_chunks(self, chunks: list[UsefulLabeledChunk]) -> list[UsefulLabeledChunk]:
        selected: list[UsefulLabeledChunk] = []
        for item in chunks:
            duplicate = any(
                self.deduplicator.is_duplicate(
                    item.chunk.text,
                    existing.chunk.text,
                    threshold=self.duplicate_threshold,
                )
                for existing in selected
            )
            if duplicate:
                continue
            selected.append(item)
        return selected

    def _to_evidence_items(
        self,
        chunks: list[UsefulLabeledChunk],
        *,
        max_chars_per_item: int,
    ) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        for index, item in enumerate(chunks, start=1):
            text = item.chunk.text[:max_chars_per_item].strip()
            if len(item.chunk.text) > max_chars_per_item:
                text += " ..."
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"E{index}",
                    source_id=item.chunk.source_id,
                    query_id=item.chunk.query_id,
                    text=text,
                    title=item.chunk.title,
                    url=item.chunk.url,
                    matched_terms=item.kept_tokens[:12],
                    helpfulness_score=0.0,
                    evidence_quality=0.0,
                    cleaning_reasons=[
                        f"label:{item.label}",
                        "seer_dedup",
                        "helpfulness:disabled",
                    ],
                )
            )
        return evidence_items


__all__ = [
    "EfficientRAGLabelerAdapter",
    "SourceAnalysis",
    "SourceChunk",
    "SourceUsefulnessResult",
    "UsefulLabeledChunk",
]
