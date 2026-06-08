from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, SearchSourceCandidate
from .seer.helpfulness_expert import HelpfulnessExpert
from .seer.ngram_deduplicate import NgramDeduplicator
from .seer.page_content_fetcher import PageContentFetcher
from .seer.source_filter import SourceFilter
from ..next_hop_query.rag_labeler import EfficientRAGLabelerAdapter

@dataclass
class SourceChunk:
    """
    儲存從單一 search source 切出的 evidence chunk。

    Args:
        - chunk_id: chunk id。
        - source_id: 來源 source id。
        - query_id: 來源 query id。
        - title: source title。
        - url: source URL。
        - text: chunk 文字。
        - source: 原始 SearchSourceCandidate。

    Returns:
        - SourceChunk: 可交給 helpfulness / labeler 判斷的 chunk。
    """

    chunk_id: str
    source_id: str
    query_id: str
    title: str
    url: str
    text: str
    source: SearchSourceCandidate | None = None


@dataclass
class HelpfulnessSignal:
    """
    儲存 Helpfulness Expert 對 chunk 的判斷。

    Args:
        - chunk_id: chunk id。
        - useful_probability: evidence useful probability。
        - ratio: 預留給 P(with evidence) / P(without evidence)。
        - threshold: useful threshold。
        - passed: 是否通過 threshold。
        - method: 使用的判斷方法。

    Returns:
        - HelpfulnessSignal: helpfulness gate 結果。
    """

    chunk_id: str
    useful_probability: float
    ratio: float | None
    threshold: float
    passed: bool
    method: str


@dataclass
class UsefulLabeledChunk:
    """
    儲存 EfficientRAG labeler 後的 useful / useless chunk。

    Args:
        - chunk: 原始 SourceChunk。
        - helpfulness: HelpfulnessSignal。
        - label: useful 或 useless。
        - kept_tokens: 被保留的 token。
        - dropped_tokens: 被捨棄的 token。

    Returns:
        - UsefulLabeledChunk: source analysis 的 labeled chunk。
    """

    chunk: SourceChunk
    label: str
    kept_tokens: list[str]
    dropped_tokens: list[str]
    helpfulness: HelpfulnessSignal | None = None


@dataclass
class SourceUsefulnessResult:
    """
    Source analysis 回傳給 EvidenceSearcher 的結果。

    Args:
        - sources: 通過 hard filter 的 sources。
        - stop_query: 是否達到 helpfulness threshold。
        - best_helpfulness: 本批最高 helpfulness。
        - threshold: helpfulness threshold。
        - fetched_pages: full-page fetch 數量。

    Returns:
        - SourceUsefulnessResult: source analysis pipeline 結果。
    """

    sources: list[SearchSourceCandidate]
    stop_query: bool = False
    best_helpfulness: float = 0.0
    threshold: float = 0.6
    fetched_pages: int = 0


class SourceAnalysis:
    """
    Source Usefulness Pipeline。

    Args:
        - source_filter: hard source filter。
        - page_content_fetcher: full-page fetcher。
        - helpfulness_expert: SEER helpfulness expert。
        - labeler: EfficientRAG labeler adapter。
        - deduplicator: SEER n-gram deduplicator。
        - helpfulness_threshold: useful evidence early-stop threshold。
        - min_useful_chunks: 達成 stop_query 所需最少 useful chunks。

    Returns:
        - SourceAnalysis: source analysis pipeline。
    """

    def __init__(
        self,
        *,
        source_filter: SourceFilter | None = None,
        page_content_fetcher: PageContentFetcher | None = None,
        helpfulness_expert: HelpfulnessExpert | None = None,
        labeler: EfficientRAGLabelerAdapter | None = None,
        deduplicator: NgramDeduplicator | None = None,
        helpfulness_threshold: float = 0.6,
        min_useful_chunks: int = 1,
        min_chunk_chars: int = 40,
        duplicate_threshold: float = 0.82,
    ) -> None:
        self.source_filter = source_filter or SourceFilter()
        self.page_content_fetcher = page_content_fetcher or PageContentFetcher()
        self.helpfulness_expert = helpfulness_expert or HelpfulnessExpert()
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
        對 search sources 執行 hard filter、helpfulness gate、labeling 與 dedup。

        Args:
            - question: 原始問題。
            - sources: search tool 回傳 sources。
            - query_text_by_id: query id 到 query 文字的對應。
            - fetch_limit: 最多標記幾個 source 抓全文。
            - max_pages: 最多實際抓取全文頁數。
            - max_evidence_items: 最多保留 evidence items。
            - max_chars_per_item: 單一 evidence 最長字元數。
            - max_candidates: 候選答案上限，目前 source analysis 不主動抽候選。

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

        useful_chunks = [item for item in labeled_chunks if item.label == "useful"]
        useless_chunks = [item for item in labeled_chunks if item.label != "useful"]
        deduped_useful = self._dedupe_labeled_chunks(useful_chunks)
        scored_useful = self._score_labeled_chunks(question=question, chunks=deduped_useful)
        scored_useful.sort(
            key=lambda item: self._helpfulness_score(item),
            reverse=True,
        )
        evidence_items = self._to_evidence_items(
            scored_useful[:max_evidence_items],
            max_chars_per_item=max_chars_per_item,
        )
        best_helpfulness = max(
            (self._helpfulness_score(item) for item in scored_useful),
            default=0.0,
        )
        stop_query = best_helpfulness >= self.helpfulness_threshold and len(scored_useful) >= self.min_useful_chunks
        diagnostics = {
            "source_pipeline": "hard_filter->fetch->chunk->efficientrag_labeler->seer_dedup->helpfulness_scoring->evidence_conversion",
            "source_count": len(sources),
            "filtered_source_count": len(filtered_sources),
            "blocked_source_count": len(blocked_sources),
            "chunk_count": len(chunks),
            "useful_chunk_count": len(scored_useful),
            "useless_chunk_count": len(useless_chunks),
            "best_helpfulness": best_helpfulness,
            "helpfulness_threshold": self.helpfulness_threshold,
            "stop_query": stop_query,
        }

        self.last_blocked_sources = blocked_sources
        self.last_evidence_items = evidence_items
        self.last_candidates = []
        self.last_rejected_candidates = []
        self.last_useful_chunks = scored_useful
        self.last_useless_chunks = useless_chunks
        self.last_diagnostics = diagnostics

        return SourceUsefulnessResult(
            sources=filtered_sources,
            stop_query=stop_query,
            best_helpfulness=best_helpfulness,
            threshold=self.helpfulness_threshold,
            fetched_pages=fetched_pages,
        )

    def build(self, **kwargs: Any) -> SourceUsefulnessResult:
        """
        以 builder 形式執行 source analysis。

        Args:
            - kwargs: analyze() 參數。

        Returns:
            - SourceUsefulnessResult: source analysis 結果。
        """
        return self.analyze(**kwargs)

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

    def _score_helpfulness(self, *, question: str, chunk: SourceChunk) -> HelpfulnessSignal:
        score = self.helpfulness_expert.score(question=question, evidence=chunk.text)
        method = "seer_helpfulness_expert"
        if score is None:
            score = self._fallback_helpfulness(question=question, evidence=chunk.text)
            method = "keyword_overlap_fallback"
        score = max(0.0, min(float(score), 1.0))
        return HelpfulnessSignal(
            chunk_id=chunk.chunk_id,
            useful_probability=score,
            ratio=None,
            threshold=self.helpfulness_threshold,
            passed=score >= self.helpfulness_threshold,
            method=method,
        )

    def _score_labeled_chunks(
        self,
        *,
        question: str,
        chunks: list[UsefulLabeledChunk],
    ) -> list[UsefulLabeledChunk]:
        for item in chunks:
            item.helpfulness = self._score_helpfulness(question=question, chunk=item.chunk)
        return chunks

    def _helpfulness_score(self, item: UsefulLabeledChunk) -> float:
        if item.helpfulness is None:
            return 0.0
        return item.helpfulness.useful_probability

    def _fallback_helpfulness(self, *, question: str, evidence: str) -> float:
        question_terms = self._keywords(question)
        evidence_terms = self._keywords(evidence)
        if not question_terms or not evidence_terms:
            return 0.0
        overlap = len(question_terms & evidence_terms) / len(question_terms)
        return max(0.0, min(overlap, 1.0))

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
            helpfulness = item.helpfulness
            useful_probability = helpfulness.useful_probability if helpfulness is not None else 0.0
            helpfulness_method = helpfulness.method if helpfulness is not None else "not_scored"
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
                    helpfulness_score=useful_probability,
                    evidence_quality=useful_probability,
                    cleaning_reasons=[
                        f"label:{item.label}",
                        "seer_dedup",
                        f"helpfulness:{useful_probability:.2f}",
                        helpfulness_method,
                    ],
                )
            )
        return evidence_items

    def _keywords(self, text: str) -> set[str]:
        stopwords = EfficientRAGLabelerAdapter.STOPWORDS
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower())
            if token not in stopwords and len(token) > 2
        }



SEERBuildResult = SourceUsefulnessResult
SEERBuilder = SourceAnalysis

__all__ = [
    "EfficientRAGLabelerAdapter",
    "HelpfulnessSignal",
    "SEERBuildResult",
    "SEERBuilder",
    "SourceAnalysis",
    "SourceChunk",
    "SourceUsefulnessResult",
    "UsefulLabeledChunk",
]
