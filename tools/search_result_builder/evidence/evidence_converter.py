from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text

from .span_builder import SpanBuilder


@dataclass(frozen=True)
class EvidenceConversionDiagnostics:
    """
    記錄 evidence conversion 的選擇結果。

    Args:
        - candidate_count: retrieval trace 中可考慮的文件數量。
        - selected_count: 最後輸出的 evidence 數量。
        - fallback_used: 是否因為 labeler 沒有 useful span 而使用 retrieval fallback。
        - dropped_duplicates: 因重複內容被丟棄的數量。

    Returns:
        - EvidenceConversionDiagnostics: 可寫入 log 的 conversion 摘要。
    """

    candidate_count: int
    selected_count: int
    fallback_used: bool
    dropped_duplicates: int


@dataclass(frozen=True)
class _CandidateEvidence:
    item: dict[str, Any]
    label_priority: float
    retrieval_score: float
    span_score: float
    coverage_score: float
    diversity_key: str
    fallback: bool

    @property
    def conversion_score(self) -> float:
        return round(
            self.label_priority
            + 0.35 * self.retrieval_score
            + 0.25 * self.span_score
            + 0.15 * self.coverage_score,
            6,
        )


class EvidenceConverter:
    """
    將 retrieval trace 轉成短而可支撐答案的 Stage1 EvidenceItem。

    Args:
        - span_builder: 將 labeler useful tokens 還原成原文 span/context 的工具。
        - max_items: 最多輸出幾條 evidence。
        - max_chars: 單條 evidence 最大字元數。
        - fallback_items: labeler 沒有可用 span 時保留的 retrieval fallback 數量。
        - duplicate_overlap_threshold: evidence sentence lexical overlap 去重門檻。

    Returns:
        - EvidenceConverter: answer-oriented evidence conversion pipeline。
    """

    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
    _WEAK_TERMS = {
        "answer",
        "article",
        "chunk",
        "content",
        "document",
        "evidence",
        "find",
        "give",
        "page",
        "question",
        "search",
        "source",
        "text",
        "unknown",
    }

    def __init__(
        self,
        *,
        span_builder: SpanBuilder | None = None,
        max_items: int = 8,
        max_chars: int = 520,
        fallback_items: int = 2,
        duplicate_overlap_threshold: float = 0.86,
    ) -> None:
        self.span_builder = span_builder or SpanBuilder(max_context_chars=max_chars)
        self.max_items = max(1, max_items)
        self.max_chars = max(120, max_chars)
        self.fallback_items = max(0, fallback_items)
        self.duplicate_overlap_threshold = max(
            0.0,
            min(1.0, duplicate_overlap_threshold),
        )
        self.last_diagnostics = EvidenceConversionDiagnostics(
            candidate_count=0,
            selected_count=0,
            fallback_used=False,
            dropped_duplicates=0,
        )

    def convert_web_retrieval_output(
        self,
        output_dict: dict[str, Any],
        *,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        從 WebRetrievalControl output 建立 Stage1 evidence items。

        Args:
            - output_dict: WebRetrievalControl 的 dataclass-to-dict 結果。
            - question: 原始問題，用於 coverage 計算。

        Returns:
            - list[dict[str, Any]]: prompt-ready evidence items。
        """
        retrieval = output_dict.get("retrieval") or {}
        rounds = retrieval.get("rounds") or []
        question_terms = self._important_terms(question)
        candidates: list[_CandidateEvidence] = []
        dropped_duplicates = 0

        for round_info in rounds:
            round_index = int(round_info.get("round_index", 0) or 0)
            query = normalize_text(round_info.get("query", ""))
            for document in round_info.get("documents") or []:
                if not isinstance(document, dict):
                    continue
                if document.get("duplicate"):
                    continue
                candidate = self._candidate_from_document(
                    document,
                    question_terms=question_terms,
                    round_index=round_index,
                    query=query,
                )
                if candidate is not None:
                    candidates.append(candidate)

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.fallback,
                -candidate.conversion_score,
                -candidate.retrieval_score,
            ),
        )
        selected: list[_CandidateEvidence] = []
        seen_domains: dict[str, int] = {}
        primary_ranked = [candidate for candidate in ranked if not candidate.fallback]
        fallback_ranked = [candidate for candidate in ranked if candidate.fallback]
        selection_pool = primary_ranked or fallback_ranked
        for candidate in selection_pool:
            if len(selected) >= self.max_items:
                break
            if candidate.fallback and len([item for item in selected if item.fallback]) >= self.fallback_items:
                continue
            if self._is_duplicate(candidate.item, [item.item for item in selected]):
                dropped_duplicates += 1
                continue
            domain = candidate.diversity_key
            if domain:
                domain_count = seen_domains.get(domain, 0)
                if domain_count >= 3 and len(selected) >= 3:
                    continue
                seen_domains[domain] = domain_count + 1
            selected.append(candidate)

        evidence_items = []
        for index, candidate in enumerate(selected, start=1):
            item = dict(candidate.item)
            item["evidence_id"] = f"E{index}"
            item["conversion_score"] = candidate.conversion_score
            item["conversion_features"] = {
                "label_priority": candidate.label_priority,
                "retrieval_score": round(candidate.retrieval_score, 6),
                "span_score": round(candidate.span_score, 6),
                "coverage_score": round(candidate.coverage_score, 6),
                "fallback": candidate.fallback,
            }
            evidence_items.append(item)

        self.last_diagnostics = EvidenceConversionDiagnostics(
            candidate_count=len(candidates),
            selected_count=len(evidence_items),
            fallback_used=any(candidate.fallback for candidate in selected),
            dropped_duplicates=dropped_duplicates,
        )
        return evidence_items

    def _candidate_from_document(
        self,
        document: dict[str, Any],
        *,
        question_terms: set[str],
        round_index: int,
        query: str,
    ) -> _CandidateEvidence | None:
        text = normalize_text(document.get("text", ""))
        if not text:
            return None

        matched_terms = [
            normalize_text(str(term or ""))
            for term in document.get("useful_tokens") or []
        ]
        matched_terms = [
            term
            for term in matched_terms
            if self._is_informative_term(term)
        ]
        evidence_text, matched_spans = self.span_builder.build_context(
            text,
            matched_terms,
            fallback_chars=self.max_chars,
        )
        evidence_text = self._truncate(evidence_text or text, self.max_chars)
        if not evidence_text:
            return None

        sequence_tag = normalize_text(document.get("sequence_tag", ""))
        has_strong_terms = any(self._is_strong_term(term) for term in matched_terms)
        label_priority = self._label_priority(
            sequence_tag,
            bool(matched_terms),
            has_strong_terms=has_strong_terms,
        )
        fallback = label_priority <= 0
        if fallback and not self._is_reasonable_fallback(document, evidence_text):
            return None

        retrieval_score = self._safe_float(document.get("retrieval_score"))
        span_score = self._span_score(matched_terms, matched_spans)
        coverage_score = self._coverage_score(evidence_text, question_terms)
        source_id = normalize_text(document.get("document_id", ""))
        title = normalize_text(document.get("title", ""))
        url = normalize_text(document.get("url", ""))
        item = {
            "evidence_id": "",
            "source_id": source_id,
            "query_id": f"R{round_index}" if round_index else "R",
            "title": title,
            "text": evidence_text,
            "url": url,
            "matched_terms": matched_terms[:16],
            "matched_spans": [asdict(span) for span in matched_spans],
            "retrieval_score": retrieval_score,
            "sequence_tag": sequence_tag,
            "label": normalize_text(document.get("label", "")),
            "round_index": round_index,
            "retrieval_query": query,
            "selection_reason": self._selection_reason(
                sequence_tag=sequence_tag,
                matched_terms=matched_terms,
                fallback=fallback,
            ),
        }
        return _CandidateEvidence(
            item=item,
            label_priority=label_priority,
            retrieval_score=retrieval_score,
            span_score=span_score,
            coverage_score=coverage_score,
            diversity_key=self._domain(url),
            fallback=fallback,
        )

    def _label_priority(
        self,
        sequence_tag: str,
        has_terms: bool,
        *,
        has_strong_terms: bool = False,
    ) -> float:
        if sequence_tag == "<FINISH>":
            return 3.0 if has_terms else 2.2
        if sequence_tag == "<CONTINUE>":
            return 2.6 if has_terms else 1.5
        if sequence_tag == "<TERMINATE>":
            return 2.2 if has_strong_terms else 0.0
        return 0.0

    def _selection_reason(
        self,
        *,
        sequence_tag: str,
        matched_terms: list[str],
        fallback: bool,
    ) -> str:
        if fallback:
            return "fallback_retrieval_order"
        if sequence_tag == "<FINISH>":
            return "primary_labeler_sequence"
        if sequence_tag == "<CONTINUE>":
            return "primary_labeler_sequence"
        if sequence_tag == "<TERMINATE>":
            return "secondary_terminate_with_terms"
        if matched_terms:
            return "useful_span_context"
        return "retrieval_context"

    def _span_score(self, matched_terms: list[str], matched_spans: list[Any]) -> float:
        if not matched_terms:
            return 0.0
        strong_terms = sum(1 for term in matched_terms if self._is_strong_term(term))
        span_count = len(matched_spans)
        return min(1.0, 0.2 + 0.18 * strong_terms + 0.12 * span_count)

    def _coverage_score(self, evidence_text: str, question_terms: set[str]) -> float:
        if not question_terms:
            return 0.0
        evidence_terms = self._important_terms(evidence_text)
        if not evidence_terms:
            return 0.0
        return min(1.0, len(question_terms & evidence_terms) / max(1, len(question_terms)))

    def _important_terms(self, text: Any) -> set[str]:
        normalized = normalize_text(text).casefold()
        terms: set[str] = set()
        for match in self._WORD_RE.finditer(normalized):
            term = match.group(0).strip("'_-")
            if not self._is_informative_term(term):
                continue
            terms.add(term)
        return terms

    def _is_informative_term(self, term: str) -> bool:
        cleaned = normalize_text(term).casefold().strip(" ,.;:!?()[]{}'\"")
        if not cleaned or cleaned in self._WEAK_TERMS:
            return False
        if cleaned.isdigit():
            return len(cleaned) >= 2
        if len(cleaned) < 4 and not any(char.isdigit() for char in cleaned):
            return False
        return True

    def _is_strong_term(self, term: str) -> bool:
        cleaned = normalize_text(term).casefold()
        if any(character.isdigit() for character in cleaned):
            return True
        words = [word for word in self._WORD_RE.findall(cleaned) if self._is_informative_term(word)]
        if len(words) >= 2:
            return True
        return len(cleaned) >= 8

    def _is_reasonable_fallback(self, document: dict[str, Any], evidence_text: str) -> bool:
        if len(evidence_text) < 20:
            return False
        return self._safe_float(document.get("retrieval_score")) > 0

    def _is_duplicate(
        self,
        item: dict[str, Any],
        selected: list[dict[str, Any]],
    ) -> bool:
        text = normalize_text(item.get("text", "")).casefold()
        if not text:
            return True
        item_hash = self._content_hash(text)
        item_terms = self._important_terms(text)
        for existing in selected:
            existing_text = normalize_text(existing.get("text", "")).casefold()
            if item_hash == self._content_hash(existing_text):
                return True
            existing_terms = self._important_terms(existing_text)
            if not item_terms or not existing_terms:
                continue
            overlap = len(item_terms & existing_terms) / max(
                1,
                min(len(item_terms), len(existing_terms)),
            )
            if overlap >= self.duplicate_overlap_threshold:
                return True
        return False

    def _content_hash(self, text: str) -> str:
        compact = re.sub(r"\W+", " ", text.casefold()).strip()
        return hashlib.sha1(compact.encode("utf-8")).hexdigest()

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.casefold()
        except Exception:
            return ""

    def _safe_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return max(0.0, min(1.0, number))

    def _truncate(self, text: str, max_chars: int) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."


__all__ = ["EvidenceConversionDiagnostics", "EvidenceConverter"]
