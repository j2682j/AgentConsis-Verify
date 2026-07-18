from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text

from .evidence_contract import EvidenceSelectionContract
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
    bucket: str
    bucket_priority: int
    order: int
    diversity_key: str


class EvidenceConverter:
    """
    將 retrieval trace 轉成短而可支撐答案的 Stage1 EvidenceItem。

    Args:
        - span_builder: 將 labeler useful tokens 還原成原文 span/context 的工具。
        - max_items: 最多輸出幾條 evidence。
        - max_chars: 單條 evidence 最大字元數。
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
    BUCKET_ANSWER_COMPATIBLE = "ANSWER_COMPATIBLE"
    BUCKET_PRIORITY = {
        BUCKET_ANSWER_COMPATIBLE: 0,
    }

    def __init__(
        self,
        *,
        span_builder: SpanBuilder | None = None,
        max_items: int = 8,
        max_chars: int = 520,
        duplicate_overlap_threshold: float = 0.86,
    ) -> None:
        self.span_builder = span_builder or SpanBuilder(max_context_chars=max_chars)
        self.max_items = max(1, max_items)
        self.max_chars = max(120, max_chars)
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
        contract: EvidenceSelectionContract | None = None,
        question: str = "",
    ) -> list[dict[str, Any]]:
        """
        從 WebRetrievalControl output 建立 Stage1 evidence items。

        Args:
            - output_dict: WebRetrievalControl 的 dataclass-to-dict 結果。
            - question: 原始問題，用於 coverage 計算。

        Returns:
            - list[dict[str, Any]]: prompt-ready evidence items。
        """
        evidence_contract = contract or EvidenceSelectionContract.from_parts(question=question)
        retrieval = output_dict.get("retrieval") or {}
        rounds = retrieval.get("rounds") or []
        candidates: list[_CandidateEvidence] = []
        dropped_duplicates = 0
        order = 0

        for round_info in rounds:
            round_index = int(round_info.get("round_index", 0) or 0)
            query = normalize_text(round_info.get("query", ""))
            for document in round_info.get("documents") or []:
                if not isinstance(document, dict):
                    continue
                if document.get("duplicate"):
                    continue
                order += 1
                candidate = self._candidate_from_document(
                    document,
                    contract=evidence_contract,
                    round_index=round_index,
                    query=query,
                    order=order,
                )
                if candidate is not None:
                    candidates.append(candidate)

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.bucket_priority,
                candidate.order,
            ),
        )
        selected: list[_CandidateEvidence] = []
        seen_domains: dict[str, int] = {}
        for candidate in ranked:
            if len(selected) >= self.max_items:
                break
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
            evidence_items.append(item)

        self.last_diagnostics = EvidenceConversionDiagnostics(
            candidate_count=len(candidates),
            selected_count=len(evidence_items),
            fallback_used=False,
            dropped_duplicates=dropped_duplicates,
        )
        return evidence_items

    def _candidate_from_document(
        self,
        document: dict[str, Any],
        *,
        contract: EvidenceSelectionContract,
        round_index: int,
        query: str,
        order: int,
    ) -> _CandidateEvidence | None:
        text = normalize_text(document.get("text", ""))
        if not text:
            return None
        direct_contracts = [
            dict(item)
            for item in list(document.get("direct_contracts") or [])
            if isinstance(item, dict)
            and normalize_text(str(item.get("answer_span") or ""))
            and normalize_text(str(item.get("answer_span") or "")).casefold()
            in text.casefold()
        ]
        if not direct_contracts:
            return None
        direct_fact_ids = {
            normalize_text(str(item.get("fact_id") or ""))
            for item in direct_contracts
            if normalize_text(str(item.get("fact_id") or ""))
        }
        semantic_facts = [
            dict(fact)
            for fact in list(document.get("semantic_facts") or [])
            if isinstance(fact, dict)
            and normalize_text(str(fact.get("grounding_status") or "")) == "grounded"
            and (
                not direct_fact_ids
                or normalize_text(str(fact.get("fact_id") or "")) in direct_fact_ids
            )
        ]
        matched_terms = [
            normalize_text(str(item.get("answer_span") or ""))
            for item in direct_contracts
        ]
        matched_terms = [term for term in matched_terms if term]
        built_context, matched_spans = self.span_builder.build_context(
            text,
            matched_terms,
            fallback_chars=self.max_chars,
        )
        contract_context = " ".join(
            dict.fromkeys(
                normalize_text(str(item.get("context") or ""))
                for item in direct_contracts
                if normalize_text(str(item.get("context") or ""))
            )
        )
        evidence_text = self._truncate(
            contract_context or built_context,
            self.max_chars,
        )
        if not evidence_text:
            return None

        sequence_tag = normalize_text(document.get("sequence_tag", ""))
        compatible_spans = list(matched_terms)
        bucket = self.BUCKET_ANSWER_COMPATIBLE

        retrieval_score = self._safe_float(document.get("retrieval_score"))
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
            "matched_spans": self._span_dicts(matched_spans),
            "retrieval_score": retrieval_score,
            "sequence_tag": sequence_tag,
            "label": normalize_text(document.get("label", "")),
            "useful_spans": matched_terms[:16],
            "answer_support_spans": [
                normalize_text(str(item.get("answer_span") or ""))
                for item in direct_contracts
            ][:16],
            "bridge_spans": [],
            "direct_contracts": direct_contracts[:16],
            "semantic_facts": semantic_facts[:16],
            "span_roles": list(document.get("span_roles") or [])[:24],
            "support_level": normalize_text(document.get("support_level", "")),
            "answer_requirement": contract.answer_requirement,
            "answer_target": contract.answer_target,
            "must_include": list(contract.must_include),
            "compatible_spans": compatible_spans[:8],
            "compatibility_results": [],
            "evidence_bucket": bucket,
            "valid_for_next_hop": False,
            "round_index": round_index,
            "retrieval_query": query,
            "selection_reason": "direct_evidence_contract",
        }
        return _CandidateEvidence(
            item=item,
            bucket=bucket,
            bucket_priority=self.BUCKET_PRIORITY.get(bucket, 99),
            order=order,
            diversity_key=self._domain(url),
        )

    def _span_dicts(self, matched_spans: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for span in matched_spans or []:
            if isinstance(span, dict):
                result.append(dict(span))
                continue
            try:
                result.append(asdict(span))
            except TypeError:
                continue
        return result

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
