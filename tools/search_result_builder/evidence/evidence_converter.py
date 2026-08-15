from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore

from ..config import EvidenceTier
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
    grounded_fact_count: int = 0
    answer_bound_fact_count: int = 0
    direct_fact_count: int = 0
    promoted_contract_count: int = 0
    orphan_direct_fact_count: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    relaxed_candidate_count: int = 0
    relaxed_selected_count: int = 0
    relaxed_bucket_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _CandidateEvidence:
    item: dict[str, Any]
    bucket: str
    bucket_priority: int
    order: int
    diversity_key: str
    match_score: float = 0.0


@dataclass(frozen=True)
class _DiverseSelection:
    chosen: list[_CandidateEvidence]
    dropped_duplicates: int


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
    # Relaxed buckets carry no direct_contracts, so the evidence support
    # checker can never treat them as verified answer support; they exist to
    # give Stage1 agents real retrieved passages instead of empty context.
    BUCKET_GROUNDED_FACT = "GROUNDED_FACT"
    BUCKET_SPAN_SUPPORT = "SPAN_SUPPORT"
    BUCKET_RELEVANCE_FALLBACK = "RELEVANCE_FALLBACK"
    BUCKET_PRIORITY = {
        BUCKET_ANSWER_COMPATIBLE: 0,
        BUCKET_GROUNDED_FACT: 1,
        BUCKET_SPAN_SUPPORT: 2,
        BUCKET_RELEVANCE_FALLBACK: 3,
    }
    RELAXED_BUCKETS = frozenset(
        {BUCKET_GROUNDED_FACT, BUCKET_SPAN_SUPPORT, BUCKET_RELEVANCE_FALLBACK}
    )

    def __init__(
        self,
        *,
        span_builder: SpanBuilder | None = None,
        max_items: int = 8,
        # 520 cut 59% of references mid-content on level1_final_16: their
        # untruncated length is 720 on average, 600 median, 956 at p90. At 900
        # the share arriving complete goes from 37% to 86%, which is what the
        # Agents read -- a reference ending "Hiccup would have had to carry 8
        # ..." is what the old cap produced. The p95 is 1381 and the longest
        # 20,584, so a cap is still needed; this one just sits above the bulk
        # of the distribution instead of through the middle of it.
        max_chars: int = 900,
        duplicate_overlap_threshold: float = 0.86,
        max_relaxed_references: int = 8,
    ) -> None:
        self.span_builder = span_builder or SpanBuilder(max_context_chars=max_chars)
        self.max_items = max(1, max_items)
        # Read-only references get their own budget: they carry no support
        # authority, so a few extra passages cost only prompt length.
        self.max_relaxed_references = max(1, max_relaxed_references)
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
            grounded_fact_count=0,
            answer_bound_fact_count=0,
            direct_fact_count=0,
            promoted_contract_count=0,
            orphan_direct_fact_count=0,
            rejection_reasons={},
        )
        # Ranked read-only passages from the most recent conversion. Populated
        # even when strict evidence exists so the caller can decide how much
        # extra context to show.
        self.last_relaxed_references: list[dict[str, Any]] = []

    def convert_web_retrieval_output(
        self,
        output_dict: dict[str, Any],
        *,
        contract: EvidenceSelectionContract | None = None,
        question: str = "",
        fact_store: TaskFactStore | None = None,
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
        relaxed_candidates: list[_CandidateEvidence] = []
        contract_terms = self._contract_terms(evidence_contract)
        dropped_duplicates = 0
        grounded_fact_count = 0
        answer_bound_fact_count = 0
        direct_fact_count = 0
        promoted_contract_count = 0
        orphan_direct_fact_count = 0
        rejection_reasons: Counter[str] = Counter()
        order = 0

        for round_info in rounds:
            round_index = int(round_info.get("round_index", 0) or 0)
            query = normalize_text(round_info.get("query", ""))
            for document in round_info.get("documents") or []:
                if not isinstance(document, dict):
                    continue
                if document.get("duplicate"):
                    rejection_reasons["duplicate_document"] += 1
                    continue
                (
                    grounded,
                    answer_bound,
                    direct_facts,
                    promoted_contracts,
                    orphan_direct_facts,
                ) = self._collect_document_facts(
                    document,
                    fact_store=fact_store,
                )
                grounded_fact_count += grounded
                answer_bound_fact_count += answer_bound
                direct_fact_count += direct_facts
                promoted_contract_count += promoted_contracts
                orphan_direct_fact_count += orphan_direct_facts
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
                else:
                    rejection_reasons[
                        self._document_rejection_reason(document)
                    ] += 1
                    relaxed = self._relaxed_candidate_from_document(
                        document,
                        contract=evidence_contract,
                        contract_terms=contract_terms,
                        round_index=round_index,
                        query=query,
                        order=order,
                    )
                    if relaxed is not None:
                        relaxed_candidates.append(relaxed)

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.bucket_priority,
                candidate.order,
            ),
        )
        selected = self._select_diverse(
            ranked,
            domain_cap=3,
            limit=self.max_items,
        )
        dropped_duplicates += selected.dropped_duplicates

        evidence_items = []
        for index, candidate in enumerate(selected.chosen, start=1):
            item = dict(candidate.item)
            item["evidence_id"] = f"E{index}"
            item["evidence_tier"] = EvidenceTier.ANSWER_GROUNDED.value
            item["support_eligible"] = True
            item["verification_ready"] = True
            item["grounding_status"] = "grounded"
            item["promotion_reason"] = "direct_evidence_contract"
            evidence_items.append(item)

        # Relaxed passages never become E-ID evidence. Strict conversion
        # failing does not make an unverified passage trustworthy: on saved
        # GAIA runs, letting relaxed passages carry support turned generic
        # question-vocabulary overlap into "bridge" support for whatever the
        # agents guessed. They are still worth reading, so they are ranked
        # here and handed to the caller as read-only references (R-ID).
        relaxed_selected = self._select_relaxed_references(
            relaxed_candidates,
            contract=evidence_contract,
        )
        relaxed_bucket_counts: Counter[str] = Counter()
        self.last_relaxed_references = []
        for index, candidate in enumerate(relaxed_selected.chosen, start=1):
            item = dict(candidate.item)
            item["reference_id"] = f"R{index}"
            item["evidence_id"] = ""
            item["evidence_tier"] = EvidenceTier.RELAXED_CONTEXT.value
            item["support_eligible"] = False
            item["verification_ready"] = False
            item["relaxed"] = True
            item["grounding_status"] = "ungrounded"
            item["promotion_reason"] = ""
            # Span fields are what the support checker mines for intermediate
            # values; a read-only reference must not expose them.
            item["matched_terms"] = []
            item["useful_spans"] = []
            item["answer_support_spans"] = []
            item["compatible_spans"] = []
            item["bridge_spans"] = []
            item["direct_contracts"] = []
            item["semantic_facts"] = []
            relaxed_bucket_counts[candidate.bucket] += 1
            self.last_relaxed_references.append(item)

        self.last_diagnostics = EvidenceConversionDiagnostics(
            candidate_count=len(candidates),
            selected_count=len(evidence_items),
            fallback_used=bool(self.last_relaxed_references and not evidence_items),
            dropped_duplicates=dropped_duplicates,
            grounded_fact_count=grounded_fact_count,
            answer_bound_fact_count=answer_bound_fact_count,
            direct_fact_count=direct_fact_count,
            promoted_contract_count=promoted_contract_count,
            orphan_direct_fact_count=orphan_direct_fact_count,
            rejection_reasons=dict(rejection_reasons),
            relaxed_candidate_count=len(relaxed_candidates),
            relaxed_selected_count=sum(relaxed_bucket_counts.values()),
            relaxed_bucket_counts=dict(relaxed_bucket_counts),
        )
        return evidence_items

    def _select_relaxed_references(
        self,
        relaxed_candidates: list[_CandidateEvidence],
        *,
        contract: EvidenceSelectionContract,
    ) -> _DiverseSelection:
        """Pick read-only passages by question-term coverage.

        The per-domain cap stays at two on purpose. A looser cap does surface
        one more answer passage across the saved runs, but it does so by
        admitting near-duplicate chunks of the same page, and Stage1 context is
        already truncated on most search tasks — so the extra passages evict
        other context on tasks that were already answered correctly. Widening
        this budget needs page-constrained retrieval (which picks a *better*
        passage rather than more of them), not a bigger cap.
        """
        del contract
        return self._select_diverse(
            sorted(
                relaxed_candidates,
                key=lambda candidate: (-candidate.match_score, candidate.order),
            ),
            domain_cap=2,
            limit=self.max_relaxed_references,
        )

    def _select_diverse(
        self,
        ranked: list[_CandidateEvidence],
        *,
        domain_cap: int,
        limit: int,
    ) -> _DiverseSelection:
        """Take the top candidates while capping how many come from one domain."""
        chosen: list[_CandidateEvidence] = []
        seen_domains: dict[str, int] = {}
        dropped_duplicates = 0
        for candidate in ranked:
            if len(chosen) >= limit:
                break
            if self._is_duplicate(candidate.item, [item.item for item in chosen]):
                dropped_duplicates += 1
                continue
            domain = candidate.diversity_key
            if domain:
                domain_count = seen_domains.get(domain, 0)
                if domain_count >= domain_cap and len(chosen) >= 3:
                    continue
                seen_domains[domain] = domain_count + 1
            chosen.append(candidate)
        return _DiverseSelection(chosen=chosen, dropped_duplicates=dropped_duplicates)

    def _collect_document_facts(
        self,
        document: dict[str, Any],
        *,
        fact_store: TaskFactStore | None,
    ) -> tuple[int, int, int, int, int]:
        grounded: list[EvidenceFact] = []
        answer_bound = 0
        for value in list(document.get("semantic_facts") or []):
            if not isinstance(value, dict):
                continue
            fact = EvidenceFact.from_dict(value)
            if fact.grounding_status != "grounded":
                continue
            grounded.append(fact)
            if (
                fact.role == "ANSWER_SUPPORT"
                and fact.qualifiers.get("answer_binding") == "direct"
            ):
                answer_bound += 1
        if fact_store is not None:
            fact_store.extend(grounded)
        contracts = [
            item
            for item in list(document.get("direct_contracts") or [])
            if isinstance(item, dict)
        ]
        contract_fact_ids = {
            normalize_text(str(item.get("fact_id") or ""))
            for item in contracts
            if normalize_text(str(item.get("fact_id") or ""))
        }
        direct_facts = [
            fact
            for fact in grounded
            if fact.role == "ANSWER_SUPPORT"
            and fact.qualifiers.get("answer_binding") == "direct"
        ]
        promoted_contract_count = sum(
            normalize_text(str(item.get("contract_method") or ""))
            == "grounded_answer_value_promotion"
            for item in contracts
        )
        orphan_direct_count = sum(
            bool(fact.fact_id and fact.fact_id not in contract_fact_ids)
            for fact in direct_facts
        )
        return (
            len(grounded),
            answer_bound,
            len(direct_facts),
            promoted_contract_count,
            orphan_direct_count,
        )

    def _document_rejection_reason(self, document: dict[str, Any]) -> str:
        text = normalize_text(document.get("text", ""))
        if not text:
            return "source_content_empty"
        contracts = [
            item
            for item in list(document.get("direct_contracts") or [])
            if isinstance(item, dict)
        ]
        if not contracts:
            facts = [
                item
                for item in list(document.get("semantic_facts") or [])
                if isinstance(item, dict)
                and normalize_text(item.get("grounding_status")) == "grounded"
            ]
            if any(
                normalize_text(str(item.get("role") or "")).upper()
                == "ANSWER_SUPPORT"
                and normalize_text(
                    str(dict(item.get("qualifiers") or {}).get("answer_binding") or "")
                )
                == "direct"
                for item in facts
            ):
                return "orphan_direct_fact"
            return "bridge_only" if facts else "no_grounded_fact"
        if not any(
            normalize_text(item.get("answer_span", ""))
            and normalize_text(item.get("answer_span", "")).casefold()
            in text.casefold()
            for item in contracts
        ):
            return "answer_span_not_grounded"
        return "direct_contract_invalid"

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
            "selection_reason": (
                "grounded_answer_value_promotion"
                if any(
                    normalize_text(str(contract_item.get("contract_method") or ""))
                    == "grounded_answer_value_promotion"
                    for contract_item in direct_contracts
                )
                else "direct_evidence_contract"
            ),
        }
        return _CandidateEvidence(
            item=item,
            bucket=bucket,
            bucket_priority=self.BUCKET_PRIORITY.get(bucket, 99),
            order=order,
            diversity_key=self._domain(url),
        )

    def _contract_terms(self, contract: EvidenceSelectionContract) -> set[str]:
        """Informative terms from the question and the selection contract."""
        terms: set[str] = set()
        terms.update(self._important_terms(contract.question))
        terms.update(self._important_terms(contract.answer_target))
        terms.update(self._important_terms(contract.answer_requirement))
        for entry in contract.must_include:
            terms.update(self._important_terms(entry))
        return terms

    def _relaxed_candidate_from_document(
        self,
        document: dict[str, Any],
        *,
        contract: EvidenceSelectionContract,
        contract_terms: set[str],
        round_index: int,
        query: str,
        order: int,
    ) -> _CandidateEvidence | None:
        """Build an unverified passage candidate for a strict-rejected document.

        Tiers mirror the trust the upstream pipeline expressed in the
        document: grounded semantic facts, then labeler-useful documents,
        then any passage overlapping the question terms. The emitted item
        never carries direct_contracts, so downstream support checking keeps
        treating it as unverified context.
        """
        text = normalize_text(document.get("text", ""))
        if not text:
            return None
        text_terms = self._important_terms(text)
        overlap = contract_terms & text_terms
        grounded_facts = [
            dict(fact)
            for fact in list(document.get("semantic_facts") or [])
            if isinstance(fact, dict)
            and normalize_text(str(fact.get("grounding_status") or "")) == "grounded"
        ]
        labeler_useful = (
            normalize_text(str(document.get("label") or "")).casefold() == "useful"
            or bool(document.get("grounded_labeler_spans"))
        )
        if grounded_facts:
            bucket = self.BUCKET_GROUNDED_FACT
            selection_reason = "relaxed_grounded_fact"
        elif labeler_useful:
            bucket = self.BUCKET_SPAN_SUPPORT
            selection_reason = "relaxed_labeler_useful"
        elif overlap:
            bucket = self.BUCKET_RELEVANCE_FALLBACK
            selection_reason = "relaxed_relevance_fallback"
        else:
            return None

        # Focus the passage on contract terms when possible; grounded fact
        # spans anchor the context for fact-backed documents.
        anchor_terms: list[str] = []
        for fact in grounded_facts:
            for span in list(fact.get("evidence_spans") or []):
                span_text = normalize_text(str(span))
                if span_text and span_text.casefold() in text.casefold():
                    anchor_terms.append(span_text)
        anchor_terms.extend(sorted(overlap))
        built_context, matched_spans = self.span_builder.build_context(
            text,
            anchor_terms[:16],
            fallback_chars=self.max_chars,
        )
        evidence_text = self._truncate(built_context or text, self.max_chars)
        if not evidence_text:
            return None

        retrieval_score = self._safe_float(document.get("retrieval_score"))
        selection_score = self._safe_float(document.get("selection_score"))
        coverage = len(overlap) / max(1, len(contract_terms))
        match_score = coverage + max(retrieval_score, selection_score)
        url = normalize_text(document.get("url", ""))
        item = {
            "evidence_id": "",
            "source_id": normalize_text(document.get("document_id", "")),
            "query_id": f"R{round_index}" if round_index else "R",
            "title": normalize_text(document.get("title", "")),
            "text": evidence_text,
            "url": url,
            "matched_terms": sorted(overlap)[:16],
            "matched_spans": self._span_dicts(matched_spans),
            "retrieval_score": retrieval_score,
            "sequence_tag": normalize_text(document.get("sequence_tag", "")),
            "label": normalize_text(document.get("label", "")),
            "useful_spans": [],
            "answer_support_spans": [],
            "bridge_spans": [],
            "direct_contracts": [],
            "semantic_facts": grounded_facts[:16],
            "span_roles": list(document.get("span_roles") or [])[:24],
            "support_level": "",
            "answer_requirement": contract.answer_requirement,
            "answer_target": contract.answer_target,
            "must_include": list(contract.must_include),
            "compatible_spans": [],
            "compatibility_results": [],
            "evidence_bucket": bucket,
            "valid_for_next_hop": False,
            "round_index": round_index,
            "retrieval_query": query,
            "selection_reason": selection_reason,
            "relaxed": True,
        }
        return _CandidateEvidence(
            item=item,
            bucket=bucket,
            bucket_priority=self.BUCKET_PRIORITY.get(bucket, 99),
            order=order,
            diversity_key=self._domain(url),
            match_score=match_score,
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
