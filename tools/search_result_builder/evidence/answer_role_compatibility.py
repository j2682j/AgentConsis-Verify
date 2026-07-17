from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from utils.network_utils import normalize_text

from ..query.semantic_impact import SemanticImpactScorer
from .evidence_contract import EvidenceSelectionContract


@dataclass(frozen=True)
class AnswerRoleCompatibilityResult:
    """
    判斷候選 span/context 是否符合自然語言答案需求。

    Args:
     - compatible: span/context 是否符合 evidence contract。
     - matched_role: 保留相容欄位，現在代表 requirement。
     - target_role_rank: 保留相容欄位，現在固定為 1 或 999。
     - semantic_margin: candidate/context 與 requirement 的語意相似度。
     - guard_passed: 薄型態保護是否通過。
     - reason: 判斷原因。

    Returns:
     - AnswerRoleCompatibilityResult: 可記錄與可報告的 compatibility 結果。

    """

    compatible: bool
    matched_role: str
    target_role_rank: int
    semantic_margin: float
    guard_passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerRoleCompatibilityGate:
    """
    以自然語言答案需求判斷 evidence span/context 是否可支撐答案。

    Args:
     - scorer: 可重用的 encoder embedding scorer。
     - similarity_threshold: requirement 與 candidate/context 的最低語意相似度。

    Returns:
     - AnswerRoleCompatibilityGate: evidence selection 的 semantic requirement gate。

    """

    NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
    YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
    DATE_RE = re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        flags=re.IGNORECASE,
    )
    UNIT_RE = re.compile(
        r"\b(?:m\^?3|m3|km|mi|mile|miles|meter|meters|metre|metres|kg|g|lb|lbs|hour|hours|hr|hrs|min|minute|minutes|"
        r"second|seconds|sec|secs|cm|mm|ft|feet|inch|inches|%|percent|mph|km/h|m/s|sqm|square|cubic)\b",
        flags=re.IGNORECASE,
    )
    COUNT_HINT_RE = re.compile(
        r"\b(how many|number of|count|total|highest number|lowest number|fewest|most|least)\b",
        flags=re.IGNORECASE,
    )
    MEASUREMENT_HINT_RE = re.compile(
        r"\b(volume|m\^?3|m3|distance|height|weight|duration|how much|speed|area|size|capacity|hours?|minutes?|seconds?|km|kg|meter|metre)\b",
        flags=re.IGNORECASE,
    )
    DATE_HINT_RE = re.compile(r"\b(when|date|year|month|day|time|published|released)\b", flags=re.IGNORECASE)
    TEXT_HINT_RE = re.compile(
        r"\b(title|called|named|exactly|setting|scene heading|phrase|wording|label|code|string|location)\b",
        flags=re.IGNORECASE,
    )
    WEAK_SPANS = {
        "appendix",
        "because",
        "comparison",
        "content",
        "during",
        "figure",
        "hence",
        "metadata",
        "obviously",
        "plot",
        "ripped by",
        "size",
        "spoilers",
        "therefore",
        "theme music playing",
        "unknown",
        "using",
        "width",
    }
    METADATA_CONTEXT_RE = re.compile(
        r"\b(?:metadata|datepublished|datemodified|publisher|author\.name|structured data|ray id|cloudflare|security verification)\b",
        flags=re.IGNORECASE,
    )
    AGE_CONTEXT_RE = re.compile(r"\b(?:died|death|age|aged|years old)\b", flags=re.IGNORECASE)

    def __init__(
        self,
        *,
        scorer: SemanticImpactScorer | None = None,
        similarity_threshold: float = 0.52,
    ) -> None:
        self.scorer = scorer or SemanticImpactScorer(max_input_tokens=192)
        self.similarity_threshold = similarity_threshold

    def assess(
        self,
        *,
        contract: EvidenceSelectionContract | None = None,
        question: str = "",
        answer_role: str = "",
        answer_target: str = "",
        must_include: list[str] | None = None,
        span: str,
        context: str = "",
        source_title: str = "",
    ) -> AnswerRoleCompatibilityResult:
        """
        判斷單一 span/context 是否符合自然語言答案需求。

        Args:
         - contract: Evidence selection 的任務契約。
         - question: 原始問題，未提供 contract 時使用。
         - answer_role: 自然語言答案需求，未提供 contract 時使用。
         - answer_target: 答案目標，未提供 contract 時使用。
         - must_include: 題目必要限制，未提供 contract 時使用。
         - span: 候選 evidence span。
         - context: span 所在附近文字。
         - source_title: 來源標題。

        Returns:
         - AnswerRoleCompatibilityResult: compatibility 判斷結果。

        """
        normalized_span = normalize_text(span)
        if not normalized_span:
            return AnswerRoleCompatibilityResult(False, "", 999, 0.0, False, "empty_span")

        evidence_contract = contract or EvidenceSelectionContract.from_parts(
            question=question,
            answer_requirement=answer_role,
            answer_target=answer_target,
            must_include=must_include,
        )
        requirement = self._requirement_text(evidence_contract)
        if not requirement:
            return AnswerRoleCompatibilityResult(False, "", 999, 0.0, False, "empty_requirement")

        guard_passed, guard_reason = self._type_guard(
            requirement=f"{requirement} {evidence_contract.question}",
            span=normalized_span,
            context=context,
            target=evidence_contract.answer_target,
        )
        if not guard_passed:
            return AnswerRoleCompatibilityResult(
                compatible=False,
                matched_role=requirement,
                target_role_rank=999,
                semantic_margin=0.0,
                guard_passed=False,
                reason=guard_reason,
            )

        candidate = self._candidate_representation(
            contract=evidence_contract,
            span=normalized_span,
            context=context,
            source_title=source_title,
        )
        try:
            similarity = self.scorer.semantic_similarities(requirement, [candidate])[0]
        except Exception:
            return AnswerRoleCompatibilityResult(
                compatible=True,
                matched_role=requirement,
                target_role_rank=1,
                semantic_margin=0.0,
                guard_passed=True,
                reason="guard_passed_semantic_unavailable",
            )
        compatible = similarity >= self.similarity_threshold
        return AnswerRoleCompatibilityResult(
            compatible=compatible,
            matched_role=requirement,
            target_role_rank=1 if compatible else 999,
            semantic_margin=round(float(similarity), 6),
            guard_passed=True,
            reason="semantic_requirement_compatible" if compatible else "semantic_requirement_mismatch",
        )

    def _requirement_text(self, contract: EvidenceSelectionContract) -> str:
        answer_requirement = normalize_text(contract.answer_requirement)
        if answer_requirement.casefold() == "unknown":
            answer_requirement = ""
        parts = [
            answer_requirement,
            contract.answer_target,
            " ".join(contract.must_include or []),
        ]
        requirement = normalize_text(" ".join(part for part in parts if normalize_text(part)))
        return requirement or normalize_text(contract.question)

    def _candidate_representation(
        self,
        *,
        contract: EvidenceSelectionContract,
        span: str,
        context: str,
        source_title: str,
    ) -> str:
        return normalize_text(
            "Candidate span: "
            + span
            + "\nContext: "
            + context[:700]
            + "\nSource: "
            + source_title
        )

    def _type_guard(
        self,
        *,
        requirement: str,
        span: str,
        context: str,
        target: str = "",
    ) -> tuple[bool, str]:
        span_text = normalize_text(span)
        context_text = normalize_text(context)
        requirement_text = normalize_text(requirement)
        weak = span_text.casefold().strip(" .,:;!?()[]{}'\"")
        if weak in self.WEAK_SPANS:
            return False, "weak_generic_span"
        if self.METADATA_CONTEXT_RE.search(context_text[:500]):
            return False, "metadata_or_page_chrome_context"
        if self.COUNT_HINT_RE.search(requirement_text):
            if not self.NUMBER_RE.search(span_text):
                return False, "count_requirement_needs_number"
            if self.YEAR_RE.fullmatch(span_text.strip()):
                return False, "count_rejects_standalone_year"
            if self.AGE_CONTEXT_RE.search(context_text[:260]):
                return False, "count_rejects_age_context"
            if target and not self._target_bound(target, context_text):
                return False, "count_not_bound_to_answer_target"
        if self.MEASUREMENT_HINT_RE.search(requirement_text):
            if not (self.NUMBER_RE.search(span_text) and self.UNIT_RE.search(span_text)):
                return False, "measurement_requirement_needs_number_and_unit"
        if self.DATE_HINT_RE.search(requirement_text):
            if not (self.DATE_RE.search(span_text) or self.YEAR_RE.search(span_text)):
                return False, "date_requirement_needs_temporal_pattern"
        if self.TEXT_HINT_RE.search(requirement_text):
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", span_text)
            if not words:
                return False, "text_requirement_empty_words"
            if span_text.isdigit():
                return False, "text_requirement_rejects_pure_number"
            if len(words) == 1 and len(words[0]) < 4:
                return False, "text_requirement_span_too_short"
        return True, "guard_passed"

    def _target_bound(self, answer_target: str, context: str) -> bool:
        terms = [
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", normalize_text(answer_target))
            if len(term) >= 4 and term.casefold() not in SemanticImpactScorer.STOPWORDS
        ]
        if not terms:
            return True
        normalized_context = normalize_text(context).casefold()
        return any(term in normalized_context for term in terms)


__all__ = ["AnswerRoleCompatibilityGate", "AnswerRoleCompatibilityResult"]
