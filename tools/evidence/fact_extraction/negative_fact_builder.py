from __future__ import annotations

from dataclasses import replace
import hashlib
import re

from utils.network_utils import normalize_text

from .completeness_contract import AbsenceCheck, CompletenessContract
from .models import EvidenceFact


class NegativeFactBuilder:
    """建立明確否定或完整範圍缺席所支持的負向事實。"""

    _NEGATION_RE = re.compile(
        r"(?:\b(?:not|no|never|without|neither|nor|cannot|can't|doesn't|does not|"
        r"didn't|did not|isn't|is not|wasn't|was not|lacks?|absent|omits?|"
        r"fails? to)\b|(?:不|未|沒有|并未|並未|無|非|缺少|不含|未提及))",
        re.IGNORECASE,
    )

    def validate_explicit(self, fact: EvidenceFact) -> EvidenceFact:
        """拒絕沒有明確否定來源片段、僅由問題語氣推測出的 negative。"""

        if fact.polarity != "negative":
            return fact
        explicit_spans = [
            span
            for span in fact.evidence_spans
            if self._is_explicit_negative(span, fact.object)
        ]
        if not explicit_spans:
            return replace(
                fact,
                grounding_status="invalid",
                qualifiers={
                    **fact.qualifiers,
                    "negative_validation": "explicit_negative_span_required",
                },
            )
        return replace(
            fact,
            evidence_spans=explicit_spans,
            qualifiers={
                **fact.qualifiers,
                "negation_type": "explicit_negative",
                "negative_validation": "exact_negative_span_grounded",
            },
        )

    def from_absence(
        self,
        *,
        check: AbsenceCheck,
        contract: CompletenessContract,
        subject: str,
        relation: str,
        goal_id: str = "",
        source_title: str = "",
        answer_requirement: str = "",
    ) -> EvidenceFact | None:
        if (
            check.status != "absent"
            or not contract.complete
            or check.scope_id != contract.scope_id
        ):
            return None
        subject_value = normalize_text(subject) or normalize_text(source_title)
        relation_value = normalize_text(relation) or "does not contain"
        target = normalize_text(check.target)
        if not subject_value or not target:
            return None
        payload = "\x1f".join(
            [contract.contract_id, subject_value, relation_value, target]
        )
        context = (
            f"Complete scope {contract.scope_id} was checked for {target}; "
            "no occurrence was found."
        )
        return EvidenceFact(
            fact_id="NF-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12],
            subject=subject_value,
            relation=relation_value,
            object=target,
            qualifiers={
                "answer_binding": "direct",
                "answer_requirement": normalize_text(answer_requirement),
                "negation_type": "closed_world_absence",
                "scope_id": contract.scope_id,
                "completeness_contract_id": contract.contract_id,
                "absence_check_id": check.check_id,
            },
            polarity="negative",
            role="ANSWER_SUPPORT",
            goal_id=normalize_text(goal_id),
            evidence_spans=[],
            context=context,
            source_id=contract.source_id,
            source_type=contract.source_type,
            source_title=normalize_text(source_title) or subject_value,
            grounding_status="grounded",
            extraction_method="deterministic_absence_check",
            derivation_type="closed_world_absence",
        )

    def _is_explicit_negative(self, span: str, target: str) -> bool:
        cleaned = normalize_text(span)
        target_key = normalize_text(target).casefold()
        return bool(
            cleaned
            and target_key
            and target_key in cleaned.casefold()
            and self._NEGATION_RE.search(cleaned)
        )


__all__ = ["NegativeFactBuilder"]
