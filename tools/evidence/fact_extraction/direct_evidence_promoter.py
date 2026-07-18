from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Iterable

from utils.network_utils import normalize_text

from .answer_bound_validator import AnswerBoundFactValidator
from .models import EvidenceFact, FactEvidenceRef


@dataclass(frozen=True)
class GroundedAnswerValue:
    """A value-centric answer candidate grounded in one source context."""

    value: str
    evidence_span: str
    context: str
    answer_requirement: str
    answer_target: str
    source_id: str
    source_title: str
    document_id: str
    goal_id: str
    origin_fact_id: str
    value_type: str
    promotion_reason: str
    scope_status: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDiagnostic:
    """Record the ordered gate that accepted or rejected one candidate value."""

    candidate_value: str
    origin_fact_id: str
    accepted: bool
    failed_gate: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DirectEvidencePromotionResult:
    promoted_values: list[GroundedAnswerValue] = field(default_factory=list)
    promoted_facts: list[EvidenceFact] = field(default_factory=list)
    diagnostics: list[PromotionDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted_values": [item.to_dict() for item in self.promoted_values],
            "promoted_facts": [item.to_dict() for item in self.promoted_facts],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _CanonicalValue:
    value: str
    evidence_span: str
    origin_fact_id: str
    polarity: str


class AnswerValueCanonicalizer:
    """Find answer-shaped values that can be restored verbatim from source text."""

    _MEASUREMENT_RE = re.compile(
        r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:m\^?3|m³|km/h|m/s|km|mi|miles?|"
        r"meters?|metres?|cm|mm|ft|feet|inches?|kg|g|lb|lbs|hours?|hrs?|"
        r"minutes?|mins?|seconds?|secs?|%|percent|mph|sqm|square meters?|"
        r"cubic meters?)\b",
        re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])")
    _BOOLEAN_RE = re.compile(r"\b(?:yes|no|true|false)\b", re.IGNORECASE)

    def candidates(
        self,
        *,
        candidate_span: str,
        context: str,
        facts: Iterable[EvidenceFact],
        value_type: str,
    ) -> list[_CanonicalValue]:
        source = normalize_text(context)
        raw_values: list[tuple[str, str, str]] = []
        span = normalize_text(candidate_span).strip(" \"'`.,;:")
        if span:
            raw_values.append((span, "", "positive"))
        for fact in facts:
            raw_values.append((fact.object, fact.fact_id, fact.polarity))
            raw_values.extend(
                (value, fact.fact_id, fact.polarity)
                for value in fact.qualifiers.values()
            )
            raw_values.extend(
                (value, fact.fact_id, fact.polarity)
                for value in fact.evidence_spans
            )

        output: list[_CanonicalValue] = []
        seen: set[str] = set()
        for raw, fact_id, polarity in raw_values:
            for value in self._extract_values(raw, value_type=value_type):
                grounded = self._restore_exact(source, value)
                key = normalize_text(grounded).casefold()
                if not grounded or key in seen:
                    continue
                output.append(
                    _CanonicalValue(
                        value=normalize_text(grounded),
                        evidence_span=grounded,
                        origin_fact_id=fact_id,
                        polarity=normalize_text(polarity).casefold() or "positive",
                    )
                )
                seen.add(key)
        return output

    def _extract_values(self, raw: str, *, value_type: str) -> list[str]:
        text = normalize_text(raw).strip(" \"'`.,;:")
        if not text:
            return []
        if value_type == "measurement":
            return [match.group(0) for match in self._MEASUREMENT_RE.finditer(text)]
        if value_type == "count":
            return [match.group(0) for match in self._NUMBER_RE.finditer(text)]
        if value_type == "boolean":
            return [match.group(0) for match in self._BOOLEAN_RE.finditer(text)]
        if value_type == "list":
            return [text] if len([part for part in re.split(r"[,;\n]", text) if part.strip()]) >= 2 else []
        return [text]

    @staticmethod
    def _restore_exact(source: str, value: str) -> str:
        if not source or not value:
            return ""
        match = re.search(re.escape(value), source, re.IGNORECASE)
        if match:
            return source[match.start() : match.end()]
        variants = {
            value.replace("m^3", "m3"),
            value.replace("m3", "m^3"),
            value.replace("m³", "m3"),
            value.replace("m3", "m³"),
        }
        for variant in variants:
            match = re.search(re.escape(variant), source, re.IGNORECASE)
            if match:
                return source[match.start() : match.end()]
        return ""


class DirectEvidencePromoter:
    """Promote a grounded answer value through fixed, non-weighted gates."""

    _COUNT_RE = re.compile(
        r"\b(?:how many|number of|count of|total number|highest number|lowest number|"
        r"fewest|most|least)\b",
        re.IGNORECASE,
    )
    _MEASUREMENT_RE = re.compile(
        r"\b(?:volume|distance|height|weight|duration|speed|area|capacity|m\^?3|m3|"
        r"kilometers?|miles?|meters?|metres?|kilograms?|hours?|minutes?|seconds?)\b",
        re.IGNORECASE,
    )
    _BOOLEAN_RE = re.compile(r"\b(?:yes\s*(?:or|/)\s*no|whether)\b", re.IGNORECASE)
    _LIST_RE = re.compile(r"\b(?:list|names of|titles of|all of the|which of)\b", re.IGNORECASE)
    _GLOBAL_RE = re.compile(
        r"\b(?:how many|number of|count|total|highest|lowest|maximum|minimum|most|"
        r"least|all|list|missing|difference)\b",
        re.IGNORECASE,
    )
    _EXPLICIT_AGGREGATE_RE = re.compile(
        r"\b(?:total(?:s|ed)?|altogether|in all|has|have|contains?|includes?|"
        r"maximum|minimum|highest|lowest|most|least|number of|count(?:s|ed)?)\b",
        re.IGNORECASE,
    )
    _NEGATIVE_REQUIREMENT_RE = re.compile(
        r"\b(?:does not|do not|did not|is not|was not|without|missing|absent|"
        r"not mention|not contain|lacks?)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        canonicalizer: AnswerValueCanonicalizer | None = None,
        answer_bound_validator: AnswerBoundFactValidator | None = None,
    ) -> None:
        self.canonicalizer = canonicalizer or AnswerValueCanonicalizer()
        self.answer_bound_validator = answer_bound_validator or AnswerBoundFactValidator()

    def promote(
        self,
        *,
        model_role: str,
        candidate_span: str,
        context: str,
        question: str,
        answer_requirement: str,
        answer_target: str,
        source_id: str,
        source_title: str,
        document_id: str,
        goal_id: str,
        semantic_facts: Iterable[EvidenceFact],
    ) -> DirectEvidencePromotionResult:
        result = DirectEvidencePromotionResult()
        facts = list(semantic_facts)
        requirement = normalize_text(answer_requirement) or normalize_text(question)
        value_type = self._value_type(requirement)
        candidates = self.canonicalizer.candidates(
            candidate_span=candidate_span,
            context=context,
            facts=facts,
            value_type=value_type,
        )
        if normalize_text(model_role).upper() != "ANSWER_SUPPORT":
            return self._reject_all(result, candidates, "role_authorization", "model_role_not_answer_support")
        if self._NEGATIVE_REQUIREMENT_RE.search(requirement) or (
            facts and all(fact.polarity == "negative" for fact in facts)
        ):
            return self._reject_all(
                result,
                candidates,
                "positive_polarity",
                "negative_value_uses_negative_fact_pipeline",
            )
        positive = [item for item in candidates if item.polarity != "negative"]
        for item in candidates:
            if item.polarity == "negative":
                result.diagnostics.append(self._rejection(item, "positive_polarity", "negative_value_uses_negative_fact_pipeline"))
        compatible: list[_CanonicalValue] = []
        for item in positive:
            ok, reason = self.answer_bound_validator.value_compatible(
                requirement=requirement,
                value=item.value,
            )
            if not ok:
                result.diagnostics.append(self._rejection(item, "answer_type", reason))
                continue
            compatible.append(item)
        if not compatible:
            return result

        span_key = normalize_text(candidate_span).strip(" \"'`.,;:").casefold()
        exact_span = [item for item in compatible if item.value.casefold() == span_key]
        if exact_span:
            compatible = exact_span
        distinct = {item.value.casefold() for item in compatible}
        if len(distinct) > 1:
            return self._reject_all(result, compatible, "conflict", "conflicting_answer_values")

        item = compatible[0]
        probe = EvidenceFact(
            fact_id=item.origin_fact_id,
            subject=normalize_text(answer_target) or requirement,
            relation=requirement,
            object=item.value,
            evidence_spans=[item.evidence_span],
            context=normalize_text(context),
            source_id=source_id,
            source_title=source_title,
            grounding_status="grounded",
            role="ANSWER_SUPPORT",
        )
        if not self.answer_bound_validator.target_bound(
            answer_target=answer_target,
            requirement=requirement,
            fact=probe,
        ):
            result.diagnostics.append(self._rejection(item, "context_binding", "answer_target_not_grounded"))
            return result

        scope_status = self._scope_status(requirement, context, facts)
        if not scope_status:
            result.diagnostics.append(self._rejection(item, "scope_completion", "aggregate_scope_not_complete"))
            return result

        grounded = GroundedAnswerValue(
            value=item.value,
            evidence_span=item.evidence_span,
            context=normalize_text(context),
            answer_requirement=requirement,
            answer_target=normalize_text(answer_target),
            source_id=normalize_text(source_id),
            source_title=normalize_text(source_title),
            document_id=normalize_text(document_id),
            goal_id=normalize_text(goal_id),
            origin_fact_id=item.origin_fact_id,
            value_type=value_type,
            promotion_reason="ordered_gates_passed",
            scope_status=scope_status,
        )
        fact = self._promoted_fact(grounded)
        result.promoted_values.append(grounded)
        result.promoted_facts.append(fact)
        result.diagnostics.append(
            PromotionDiagnostic(item.value, item.origin_fact_id, True, "", "promoted")
        )
        return result

    def _value_type(self, requirement: str) -> str:
        if self._MEASUREMENT_RE.search(requirement):
            return "measurement"
        if self._COUNT_RE.search(requirement):
            return "count"
        if self._BOOLEAN_RE.search(requirement):
            return "boolean"
        if self._LIST_RE.search(requirement):
            return "list"
        return "text"

    def _scope_status(
        self,
        requirement: str,
        context: str,
        facts: list[EvidenceFact],
    ) -> str:
        if not self._GLOBAL_RE.search(requirement):
            return "local_explicit"
        declared = {
            normalize_text(fact.qualifiers.get("scope_status", "")).casefold()
            for fact in facts
        }
        if declared & {"global_complete", "derived_complete"}:
            return next(iter(declared & {"global_complete", "derived_complete"}))
        if self._EXPLICIT_AGGREGATE_RE.search(context):
            return "local_explicit"
        return ""

    def _promoted_fact(self, value: GroundedAnswerValue) -> EvidenceFact:
        seed = "\x1f".join(
            [value.source_id, value.goal_id, value.answer_requirement, value.value]
        )
        fact_id = f"promotion-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"
        qualifiers = {
            "answer_binding": "direct",
            "binding_reason": "grounded_answer_value_promotion",
            "answer_requirement": value.answer_requirement,
            "answer_target": value.answer_target,
            "contract_method": "grounded_answer_value_promotion",
            "origin_fact_id": value.origin_fact_id,
            "scope_status": value.scope_status,
            "value_type": value.value_type,
        }
        return EvidenceFact(
            fact_id=fact_id,
            subject=value.answer_target or value.answer_requirement,
            relation=value.answer_requirement,
            object=value.value,
            qualifiers=qualifiers,
            polarity="positive",
            role="ANSWER_SUPPORT",
            goal_id=value.goal_id,
            evidence_spans=[value.evidence_span],
            evidence_refs=[
                FactEvidenceRef(
                    source_id=value.source_id,
                    unit_id=value.document_id or value.source_id,
                    document_id=value.document_id,
                    text=value.context,
                )
            ],
            context=value.context,
            source_id=value.source_id,
            source_type="web",
            source_title=value.source_title,
            grounding_status="grounded",
            extraction_method="grounded_answer_value_promotion",
            parent_fact_ids=[value.origin_fact_id] if value.origin_fact_id else [],
            derivation_type="answer_value_promotion",
        )

    @staticmethod
    def _rejection(item: _CanonicalValue, gate: str, reason: str) -> PromotionDiagnostic:
        return PromotionDiagnostic(item.value, item.origin_fact_id, False, gate, reason)

    def _reject_all(
        self,
        result: DirectEvidencePromotionResult,
        candidates: Iterable[_CanonicalValue],
        gate: str,
        reason: str,
    ) -> DirectEvidencePromotionResult:
        result.diagnostics.extend(self._rejection(item, gate, reason) for item in candidates)
        return result


__all__ = [
    "AnswerValueCanonicalizer",
    "DirectEvidencePromoter",
    "DirectEvidencePromotionResult",
    "GroundedAnswerValue",
    "PromotionDiagnostic",
]
