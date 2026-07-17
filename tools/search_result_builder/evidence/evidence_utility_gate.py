from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text

from ..next_hop_query.evidence_sufficiency_gate import EvidenceSufficiencyGate
from .span_builder import EvidenceSpan, SpanBuilder
from .span_recovery import RecoveredSpans, SpanRecovery


DIRECT = "direct"
DIRECT_STRONG = "direct_strong"
DIRECT_WEAK = "direct_weak"
BRIDGE = "bridge"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvidenceUtilityResult:
    """
    Classify whether one retrieved chunk can support an answer.

    Args:
        - support_level: direct_strong / direct_weak / bridge / unsupported.
        - answer_spans: Spans that can directly support an answer.
        - bridge_spans: Spans useful for later retrieval or reasoning.
        - supporting_context: Context selected by SpanBuilder or sufficiency gate.
        - reasons: Human-readable diagnostic reasons.
        - can_support_sufficient: Whether this chunk may satisfy sufficiency.
        - valid_for_evidence: Whether this chunk may be converted to Stage1 evidence.
        - valid_for_next_hop: Whether this chunk may drive a next-hop query.

    Returns:
        - EvidenceUtilityResult: Utility contract for one retrieval document.
    """

    support_level: str
    answer_spans: list[str] = field(default_factory=list)
    bridge_spans: list[str] = field(default_factory=list)
    matched_spans: list[dict[str, Any]] = field(default_factory=list)
    supporting_context: str = ""
    reasons: list[str] = field(default_factory=list)
    span_recovery_used: bool = False
    can_support_sufficient: bool = False
    valid_for_evidence: bool = False
    valid_for_next_hop: bool = False
    support_strength: str = ""
    normalized_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceUtilityGate:
    """
    Convert labeler spans into direct / bridge / unsupported evidence contracts.

    Args:
        - span_builder: Span restoration helper used to locate useful spans.
        - sufficiency_gate: Single-document answer binding gate for direct support.

    Returns:
        - EvidenceUtilityGate: Stateless evidence utility classifier.
    """

    _TERMINAL_TAGS = {"<FINISH>", "<TERMINATE>"}

    def __init__(
        self,
        *,
        span_builder: SpanBuilder | None = None,
        sufficiency_gate: EvidenceSufficiencyGate | None = None,
        span_recovery: SpanRecovery | None = None,
    ) -> None:
        self.span_builder = span_builder or SpanBuilder()
        self.sufficiency_gate = sufficiency_gate or EvidenceSufficiencyGate()
        self.span_recovery = span_recovery or SpanRecovery()

    def assess(
        self,
        *,
        question: str,
        document: Any,
        intent_plan: Any | None = None,
    ) -> EvidenceUtilityResult:
        """
        Classify one retrieved document.

        Args:
            - question: Original task question.
            - document: RetrievedDocumentTrace-like object or dictionary.
            - intent_plan: Optional search intent state.

        Returns:
            - EvidenceUtilityResult: Support level and downstream permissions.
        """
        text = self._field(document, "text")
        title = self._field(document, "title")
        if self._bool_field(document, "duplicate"):
            return self._unsupported("duplicate_document")
        if not normalize_text(text):
            return self._unsupported("empty_document_text")

        label_contract_valid = self._field(document, "label_status") in {
            "valid_continue",
            "valid_terminate",
        }
        useful_spans = self._clean_items(
            self._list_field(document, "useful_spans")
            or self._list_field(document, "useful_tokens")
        )
        if useful_spans:
            context, matched_spans = self.span_builder.build_context(text, useful_spans)
            context = normalize_text(context or text)
            matched_span_dicts = [asdict(span) for span in matched_spans]
            direct = self._direct_support(
                question=question,
                document=document,
                context=context,
                useful_spans=useful_spans,
                intent_plan=intent_plan,
            )
            if direct is not None:
                return EvidenceUtilityResult(
                    support_level=DIRECT_STRONG,
                    answer_spans=direct["answer_spans"],
                    bridge_spans=[],
                    matched_spans=matched_span_dicts,
                    supporting_context=direct["supporting_context"] or context,
                    reasons=direct["reasons"],
                    can_support_sufficient=True,
                    valid_for_evidence=True,
                    valid_for_next_hop=False,
                    support_strength="strong",
                )
            weak_direct = self._weak_direct_support(
                question=question,
                context=context,
                useful_spans=useful_spans,
                intent_plan=intent_plan,
            )
            if weak_direct is not None:
                return EvidenceUtilityResult(
                    support_level=DIRECT_WEAK,
                    answer_spans=weak_direct["answer_spans"],
                    bridge_spans=[],
                    matched_spans=matched_span_dicts,
                    supporting_context=context,
                    reasons=weak_direct["reasons"],
                    can_support_sufficient=True,
                    valid_for_evidence=True,
                    valid_for_next_hop=False,
                    support_strength="weak",
                    normalized_constraints=weak_direct["normalized_constraints"],
                )

            sequence_tag = self._field(document, "sequence_tag")
            reasons = ["useful_span_without_direct_answer_binding"]
            if title:
                reasons.append("source_title_available")
            if sequence_tag in self._TERMINAL_TAGS:
                reasons.append("terminal_label_demoted_to_bridge")
            else:
                reasons.append("continue_label_bridge")
            if not label_contract_valid:
                reasons.extend(self._list_field(document, "invalid_reasons"))
            bridge_spans = self._quality_bridge_spans(
                useful_spans,
                question=question,
                document=document,
                intent_plan=intent_plan,
                reasons=reasons,
            )
            if not bridge_spans:
                return self._unsupported(
                    "bridge_span_quality_gate_failed",
                    extra_reasons=reasons,
                )
            bridge_context, bridge_matches = self.span_builder.build_context(text, bridge_spans)
            return EvidenceUtilityResult(
                support_level=BRIDGE,
                answer_spans=[],
                bridge_spans=bridge_spans,
                matched_spans=[asdict(span) for span in bridge_matches] or matched_span_dicts,
                supporting_context=bridge_context or context,
                reasons=self._clean_items(reasons),
                can_support_sufficient=False,
                valid_for_evidence=False,
                valid_for_next_hop=True,
            )

        recovery = self.span_recovery.recover(
            question=question,
            title=title,
            text=text,
            intent_plan=intent_plan,
            answer_role=self._field(intent_plan, "answer_role") if intent_plan else "",
        )
        return self._from_recovery(
            question=question,
            document=document,
            recovery=recovery,
            label_contract_valid=label_contract_valid,
            intent_plan=intent_plan,
        )

    def _direct_support(
        self,
        *,
        question: str,
        document: Any,
        context: str,
        useful_spans: list[str],
        intent_plan: Any | None,
    ) -> dict[str, Any] | None:
        gate_result = self.sufficiency_gate.assess(
            question=question,
            documents=[
                {
                    "title": self._field(document, "title"),
                    "text": self._field(document, "text"),
                    "url": self._field(document, "url"),
                    "retrieval_score": self._float_field(document, "retrieval_score"),
                    "label": self._field(document, "label"),
                    "sequence_tag": self._field(document, "sequence_tag"),
                    "useful_spans": useful_spans,
                    "valid_for_evidence": True,
                }
            ],
            intent_plan=intent_plan,
        )
        if gate_result.sufficient and gate_result.mode == "typed_answer":
            return {
                "answer_spans": self._clean_items([gate_result.matched_span]),
                "supporting_context": gate_result.supporting_context,
                "reasons": [f"typed_answer:{gate_result.reason}"],
            }
        return None

    def _from_recovery(
        self,
        *,
        question: str,
        document: Any,
        recovery: RecoveredSpans,
        label_contract_valid: bool,
        intent_plan: Any | None,
    ) -> EvidenceUtilityResult:
        reasons = list(recovery.reasons)
        if not label_contract_valid:
            reasons.append("invalid_label_contract")
            reasons.extend(self._list_field(document, "invalid_reasons"))

        if recovery.answer_spans:
            context, matched_spans = self.span_builder.build_context(
                self._field(document, "text"),
                recovery.answer_spans,
            )
            direct = self._direct_support(
                question=question,
                document=document,
                context=context,
                useful_spans=recovery.answer_spans,
                intent_plan=intent_plan,
            )
            if direct is not None:
                return EvidenceUtilityResult(
                    support_level=DIRECT_STRONG,
                    answer_spans=direct["answer_spans"],
                    bridge_spans=[],
                    matched_spans=[asdict(span) for span in matched_spans],
                    supporting_context=direct["supporting_context"] or context,
                    reasons=self._clean_items(reasons + direct["reasons"]),
                    span_recovery_used=True,
                    can_support_sufficient=True,
                    valid_for_evidence=True,
                    valid_for_next_hop=False,
                    support_strength="strong",
                )
            weak_direct = self._weak_direct_support(
                question=question,
                context=context,
                useful_spans=recovery.answer_spans,
                intent_plan=intent_plan,
            )
            if weak_direct is not None:
                return EvidenceUtilityResult(
                    support_level=DIRECT_WEAK,
                    answer_spans=weak_direct["answer_spans"],
                    bridge_spans=[],
                    matched_spans=[asdict(span) for span in matched_spans],
                    supporting_context=context,
                    reasons=self._clean_items(reasons + weak_direct["reasons"]),
                    span_recovery_used=True,
                    can_support_sufficient=True,
                    valid_for_evidence=True,
                    valid_for_next_hop=False,
                    support_strength="weak",
                    normalized_constraints=weak_direct["normalized_constraints"],
                )

        bridge_spans = self._clean_items(
            list(recovery.answer_spans) + list(recovery.bridge_spans)
        )
        bridge_spans = self._quality_bridge_spans(
            bridge_spans,
            question=question,
            document=document,
            intent_plan=intent_plan,
            reasons=reasons,
            span_sources=recovery.span_sources,
        )
        if bridge_spans:
            context, matched_spans = self.span_builder.build_context(
                self._field(document, "text"),
                bridge_spans,
            )
            return EvidenceUtilityResult(
                support_level=BRIDGE,
                answer_spans=[],
                bridge_spans=bridge_spans,
                matched_spans=[asdict(span) for span in matched_spans],
                supporting_context=context or self._field(document, "text")[:500],
                reasons=self._clean_items(reasons + ["fallback_recovery_bridge"]),
                span_recovery_used=True,
                can_support_sufficient=False,
                valid_for_evidence=False,
                valid_for_next_hop=True,
            )

        return self._unsupported(
            "no_useful_or_recovered_span",
            extra_reasons=reasons,
            span_recovery_used=True,
        )

    def _unsupported(
        self,
        reason: str,
        *,
        extra_reasons: list[str] | None = None,
        span_recovery_used: bool = False,
    ) -> EvidenceUtilityResult:
        reasons = [reason]
        reasons.extend(extra_reasons or [])
        return EvidenceUtilityResult(
            support_level=UNSUPPORTED,
            reasons=self._clean_items(reasons),
            span_recovery_used=span_recovery_used,
        )

    def _weak_direct_support(
        self,
        *,
        question: str,
        context: str,
        useful_spans: list[str],
        intent_plan: Any | None,
    ) -> dict[str, Any] | None:
        role = self._answer_role(question, intent_plan)
        if role not in {"number", "volume", "duration", "distance", "date", "zip_code"}:
            return None
        constraints = self._normalized_constraints(question=question, intent_plan=intent_plan, role=role)
        missing = [
            constraint
            for constraint in constraints
            if not self._constraint_covered(context, constraint)
        ]
        if missing:
            return None

        spans = self._clean_items(useful_spans)
        answer_spans = [
            span
            for span in spans
            if self._contains_span(context, span)
            and (
                self._span_matches_role(span, role)
                or (
                    constraints
                    and role in {"volume", "duration", "distance"}
                    and self._span_matches_role(span, "number")
                )
            )
        ]
        if not answer_spans:
            return None

        reasons = [
            f"weak_direct:{role}",
            "answer_span_present",
            "normalized_constraint_binding",
        ]
        if not constraints:
            reasons.append("no_explicit_normalized_constraint")
        return {
            "answer_spans": answer_spans[:3],
            "reasons": self._clean_items(reasons),
            "normalized_constraints": constraints,
        }

    def _quality_bridge_spans(
        self,
        spans: list[str],
        *,
        question: str,
        document: Any,
        intent_plan: Any | None,
        reasons: list[str],
        span_sources: dict[str, str] | None = None,
    ) -> list[str]:
        role = self._answer_role(question, intent_plan)
        text = self._field(document, "text")
        title = self._field(document, "title")
        effective: list[str] = []
        for span in self._clean_items(spans):
            ok, reason = self._bridge_span_is_effective(
                span,
                question=question,
                text=text,
                title=title,
                role=role,
                intent_plan=intent_plan,
                span_source=(span_sources or {}).get(span, ""),
            )
            if ok:
                effective.append(span)
            elif reason:
                reasons.append(f"bridge_span_rejected:{reason}:{span}")
        return self._clean_items(effective)

    def _bridge_span_is_effective(
        self,
        span: str,
        *,
        question: str,
        text: str,
        title: str,
        role: str,
        intent_plan: Any | None,
        span_source: str = "",
    ) -> tuple[bool, str]:
        cleaned = normalize_text(span)
        if not cleaned:
            return False, "empty"
        if self._is_page_structure_span(cleaned):
            return False, "page_structure"
        if self._is_source_name_span(cleaned):
            return False, "source_name"
        if self._crosses_sentence_boundary(cleaned):
            return False, "sentence_boundary"
        if self._has_bad_phrase_boundary(cleaned):
            return False, "bad_phrase_boundary"
        if role != "date" and self._is_date_only_span(cleaned):
            return False, "date_only_non_date_task"
        if self._contains_span(question, cleaned) and not self._has_value_signal(cleaned):
            return False, "question_echo"
        if span_source == "title" and not self._contains_span(text, cleaned):
            return False, "title_only"

        if self._span_matches_role(cleaned, role) or self._has_value_signal(cleaned):
            return True, ""
        for term in self._intent_terms(intent_plan):
            if self._contains_span(cleaned, term) or self._contains_span(term, cleaned):
                return True, ""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}", cleaned)
        if len(words) >= 2 and self._contains_span(text, cleaned):
            return True, ""
        return False, "weak_or_contextless"

    def _answer_role(self, question: str, intent_plan: Any | None) -> str:
        planned = self._field(intent_plan, "answer_role") if intent_plan else ""
        if planned and planned != "unknown":
            return self.sufficiency_gate._normalize_answer_role(planned).lower()
        return self.sufficiency_gate._answer_role(
            normalize_text(question),
            intent_plan=intent_plan,
        ).lower()

    def _span_matches_role(self, span: str, role: str) -> bool:
        text = normalize_text(span)
        if not text:
            return False
        role_patterns = {
            "number": r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])",
            "volume": r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:m\^?3|m3|m³|cubic\s+met(?:er|re)s?|lit(?:er|re)s?|l)\b",
            "duration": r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:thousand\s+)?(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
            "distance": r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|miles?|meters?|metres?)\b",
            "date": r"\b(?:18|19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b",
            "zip_code": r"\b\d{5}(?:-\d{4})?\b",
        }
        return bool(re.search(role_patterns.get(role, ""), text, flags=re.IGNORECASE))

    def _has_value_signal(self, span: str) -> bool:
        text = normalize_text(span)
        value_patterns = [
            r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|miles?|meters?|metres?)\b",
            r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
            r"(?<![A-Za-z0-9])\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:m\^?3|m3|m糧|cubic\s+met(?:er|re)s?|lit(?:er|re)s?|l)\b",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in value_patterns)

    def _is_date_only_span(self, span: str) -> bool:
        text = normalize_text(span)
        return bool(
            re.fullmatch(
                r"(?:18|19|20)\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}|\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _crosses_sentence_boundary(self, span: str) -> bool:
        return bool(re.search(r"[.!?]\s+[A-Z0-9]", normalize_text(span)))

    def _has_bad_phrase_boundary(self, span: str) -> bool:
        tokens = self._canonical_key(span).split()
        return bool(tokens and tokens[-1] in {"and", "for", "in", "of", "on", "the", "to"})

    def _is_page_structure_span(self, span: str) -> bool:
        key = self._canonical_key(span)
        terms = {
            "advertisement",
            "caption",
            "category",
            "comments",
            "content",
            "copyright",
            "current community",
            "external links",
            "headings",
            "image alt",
            "introduction",
            "lists",
            "login",
            "metadata",
            "navigation",
            "privacy",
            "references",
            "related articles",
            "search",
            "source",
            "structured data",
            "suggested searches",
            "table",
            "terms",
            "title",
            "user name",
        }
        return any(term in key for term in terms)

    def _is_source_name_span(self, span: str) -> bool:
        key = self._canonical_key(span)
        terms = {
            "britannica",
            "facebook",
            "fandom",
            "github",
            "google",
            "instagram",
            "linkedin",
            "nasa science",
            "researchgate",
            "stack exchange",
            "twitter",
            "wikipedia",
            "wikimedia commons",
            "youtube",
        }
        return any(term in key for term in terms)

    def _intent_terms(self, intent_plan: Any | None) -> list[str]:
        if intent_plan is None:
            return []
        terms: list[str] = []
        target = self._field(intent_plan, "target")
        if target:
            terms.append(target)
        for name in ("must_include", "missing_terms", "completed_terms"):
            for value in self._list_field(intent_plan, name):
                text = normalize_text(str(value or ""))
                if text and not text.startswith("answer_candidate:"):
                    terms.append(text)
        return self._clean_items(terms)

    def _contains_span(self, context: str, span: str) -> bool:
        return self._canonical_text(span) in self._canonical_text(context)

    def _normalized_constraints(
        self,
        *,
        question: str,
        intent_plan: Any | None,
        role: str,
    ) -> list[str]:
        source = " ".join(
            part
            for part in [
                question,
                self._field(intent_plan, "target") if intent_plan else "",
                " ".join(self._list_field(intent_plan, "must_include")) if intent_plan else "",
                " ".join(self._list_field(intent_plan, "missing_terms")) if intent_plan else "",
            ]
            if part
        )
        text = self._canonical_text(source)
        constraints: list[str] = []
        if role == "volume" or any(term in text for term in ["m3", "cubic meter", "cubic metre"]):
            constraints.append("unit:volume_m3")
        if role == "distance":
            if any(term in text for term in [" km ", "kilometer", "kilometre"]):
                constraints.append("unit:distance_km")
            if "mile" in text:
                constraints.append("unit:distance_mile")
        if role == "duration":
            if "hour" in text or "hr" in text:
                constraints.append("unit:duration_hour")
            if "minute" in text or " min " in text:
                constraints.append("unit:duration_minute")
            if "second" in text or " sec " in text:
                constraints.append("unit:duration_second")
        if role == "zip_code":
            constraints.append("answer_role:zip_code")
        return self._clean_items(constraints)

    def _constraint_covered(self, context: str, constraint: str) -> bool:
        text = self._canonical_text(context)
        if constraint == "unit:volume_m3":
            return any(term in text for term in ["m3", "m^3", "m 3", "cubic meter", "cubic metre"])
        if constraint == "unit:distance_km":
            return any(term in text for term in [" km ", "kilometer", "kilometre"])
        if constraint == "unit:distance_mile":
            return "mile" in text
        if constraint == "unit:duration_hour":
            return "hour" in text or " hr" in text
        if constraint == "unit:duration_minute":
            return "minute" in text or " min " in text
        if constraint == "unit:duration_second":
            return "second" in text or " sec " in text
        if constraint == "answer_role:zip_code":
            return bool(re.search(r"\b\d{5}(?:-\d{4})?\b", text))
        return constraint in text

    def _canonical_text(self, value: Any) -> str:
        text = normalize_text(value).casefold()
        text = text.replace("m³", "m3")
        text = re.sub(r"\bm\s*\^\s*3\b", "m3", text)
        text = re.sub(r"\bm\s+3\b", "m3", text)
        text = re.sub(r"\s+", " ", text)
        return f" {text.strip()} "

    def _canonical_key(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).casefold()).strip()

    def _field(self, document: Any, name: str) -> str:
        if isinstance(document, dict):
            value = document.get(name, "")
        else:
            value = getattr(document, name, "")
        return normalize_text(value)

    def _bool_field(self, document: Any, name: str) -> bool:
        if isinstance(document, dict):
            return bool(document.get(name, False))
        return bool(getattr(document, name, False))

    def _float_field(self, document: Any, name: str) -> float:
        if isinstance(document, dict):
            value = document.get(name, 0.0)
        else:
            value = getattr(document, name, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _list_field(self, document: Any, name: str) -> list[Any]:
        if isinstance(document, dict):
            value = document.get(name, [])
        else:
            value = getattr(document, name, [])
        return value if isinstance(value, list) else []

    def _clean_items(self, items: list[Any]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = normalize_text(str(item or ""))
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result


__all__ = [
    "BRIDGE",
    "DIRECT",
    "DIRECT_STRONG",
    "DIRECT_WEAK",
    "UNSUPPORTED",
    "EvidenceUtilityGate",
    "EvidenceUtilityResult",
]
