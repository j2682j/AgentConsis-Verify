from __future__ import annotations

from dataclasses import replace
import re

from utils.network_utils import normalize_text

from .models import EvidenceFact


class AnswerBoundFactValidator:
    """Bind grounded facts to the natural-language answer requirement."""

    _NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
    _YEAR_RE = re.compile(r"^(?:1[5-9]\d{2}|20\d{2}|21\d{2})$")
    _UNIT_RE = re.compile(
        r"\b(?:m\^?3|m3|km|mi|miles?|meters?|metres?|kg|g|lb|lbs|hours?|hrs?|"
        r"minutes?|mins?|seconds?|secs?|cm|mm|ft|feet|inches?|%|percent|mph|"
        r"km/h|m/s|sqm|square|cubic)\b",
        flags=re.IGNORECASE,
    )
    _COUNT_RE = re.compile(
        r"\b(?:how many|number of|count of|total number|highest number|lowest number|"
        r"fewest|most|least)\b",
        flags=re.IGNORECASE,
    )
    _MEASUREMENT_RE = re.compile(
        r"\b(?:volume|distance|height|weight|duration|speed|area|capacity|m\^?3|m3|"
        r"kilometers?|miles?|meters?|metres?|kilograms?|hours?|minutes?|seconds?)\b",
        flags=re.IGNORECASE,
    )
    _BOOLEAN_DIRECTIVE_RE = re.compile(
        r"\b(?:yes\s*(?:or|/)\s*no|whether)\b",
        flags=re.IGNORECASE,
    )
    _BOOLEAN_QUESTION_RE = re.compile(
        r"(?:^|[.!?]\s+)"
        r"(?:can|could|do|does|did|is|are|was|were|has|have|had|"
        r"will|would|should)\b[^?]*\?",
        flags=re.IGNORECASE,
    )
    _LIST_RE = re.compile(
        r"\b(?:list|names of|titles of|all of the|which of the following)\b",
        flags=re.IGNORECASE,
    )
    _TEXT_RE = re.compile(
        r"\b(?:title|called|named|exactly|setting|scene heading|phrase|wording|"
        r"location|place)\b",
        flags=re.IGNORECASE,
    )
    _DURATION_RE = re.compile(
        r"^\d+(?::\d+){1,2}$|^\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?)$",
        flags=re.IGNORECASE,
    )
    _TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
    _STOPWORDS = {
        "about",
        "after",
        "answer",
        "before",
        "between",
        "from",
        "give",
        "highest",
        "how",
        "included",
        "latest",
        "many",
        "number",
        "question",
        "return",
        "should",
        "that",
        "the",
        "this",
        "use",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
    }

    def bind(
        self,
        fact: EvidenceFact,
        *,
        question: str,
        answer_requirement: str = "",
        answer_target: str = "",
    ) -> EvidenceFact:
        requirement = normalize_text(answer_requirement) or normalize_text(question)
        qualifiers = dict(fact.qualifiers)
        qualifiers["answer_requirement"] = requirement

        if fact.role == "CONTEXT":
            qualifiers.update(
                {"answer_binding": "context", "binding_reason": "context_fact"}
            )
            return replace(fact, qualifiers=qualifiers)
        if fact.role == "BRIDGE":
            qualifiers.update(
                {"answer_binding": "bridge", "binding_reason": "model_bridge_fact"}
            )
            return replace(fact, qualifiers=qualifiers)
        if fact.role != "ANSWER_SUPPORT" or fact.grounding_status != "grounded":
            qualifiers.update(
                {"answer_binding": "unbound", "binding_reason": "fact_not_grounded"}
            )
            return replace(fact, qualifiers=qualifiers)

        compatible, reason = self._value_compatible(
            requirement=requirement,
            value=fact.object,
        )
        target_bound = self._target_bound(
            answer_target=answer_target,
            requirement=requirement,
            fact=fact,
        )
        if compatible and target_bound:
            qualifiers.update(
                {"answer_binding": "direct", "binding_reason": reason}
            )
            return replace(fact, qualifiers=qualifiers)

        demoted_role = "BRIDGE" if target_bound else "CONTEXT"
        qualifiers.update(
            {
                "answer_binding": demoted_role.casefold(),
                "binding_reason": (
                    reason if not compatible else "answer_target_not_grounded"
                ),
                "original_role": "ANSWER_SUPPORT",
            }
        )
        return replace(fact, role=demoted_role, qualifiers=qualifiers)

    def is_direct(self, fact: EvidenceFact) -> bool:
        return (
            fact.role == "ANSWER_SUPPORT"
            and fact.grounding_status == "grounded"
            and fact.qualifiers.get("answer_binding") == "direct"
        )

    def value_compatible(self, *, requirement: str, value: str) -> tuple[bool, str]:
        """Expose answer-value compatibility without changing fact roles."""

        return self._value_compatible(requirement=requirement, value=value)

    def target_bound(
        self,
        *,
        answer_target: str,
        requirement: str,
        fact: EvidenceFact,
    ) -> bool:
        """Expose target binding for value-centric evidence promotion."""

        return self._target_bound(
            answer_target=answer_target,
            requirement=requirement,
            fact=fact,
        )

    def _value_compatible(self, *, requirement: str, value: str) -> tuple[bool, str]:
        requirement_text = normalize_text(requirement)
        value_text = normalize_text(value).strip(" .,:;!?()[]{}'\"")
        if not value_text:
            return False, "empty_answer_value"

        if self._COUNT_RE.search(requirement_text):
            if not self._NUMBER_RE.fullmatch(value_text):
                return False, "count_requires_numeric_answer_value"
            compact = value_text.replace(",", "")
            if self._YEAR_RE.fullmatch(compact):
                return False, "count_rejects_standalone_year"
            return True, "count_value_matches_requirement"

        if self._MEASUREMENT_RE.search(requirement_text):
            if not self._NUMBER_RE.search(value_text) or not self._UNIT_RE.search(value_text):
                return False, "measurement_requires_number_and_unit"
            return True, "measurement_value_matches_requirement"

        if self._is_boolean_requirement(requirement_text):
            if value_text.casefold() not in {"yes", "no", "true", "false"}:
                return False, "boolean_requires_yes_or_no_value"
            return True, "boolean_value_matches_requirement"

        if self._LIST_RE.search(requirement_text):
            items = [item.strip() for item in re.split(r"[,;\n]", value_text) if item.strip()]
            if len(items) < 2:
                return False, "list_requires_multiple_answer_values"
            return True, "list_value_matches_requirement"

        if self._TEXT_RE.search(requirement_text):
            if self._NUMBER_RE.fullmatch(value_text) or self._DURATION_RE.fullmatch(value_text):
                return False, "text_answer_rejects_numeric_or_duration_value"
            return True, "text_value_matches_requirement"

        return True, "model_direct_role_with_grounded_value"

    @classmethod
    def _is_boolean_requirement(cls, requirement: str) -> bool:
        """Recognize yes/no questions without treating ``What is ...?`` as boolean."""

        text = normalize_text(requirement)
        return bool(
            cls._BOOLEAN_DIRECTIVE_RE.search(text)
            or cls._BOOLEAN_QUESTION_RE.search(text)
        )

    def _target_bound(
        self,
        *,
        answer_target: str,
        requirement: str,
        fact: EvidenceFact,
    ) -> bool:
        target = normalize_text(answer_target) or normalize_text(requirement)
        terms = {
            match.group(0).casefold()
            for match in self._TERM_RE.finditer(target)
            if len(match.group(0)) >= 4
            and match.group(0).casefold() not in self._STOPWORDS
            and not match.group(0).isdigit()
        }
        if not terms:
            return True
        support = normalize_text(
            " ".join(
                [
                    fact.subject,
                    fact.relation,
                    fact.context,
                    " ".join(fact.evidence_spans),
                ]
            )
        ).casefold()
        return any(term in support for term in terms)


__all__ = ["AnswerBoundFactValidator"]
