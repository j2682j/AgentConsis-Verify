from __future__ import annotations

from dataclasses import dataclass
import re

from utils.network_utils import normalize_text


@dataclass
class AnswerRequirementResult:
    """Describe whether a candidate answer satisfies the answer contract."""

    outcome: str
    reason: str
    requirement: str = ""
    expected_type: str = "unknown"
    declared_type: str = "unknown"
    observed_type: str = "unknown"

    def to_dict(self) -> dict[str, str]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "requirement": self.requirement,
            "expected_type": self.expected_type,
            "declared_type": self.declared_type,
            "observed_type": self.observed_type,
        }


class AnswerRequirementGate:
    """
    Validate only clear answer-contract mismatches without calling a model.

    Ambiguous semantic roles remain ``unknown`` so this gate does not reject a
    valid answer merely because a lightweight classifier cannot identify it.
    """

    _TYPE_ALIASES = {
        "bool": "boolean",
        "boolean": "boolean",
        "yes_no": "boolean",
        "yes/no": "boolean",
        "count": "number",
        "quantity": "number",
        "amount": "number",
        "integer": "number",
        "float": "number",
        "percentage": "number",
        "ratio": "number",
        "duration": "number",
        "distance": "number",
        "volume": "number",
        "person": "person",
        "human": "person",
        "name": "person",
        "location": "place",
        "city": "place",
        "country": "place",
        "organization": "organization",
        "organisation": "organization",
        "title": "text",
        "work": "text",
        "date": "date",
        "year": "date",
        "list": "list",
        "array": "list",
        "text": "text",
        "string": "text",
    }
    _NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    }

    def evaluate(
        self,
        *,
        answer: str,
        answer_type: str = "",
        answer_requirement: str = "",
        answer_role: str = "",
    ) -> AnswerRequirementResult:
        requirement = normalize_text(answer_requirement)
        expected = self._expected_type(answer_role, requirement)
        declared = self._normalize_type(answer_type)
        observed = self._observed_type(answer)

        if expected == "unknown":
            return AnswerRequirementResult(
                outcome="unknown",
                reason="answer_requirement_type_unresolved",
                requirement=requirement,
                expected_type=expected,
                declared_type=declared,
                observed_type=observed,
            )

        if self._clearly_incompatible(expected, declared, observed):
            return AnswerRequirementResult(
                outcome="incompatible",
                reason="candidate_shape_conflicts_with_answer_requirement",
                requirement=requirement,
                expected_type=expected,
                declared_type=declared,
                observed_type=observed,
            )

        if declared == expected or observed == expected:
            return AnswerRequirementResult(
                outcome="compatible",
                reason="candidate_matches_answer_requirement",
                requirement=requirement,
                expected_type=expected,
                declared_type=declared,
                observed_type=observed,
            )

        return AnswerRequirementResult(
            outcome="unknown",
            reason="candidate_requirement_compatibility_uncertain",
            requirement=requirement,
            expected_type=expected,
            declared_type=declared,
            observed_type=observed,
        )

    def _expected_type(self, answer_role: str, requirement: str) -> str:
        role = self._normalize_type(answer_role)
        if role != "unknown":
            return role

        text = normalize_text(requirement).casefold()
        if not text:
            return "unknown"
        if re.search(r"\b(?:yes\s*(?:or|/)\s*no|whether)\b", text):
            return "boolean"
        if re.search(r"\b(?:how many|number of|count of|quantity of|total number)\b", text):
            return "number"
        if re.search(r"\b(?:percentage|percent|ratio|volume|distance|duration)\b", text):
            return "number"
        if re.search(r"\b(?:list|names of|titles of|all of the)\b", text):
            return "list"
        if re.search(r"\b(?:which|what) (?:person|author|director|founder|scientist|winner|name)\b", text):
            return "person"
        if re.search(r"\b(?:which|what) (?:city|country|place|location)\b", text):
            return "place"
        if re.search(r"\b(?:what|which) (?:date|year|month|day)\b", text):
            return "date"
        return "unknown"

    def _normalize_type(self, value: str) -> str:
        normalized = normalize_text(value).casefold().replace("-", "_").replace(" ", "_")
        if not normalized or normalized == "unknown":
            return "unknown"
        return self._TYPE_ALIASES.get(normalized, normalized if normalized in set(self._TYPE_ALIASES.values()) else "unknown")

    def _observed_type(self, answer: str) -> str:
        text = normalize_text(answer)
        lowered = text.casefold().rstrip(".")
        if lowered in {"yes", "no"}:
            return "boolean"
        if re.fullmatch(
            r"(?:\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            text,
        ):
            return "date"
        if self._looks_numeric(text):
            return "number"
        if len([part for part in re.split(r"[,;]", text) if part.strip()]) >= 2:
            return "list"
        return "text" if text else "unknown"

    def _looks_numeric(self, answer: str) -> bool:
        text = normalize_text(answer).casefold()
        words = text.split()
        if text in self._NUMBER_WORDS or (
            1 <= len(words) <= 3 and words[0] in self._NUMBER_WORDS
        ):
            return True
        return bool(
            re.fullmatch(
                r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|[A-Za-z]+|m\^?3|km|mi))?",
                text,
            )
            or re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text)
        )

    def _clearly_incompatible(
        self,
        expected: str,
        declared: str,
        observed: str,
    ) -> bool:
        if expected == "boolean":
            return observed not in {"boolean", "unknown"}
        if expected == "number":
            return observed not in {"number", "date", "unknown"} and declared != "number"
        if expected == "list":
            return declared not in {"list", "unknown"} and observed not in {"list", "unknown"}
        if expected in {"person", "place", "organization", "text", "date"}:
            if expected == "date":
                return observed == "boolean"
            return observed in {"number", "boolean"} and declared != expected
        return False


__all__ = ["AnswerRequirementGate", "AnswerRequirementResult"]
