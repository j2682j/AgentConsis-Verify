from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable, Mapping

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class FactGoalBindingResult:
    """描述一筆 grounded fact 是否真正完成指定 relation goal。"""

    status: str
    goal_id: str = ""
    fact_id: str = ""
    effective_subject: str = ""
    reason: str = ""

    @property
    def bound(self) -> bool:
        return self.status == "bound"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class FactGoalBindingValidator:
    """以 subject-relation-object contract 驗證 evidence fact。"""

    _TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
    _MIN_IDENTIFYING_TOKEN = 3
    _PERSON_TARGETS = {"person", "human", "author", "writer", "nominator", "name"}
    _NUMBER_TARGETS = {"number", "count", "quantity", "measurement", "amount"}
    _BOOLEAN_TARGETS = {"boolean", "yes no", "yes/no", "true false"}

    def validate(
        self,
        *,
        fact: Any,
        goal: Any,
        effective_subjects: Iterable[str] = (),
        answer_role: str = "",
    ) -> FactGoalBindingResult:
        fact_id = self._field(fact, "fact_id")
        goal_id = self._field(goal, "goal_id")
        fact_goal_id = self._field(fact, "goal_id")
        if goal_id and fact_goal_id != goal_id:
            return self._result(
                "goal_id_mismatch",
                goal_id,
                fact_id,
                reason="fact_goal_id_does_not_match_active_goal",
            )
        if self._field(fact, "grounding_status").casefold() != "grounded":
            return self._result(
                "ungrounded",
                goal_id,
                fact_id,
                reason="fact_is_not_grounded",
            )

        fact_subject = self._field(fact, "subject")
        subjects = self._dedupe(
            [*effective_subjects, self._field(goal, "subject")]
        )
        subjects = [item for item in subjects if item]
        if subjects and not any(
            self._entity_equivalent(fact_subject, subject)
            for subject in subjects
        ):
            return self._result(
                "subject_mismatch",
                goal_id,
                fact_id,
                effective_subject=subjects[0],
                reason="fact_subject_does_not_bind_goal_subject",
            )

        fact_relation = self.normalize_relation(self._field(fact, "relation"))
        goal_relation = self.normalize_relation(self._field(goal, "relation"))
        if not fact_relation:
            return self._result(
                "missing_fact_relation",
                goal_id,
                fact_id,
                effective_subject=subjects[0] if subjects else "",
                reason="fact_relation_is_empty",
            )
        if goal_relation and fact_relation != goal_relation:
            return self._result(
                "relation_mismatch",
                goal_id,
                fact_id,
                effective_subject=subjects[0] if subjects else "",
                reason="fact_relation_does_not_match_goal_relation",
            )

        target = normalize_text(answer_role) or self._field(goal, "target")
        if not self._object_role_compatible(self._field(fact, "object"), target):
            return self._result(
                "object_role_mismatch",
                goal_id,
                fact_id,
                effective_subject=subjects[0] if subjects else "",
                reason="fact_object_does_not_match_goal_target_role",
            )
        return self._result(
            "bound",
            goal_id,
            fact_id,
            effective_subject=subjects[0] if subjects else fact_subject,
            reason="fact_satisfies_subject_relation_object_contract",
        )

    @classmethod
    def normalize_relation(cls, value: str) -> str:
        text = normalize_text(value).casefold().replace("_", " ").replace("-", " ")
        tokens = cls._TOKEN_RE.findall(text)
        normalized = [token for token in tokens if token not in {"the", "a", "an"}]
        while normalized and normalized[0] in {
            "has",
            "have",
            "had",
            "is",
            "are",
            "was",
            "were",
        }:
            normalized.pop(0)
        if len(normalized) > 1 and normalized[-1] == "by":
            normalized.pop()
        return " ".join(normalized)

    def effective_subjects(self, plan: Any, goal: Any) -> list[str]:
        explicit = self._field(goal, "subject")
        if explicit:
            return [explicit]
        goals = list(getattr(plan, "goals", []) or [])
        goal_id = self._field(goal, "goal_id")
        goal_index = next(
            (index for index, item in enumerate(goals) if self._field(item, "goal_id") == goal_id),
            -1,
        )
        if goal_index <= 0:
            return []
        for previous in reversed(goals[:goal_index]):
            values = list(getattr(previous, "resolved_values", []) or [])
            if values:
                return self._dedupe(values)
        return []

    def _object_role_compatible(self, value: str, target: str) -> bool:
        text = normalize_text(value)
        role = self.normalize_relation(target)
        if not text:
            return False
        if role in self._BOOLEAN_TARGETS:
            return text.casefold().rstrip(".") in {"yes", "no", "true", "false"}
        if role in self._NUMBER_TARGETS or any(
            token in role for token in ("count", "number", "quantity", "measurement")
        ):
            return self._number(text) is not None
        if role in self._PERSON_TARGETS or any(
            token in role for token in ("person", "author", "writer", "nominator")
        ):
            return (
                self._number(text) is None
                and text.casefold().rstrip(".") not in {"yes", "no", "true", "false"}
                and len(text.split()) <= 8
            )
        return True

    @staticmethod
    def _number(value: str) -> Decimal | None:
        match = re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[a-zA-Z%^0-9]+)?", value)
        if not match:
            return None
        numeric = re.match(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
        if numeric is None:
            return None
        try:
            return Decimal(numeric.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    def _entity_equivalent(self, first: str, second: str) -> bool:
        """Whether two subject strings name the same entity.

        Compared token by token rather than as raw substrings. Raw containment
        binds anything short to anything longer that happens to spell it: on
        level1_final_14 a contract with subject "I" bound the goal subject
        "Wikipedia", because "i" is inside "wikipedia". That contract --
        "I nominated_by this particular article" -- then counted as a direct
        answer, marked the task sufficient, and stopped retrieval two rounds
        early on a question whose answer was a username.

        A shorter name still binds a longer one, so "Claus" reaches
        "Claus Peter Flor", but only when every one of its tokens appears in
        the other and at least one of them is long enough to identify
        something. That last part is what rejects "I", "it", and other
        single-letter or stopword-length subjects.
        """

        left = self._entity_tokens(first)
        right = self._entity_tokens(second)
        if not left or not right:
            return False
        if left == right:
            return True
        smaller, larger = sorted((left, right), key=len)
        if not smaller <= larger:
            return False
        return any(len(token) >= self._MIN_IDENTIFYING_TOKEN for token in smaller)

    def _entity_tokens(self, value: str) -> set[str]:
        return set(self._TOKEN_RE.findall(normalize_text(value).casefold()))

    def _entity_key(self, value: str) -> str:
        return " ".join(self._TOKEN_RE.findall(normalize_text(value).casefold()))

    def _field(self, value: Any, key: str) -> str:
        if isinstance(value, Mapping):
            result = value.get(key, "")
        else:
            result = getattr(value, key, "")
        return normalize_text(str(result or ""))

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                output.append(cleaned)
                seen.add(key)
        return output

    @staticmethod
    def _result(
        status: str,
        goal_id: str,
        fact_id: str,
        *,
        effective_subject: str = "",
        reason: str,
    ) -> FactGoalBindingResult:
        return FactGoalBindingResult(
            status=status,
            goal_id=goal_id,
            fact_id=fact_id,
            effective_subject=effective_subject,
            reason=reason,
        )


__all__ = ["FactGoalBindingResult", "FactGoalBindingValidator"]
