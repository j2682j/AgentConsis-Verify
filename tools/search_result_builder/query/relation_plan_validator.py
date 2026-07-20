from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from utils.network_utils import normalize_text

from .question_role_extractor import QuestionRole
from .relation_plan import RelationGoal, RelationPlan


@dataclass(frozen=True)
class RelationPlanValidationResult:
    """保存 RelationPlan 的結構驗證與安全修復結果。"""

    valid: bool
    plan: RelationPlan
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "plan": self.plan.to_dict(),
            "errors": list(self.errors),
            "repairs": list(self.repairs),
        }


class RelationPlanValidator:
    """驗證 relation goal 的方向、必要欄位與相依結構。"""

    _ANSWER_PLACEHOLDERS = {
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "when",
        "where",
        "how",
        "answer",
        "person",
        "unknown",
    }

    def validate(
        self,
        plan: RelationPlan,
        *,
        question_role: QuestionRole | None = None,
    ) -> RelationPlanValidationResult:
        role = question_role or QuestionRole()
        errors: list[str] = []
        repairs: list[str] = []
        goals: list[RelationGoal] = []
        seen: set[tuple[str, str, str]] = set()

        for index, original in enumerate(plan.goals[:6]):
            goal = original.replace(
                subject=normalize_text(original.subject),
                relation=self.normalize_relation(original.relation),
                target=normalize_text(original.target),
            )
            if not goal.relation:
                errors.append(f"{goal.goal_id}:missing_relation")
                continue
            if not goal.target:
                errors.append(f"{goal.goal_id}:missing_target")
                continue

            if self._is_placeholder(goal.subject):
                if not self._is_placeholder(goal.target):
                    answer_target = self._answer_target(role)
                    goal = goal.replace(
                        subject=goal.target,
                        target=answer_target,
                    )
                    repairs.append(
                        f"{goal.goal_id}:swap_answer_placeholder_subject_with_target"
                    )
                else:
                    errors.append(
                        f"{goal.goal_id}:answer_placeholder_used_as_subject"
                    )
                    continue

            if index > 0 and goal.subject and self._is_placeholder(goal.subject):
                errors.append(f"{goal.goal_id}:invalid_dependency_subject")
                continue

            key = (
                self._key(goal.subject),
                self._key(goal.relation),
                self._key(goal.target),
            )
            if key in seen:
                repairs.append(f"{goal.goal_id}:remove_duplicate_relation_goal")
                continue
            seen.add(key)
            goals.append(goal)

        normalized = self._normalize_states(goals, plan.active_goal_id)
        if plan.goals and not normalized.goals and not errors:
            errors.append("empty_relation_plan_after_validation")
        return RelationPlanValidationResult(
            valid=not errors and (not plan.goals or bool(normalized.goals)),
            plan=normalized,
            errors=self._dedupe(errors),
            repairs=self._dedupe(repairs),
        )

    @staticmethod
    def normalize_relation(value: str) -> str:
        text = normalize_text(value).casefold().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _is_placeholder(self, value: str) -> bool:
        key = self._key(value)
        if not key:
            return False
        return key in self._ANSWER_PLACEHOLDERS

    def _answer_target(self, role: QuestionRole) -> str:
        answer_role = normalize_text(role.answer_role).casefold()
        if answer_role and answer_role != "unknown":
            return answer_role
        return "answer"

    def _normalize_states(
        self,
        goals: list[RelationGoal],
        requested_active_goal_id: str,
    ) -> RelationPlan:
        if not goals:
            return RelationPlan()
        active_id = normalize_text(requested_active_goal_id)
        valid_ids = {goal.goal_id for goal in goals}
        if active_id not in valid_ids:
            active_id = next(
                (goal.goal_id for goal in goals if goal.state == "active"),
                goals[0].goal_id,
            )
        normalized = [
            goal.replace(
                state=(
                    "active"
                    if goal.goal_id == active_id and goal.state != "resolved"
                    else "pending" if goal.state == "active" else goal.state
                )
            )
            for goal in goals
        ]
        return RelationPlan(goals=normalized, active_goal_id=active_id)

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", normalize_text(value).casefold()))

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))


__all__ = ["RelationPlanValidationResult", "RelationPlanValidator"]
