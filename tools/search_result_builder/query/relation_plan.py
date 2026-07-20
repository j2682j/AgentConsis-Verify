from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from utils.network_utils import normalize_text

from .source_requirement import REQUIRED_CONTENT_TYPES, SOURCE_KINDS


GOAL_STATES = frozenset({"pending", "active", "resolved", "blocked"})
GOAL_POLARITIES = frozenset({"positive", "negative"})
VERIFICATION_SCOPES = frozenset({"passage", "full_document", "collection"})


@dataclass(frozen=True)
class RelationGoal:
    """
    表示多跳檢索中必須解析的一個自然語言關係目標。

    Args:
     - goal_id: Relation goal 的穩定識別碼。
     - subject: 目前已知的關係主體，空值表示沿用上一個 goal 的結果。
     - relation: 要從證據確認的自然語言關係。
     - target: 預期解析出的實體或資訊角色。
     - source_kind: 適合取得此關係的來源型態。
     - state: pending、active、resolved 或 blocked。
     - resolved_values: 已由原文證據綁定的 relation objects。
     - evidence_ids: 支撐 resolved values 的文件 IDs。

    Returns:
     - RelationGoal: 可在檢索輪次間更新的關係狀態。
    """

    goal_id: str
    subject: str
    relation: str
    target: str
    source_kind: str = "web"
    polarity: str = "positive"
    verification_scope: str = "passage"
    required_content: str = "html_text"
    state: str = "pending"
    resolved_values: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> "RelationGoal":
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any] | None,
        *,
        goal_id: str = "",
        default_state: str = "pending",
    ) -> "RelationGoal | None":
        data = dict(value or {})
        subject = normalize_text(str(data.get("subject") or ""))
        relation = normalize_text(str(data.get("relation") or ""))
        target = normalize_text(str(data.get("target") or ""))
        if not relation or not target:
            return None
        source_kind = normalize_text(str(data.get("source_kind") or "web")).lower()
        if source_kind not in SOURCE_KINDS:
            source_kind = "web"
        polarity = normalize_text(str(data.get("polarity") or "positive")).lower()
        if polarity not in GOAL_POLARITIES:
            polarity = "positive"
        verification_scope = normalize_text(
            str(data.get("verification_scope") or "passage")
        ).lower()
        if verification_scope not in VERIFICATION_SCOPES:
            verification_scope = "passage"
        required_content = normalize_text(
            str(data.get("required_content") or "html_text")
        ).lower()
        if required_content not in REQUIRED_CONTENT_TYPES:
            required_content = "html_text"
        state = normalize_text(str(data.get("state") or default_state)).lower()
        if state not in GOAL_STATES:
            state = default_state
        return cls(
            goal_id=normalize_text(str(data.get("goal_id") or goal_id)),
            subject=subject,
            relation=relation,
            target=target,
            source_kind=source_kind,
            polarity=polarity,
            verification_scope=verification_scope,
            required_content=required_content,
            state=state,
            resolved_values=_dedupe(data.get("resolved_values") or []),
            evidence_ids=_dedupe(data.get("evidence_ids") or []),
        )


@dataclass(frozen=True)
class RelationPlan:
    """
    保存依序執行的 relation goals 與目前 active goal。

    Args:
     - goals: 最多三個依序排列的 relation goals。
     - active_goal_id: 目前檢索輪次正在解析的 goal id。

    Returns:
     - RelationPlan: Relation-aware retrieval 的最小狀態容器。
    """

    goals: list[RelationGoal] = field(default_factory=list)
    active_goal_id: str = ""

    @property
    def active_goal(self) -> RelationGoal | None:
        for goal in self.goals:
            if goal.goal_id == self.active_goal_id:
                return goal
        return next((goal for goal in self.goals if goal.state == "active"), None)

    @property
    def enabled(self) -> bool:
        return bool(self.goals and self.active_goal is not None)

    @property
    def complete(self) -> bool:
        return bool(self.goals) and all(goal.state == "resolved" for goal in self.goals)

    @property
    def is_multihop(self) -> bool:
        return len(self.goals) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": [goal.to_dict() for goal in self.goals],
            "active_goal_id": self.active_goal_id,
        }

    def replace_goal(self, updated_goal: RelationGoal) -> "RelationPlan":
        return RelationPlan(
            goals=[
                updated_goal if goal.goal_id == updated_goal.goal_id else goal
                for goal in self.goals
            ],
            active_goal_id=self.active_goal_id,
        )

    def replace(self, **changes: Any) -> "RelationPlan":
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RelationPlan":
        data = dict(value or {})
        goals: list[RelationGoal] = []
        for index, item in enumerate(list(data.get("goals") or [])[:6], start=1):
            if not isinstance(item, dict):
                continue
            goal = RelationGoal.from_dict(item, goal_id=f"G{index}")
            if goal is not None:
                goals.append(goal)
        if not goals:
            return cls()
        active_goal_id = normalize_text(str(data.get("active_goal_id") or ""))
        if active_goal_id not in {goal.goal_id for goal in goals}:
            active_goal_id = next(
                (goal.goal_id for goal in goals if goal.state == "active"),
                goals[0].goal_id,
            )
        normalized_goals = [
            goal.replace(
                state=(
                    "active"
                    if goal.goal_id == active_goal_id and goal.state != "resolved"
                    else goal.state
                )
            )
            for goal in goals
        ]
        return cls(goals=normalized_goals, active_goal_id=active_goal_id)

    @classmethod
    def from_specs(cls, values: Iterable[dict[str, Any]]) -> "RelationPlan":
        goals: list[RelationGoal] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(values)[:6]:
            if not isinstance(item, dict):
                continue
            goal = RelationGoal.from_dict(
                item,
                goal_id=f"G{len(goals) + 1}",
                default_state="pending",
            )
            if goal is None:
                continue
            key = (
                goal.subject.casefold(),
                goal.relation.casefold(),
                goal.target.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            goals.append(goal)
        if not goals:
            return cls()
        goals[0] = goals[0].replace(state="active")
        return cls(goals=goals, active_goal_id=goals[0].goal_id)


def _dedupe(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_text(str(value or ""))
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        output.append(cleaned)
        seen.add(key)
    return output


__all__ = [
    "GOAL_POLARITIES",
    "GOAL_STATES",
    "VERIFICATION_SCOPES",
    "RelationGoal",
    "RelationPlan",
]
