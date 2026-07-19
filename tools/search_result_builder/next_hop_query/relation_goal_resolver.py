from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from utils.network_utils import normalize_text
from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)

from ..query.relation_plan import RelationGoal, RelationPlan
from .relation_evidence import RelationEvidence


@dataclass(frozen=True)
class RelationResolution:
    """Store one deterministic relation-plan state transition."""

    plan: RelationPlan
    resolved_goal_ids: list[str] = field(default_factory=list)
    activated_goal_id: str = ""
    transitions: list[dict[str, str]] = field(default_factory=list)
    rejected_contracts: list[dict[str, str]] = field(default_factory=list)


class RelationGoalResolver:
    """Resolve the active relation goal and activate its dependent successor."""

    def __init__(
        self,
        *,
        fact_goal_binding_validator: FactGoalBindingValidator | None = None,
    ) -> None:
        self.fact_goal_binding_validator = (
            fact_goal_binding_validator or FactGoalBindingValidator()
        )

    def effective_subjects(self, plan: RelationPlan) -> list[str]:
        active = plan.active_goal
        if active is None:
            return []
        if normalize_text(active.subject):
            return [normalize_text(active.subject)]

        active_index = next(
            (index for index, goal in enumerate(plan.goals) if goal.goal_id == active.goal_id),
            -1,
        )
        for goal in reversed(plan.goals[:active_index]):
            values = self._dedupe(goal.resolved_values)
            if values:
                return values
        return []

    def resolve(
        self,
        plan: RelationPlan,
        evidence: Iterable[RelationEvidence],
    ) -> RelationResolution:
        active = plan.active_goal
        if active is None:
            return RelationResolution(plan=plan)
        if active.state == "resolved":
            return RelationResolution(plan=plan)
        if active.polarity == "negative":
            return RelationResolution(plan=plan)

        matched = [item for item in evidence if item.goal_id == active.goal_id]
        values = self._dedupe(item.object for item in matched)
        if not values:
            return RelationResolution(plan=plan)

        evidence_ids = self._dedupe(item.document_id for item in matched)
        updated_active = active.replace(
            state="resolved",
            resolved_values=self._dedupe([*active.resolved_values, *values]),
            evidence_ids=self._dedupe([*active.evidence_ids, *evidence_ids]),
        )
        goals = [
            updated_active if goal.goal_id == active.goal_id else goal
            for goal in plan.goals
        ]
        next_goal = next((goal for goal in goals if goal.state == "pending"), None)
        activated_goal_id = ""
        if next_goal is not None:
            activated_goal_id = next_goal.goal_id
            goals = [
                goal.replace(state="active") if goal.goal_id == next_goal.goal_id else goal
                for goal in goals
            ]
        updated_plan = RelationPlan(
            goals=goals,
            active_goal_id=activated_goal_id,
        )
        return RelationResolution(
            plan=updated_plan,
            resolved_goal_ids=[active.goal_id],
            activated_goal_id=activated_goal_id,
            transitions=[
                {
                    "goal_id": active.goal_id,
                    "from_state": active.state,
                    "to_state": "resolved",
                    "resolution_type": "bridge",
                    "activated_goal_id": activated_goal_id,
                }
            ],
        )

    def resolve_direct(
        self,
        plan: RelationPlan,
        contracts: Iterable[Any],
    ) -> RelationResolution:
        """Resolve only the active goal from its explicitly bound Direct contracts."""
        active = plan.active_goal
        if active is None or active.state == "resolved":
            return RelationResolution(plan=plan)
        if active.polarity == "negative":
            return RelationResolution(plan=plan)
        values: list[str] = []
        evidence_ids: list[str] = []
        rejected_contracts: list[dict[str, str]] = []
        effective_subjects = self.effective_subjects(plan)
        for contract in contracts:
            if isinstance(contract, dict):
                getter = contract.get
            else:
                getter = lambda key, default="": getattr(contract, key, default)
            if normalize_text(str(getter("goal_id", ""))) != active.goal_id:
                continue
            fact_payload = {
                "fact_id": normalize_text(str(getter("fact_id", ""))),
                "goal_id": normalize_text(str(getter("goal_id", ""))),
                "subject": normalize_text(str(getter("subject", ""))),
                "relation": normalize_text(str(getter("relation", ""))),
                "object": normalize_text(
                    str(getter("object", "") or getter("answer_span", ""))
                ),
                "grounding_status": normalize_text(
                    str(getter("grounding_status", ""))
                ),
            }
            binding = self.fact_goal_binding_validator.validate(
                fact=fact_payload,
                goal=active,
                effective_subjects=effective_subjects,
                answer_role=active.target,
            )
            if not binding.bound:
                rejected_contracts.append(
                    {
                        **binding.to_dict(),
                        "document_id": normalize_text(
                            str(getter("document_id", ""))
                        ),
                        "answer_span": normalize_text(
                            str(getter("answer_span", ""))
                        ),
                    }
                )
                continue
            values.append(fact_payload["object"])
            evidence_ids.append(normalize_text(str(getter("document_id", ""))))
        values = self._dedupe(values)
        if not values:
            return RelationResolution(
                plan=plan,
                rejected_contracts=rejected_contracts,
            )

        updated_active = active.replace(
            state="resolved",
            resolved_values=self._dedupe([*active.resolved_values, *values]),
            evidence_ids=self._dedupe([*active.evidence_ids, *evidence_ids]),
        )
        goals = [
            updated_active if goal.goal_id == active.goal_id else goal
            for goal in plan.goals
        ]
        next_goal = next((goal for goal in goals if goal.state == "pending"), None)
        activated_goal_id = ""
        if next_goal is not None:
            activated_goal_id = next_goal.goal_id
            goals = [
                goal.replace(state="active") if goal.goal_id == next_goal.goal_id else goal
                for goal in goals
            ]
        return RelationResolution(
            plan=RelationPlan(goals=goals, active_goal_id=activated_goal_id),
            resolved_goal_ids=[active.goal_id],
            activated_goal_id=activated_goal_id,
            transitions=[
                {
                    "goal_id": active.goal_id,
                    "from_state": active.state,
                    "to_state": "resolved",
                    "resolution_type": "direct",
                    "activated_goal_id": activated_goal_id,
                }
            ],
            rejected_contracts=rejected_contracts,
        )

    def _dedupe(self, values: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            output.append(cleaned)
            seen.add(key)
        return output


__all__ = ["RelationGoalResolver", "RelationResolution"]
