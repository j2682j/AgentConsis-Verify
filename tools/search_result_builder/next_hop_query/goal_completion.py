from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from utils.network_utils import normalize_text

from ..query.relation_plan import RelationGoal, RelationPlan
from ..source_analyze.full_document_verifier import (
    FullDocumentVerifier,
    NegativeVerificationResult,
)


@dataclass(frozen=True)
class GoalCompletionEntry:
    """Describe the current completion state of one retrieval goal."""

    goal_id: str
    state: str
    support_kind: str
    evidence_ids: list[str] = field(default_factory=list)
    checked_scope: str = "passage"
    missing_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalCompletionResult:
    """Represent the only state that is allowed to stop retrieval as sufficient."""

    sufficient: bool
    direct_answer_found: bool
    relation_plan: RelationPlan
    goals: list[GoalCompletionEntry] = field(default_factory=list)
    unresolved_goal_ids: list[str] = field(default_factory=list)
    negative_verifications: list[NegativeVerificationResult] = field(
        default_factory=list
    )
    recovery_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "direct_answer_found": self.direct_answer_found,
            "relation_plan": self.relation_plan.to_dict(),
            "goals": [item.to_dict() for item in self.goals],
            "unresolved_goal_ids": list(self.unresolved_goal_ids),
            "negative_verifications": [
                item.to_dict() for item in self.negative_verifications
            ],
            "recovery_reason": self.recovery_reason,
        }


class GoalCompletionEvaluator:
    """Require completed relation goals and grounded direct answer evidence."""

    def __init__(self, verifier: FullDocumentVerifier | None = None) -> None:
        self.verifier = verifier or FullDocumentVerifier()

    def evaluate(
        self,
        *,
        relation_plan: RelationPlan | None,
        documents: Iterable[Any],
        corpus_documents: Iterable[Mapping[str, Any]] = (),
        answer_gate_sufficient: bool = False,
    ) -> GoalCompletionResult:
        observed = list(documents)
        corpus = list(corpus_documents)
        plan = relation_plan or RelationPlan()
        negative_results: list[NegativeVerificationResult] = []

        for goal in plan.goals:
            if goal.polarity != "negative" or goal.state != "active":
                continue
            effective_goal = self._effective_goal(plan, goal.goal_id)
            result = self.verifier.verify(
                goal=effective_goal,
                documents=observed,
                corpus_documents=corpus,
            )
            negative_results.append(result)
            if result.resolved:
                plan = self._advance_plan(
                    plan.replace_goal(
                        goal.replace(
                            state="resolved",
                            resolved_values=result.resolved_values,
                            evidence_ids=result.evidence_ids,
                        )
                    )
                )

        direct_contracts = [
            contract
            for document in observed
            for contract in list(self._field(document, "direct_contracts", []) or [])
        ]
        final_goal = plan.goals[-1] if plan.goals else None
        positive_direct = bool(direct_contracts) and (
            final_goal is None
            or any(
                normalize_text(self._contract_field(item, "goal_id"))
                in {"", final_goal.goal_id}
                for item in direct_contracts
            )
        )
        relation_bound_direct = bool(
            final_goal is not None
            and any(
                normalize_text(self._contract_field(item, "goal_id"))
                == final_goal.goal_id
                for item in direct_contracts
            )
        )
        negative_direct = any(item.resolved for item in negative_results)
        direct_answer_found = bool(
            (
                positive_direct
                and (answer_gate_sufficient or relation_bound_direct)
            )
            or negative_direct
        )

        entries: list[GoalCompletionEntry] = []
        negative_by_goal = {item.goal_id: item for item in negative_results}
        for goal in plan.goals:
            negative = negative_by_goal.get(goal.goal_id)
            support_kind = "negative_full_document" if negative else "relation"
            missing_reason = ""
            if goal.state != "resolved":
                missing_reason = (
                    negative.missing_reason
                    if negative is not None
                    else "relation_not_resolved"
                )
            entries.append(
                GoalCompletionEntry(
                    goal_id=goal.goal_id,
                    state=goal.state,
                    support_kind=support_kind,
                    evidence_ids=list(goal.evidence_ids),
                    checked_scope=goal.verification_scope,
                    missing_reason=missing_reason,
                )
            )

        unresolved = [goal.goal_id for goal in plan.goals if goal.state != "resolved"]
        goals_complete = not unresolved
        if plan.goals:
            sufficient = goals_complete and direct_answer_found
        else:
            sufficient = direct_answer_found
        recovery_reason = ""
        if not sufficient:
            recovery_reason = (
                "unresolved_relation_goals" if unresolved else "missing_direct_evidence"
            )
        return GoalCompletionResult(
            sufficient=sufficient,
            direct_answer_found=direct_answer_found,
            relation_plan=plan,
            goals=entries,
            unresolved_goal_ids=unresolved,
            negative_verifications=negative_results,
            recovery_reason=recovery_reason,
        )

    def _field(self, value: Any, key: str, default: Any = "") -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    def _contract_field(self, value: Any, key: str) -> str:
        return str(self._field(value, key, "") or "")

    def _advance_plan(self, plan: RelationPlan) -> RelationPlan:
        pending = next((goal for goal in plan.goals if goal.state == "pending"), None)
        if pending is None:
            return plan.replace(active_goal_id="")
        return RelationPlan(
            goals=[
                goal.replace(state="active") if goal.goal_id == pending.goal_id else goal
                for goal in plan.goals
            ],
            active_goal_id=pending.goal_id,
        )

    def _effective_goal(self, plan: RelationPlan, goal_id: str) -> RelationGoal:
        goal_index = next(
            (index for index, goal in enumerate(plan.goals) if goal.goal_id == goal_id),
            -1,
        )
        goal = plan.goals[goal_index]
        if normalize_text(goal.subject) or goal_index <= 0:
            return goal
        prior_values = [
            value
            for prior_goal in plan.goals[:goal_index]
            for value in prior_goal.resolved_values
        ]
        if not prior_values:
            return goal
        return goal.replace(subject=" ".join(prior_values[-2:]))


__all__ = [
    "GoalCompletionEntry",
    "GoalCompletionEvaluator",
    "GoalCompletionResult",
]
