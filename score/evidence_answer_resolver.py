from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from tools.evidence.fact_extraction import TaskFactStore
from tools.evidence.fact_extraction.answer_bound_validator import (
    AnswerBoundFactValidator,
)
from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)
from tools.search_result_builder.query.relation_plan import RelationPlan
from utils.network_utils import normalize_for_exact, normalize_text


@dataclass(frozen=True)
class EvidenceAnswerResolution:
    """Represent a final answer resolved directly from relation-bound facts."""

    status: str
    answer: str = ""
    supporting_fact_ids: list[str] = field(default_factory=list)
    conflicting_values: list[str] = field(default_factory=list)
    reason: str = ""
    required_relation: str = ""
    required_relation_goal_id: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and bool(self.answer)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceAnswerResolver:
    """Resolve one unambiguous answer without creating an artificial Agent run."""

    def __init__(
        self,
        *,
        binding_validator: FactGoalBindingValidator | None = None,
    ) -> None:
        self.binding_validator = binding_validator or FactGoalBindingValidator()
        self.answer_bound_validator = AnswerBoundFactValidator()

    def resolve(self, evidence: Mapping[str, Any] | None) -> EvidenceAnswerResolution:
        payload = evidence or {}
        required_relation = normalize_text(str(payload.get("required_relation") or ""))
        required_goal_id = normalize_text(
            str(payload.get("required_relation_goal_id") or "")
        )
        store = self._fact_store(payload)
        if not required_relation:
            return self._resolve_unique_promoted_answer(
                store=store,
                answer_requirement=normalize_text(
                    str(payload.get("answer_requirement") or "")
                ),
            )

        relation_plan = RelationPlan.from_dict(
            payload.get("relation_plan")
            if isinstance(payload.get("relation_plan"), dict)
            else None
        )
        if not relation_plan.goals or not relation_plan.complete:
            return EvidenceAnswerResolution(
                status="unresolved_relation_goal",
                reason="final_relation_plan_is_not_complete",
                required_relation=required_relation,
                required_relation_goal_id=required_goal_id,
            )
        final_goal = relation_plan.goals[-1]
        if required_goal_id and final_goal.goal_id != required_goal_id:
            return EvidenceAnswerResolution(
                status="unresolved_relation_goal",
                reason="final_relation_goal_id_mismatch",
                required_relation=required_relation,
                required_relation_goal_id=required_goal_id,
            )

        effective_subjects = self.binding_validator.effective_subjects(
            relation_plan,
            final_goal,
        )
        answer_role = normalize_text(str(payload.get("answer_role") or final_goal.target))
        grouped: dict[str, dict[str, Any]] = {}
        for fact in store.verifiable_answer_facts():
            if fact.polarity != "positive":
                continue
            binding = self.binding_validator.validate(
                fact=fact,
                goal=final_goal,
                effective_subjects=effective_subjects,
                answer_role=answer_role,
            )
            if not binding.bound:
                continue
            answer = normalize_text(fact.object)
            key = normalize_for_exact(answer)
            if not key:
                continue
            bucket = grouped.setdefault(key, {"answer": answer, "fact_ids": []})
            bucket["fact_ids"].append(fact.fact_id)

        if not grouped:
            return EvidenceAnswerResolution(
                status="no_bound_fact",
                reason="no_verifiable_fact_satisfies_final_relation_goal",
                required_relation=required_relation,
                required_relation_goal_id=required_goal_id,
            )
        if len(grouped) > 1:
            return EvidenceAnswerResolution(
                status="conflict",
                conflicting_values=[
                    str(item["answer"]) for item in grouped.values()
                ],
                reason="multiple_relation_bound_answer_values",
                required_relation=required_relation,
                required_relation_goal_id=required_goal_id,
            )

        selected = next(iter(grouped.values()))
        return EvidenceAnswerResolution(
            status="resolved",
            answer=str(selected["answer"]),
            supporting_fact_ids=list(dict.fromkeys(selected["fact_ids"])),
            reason="unique_relation_bound_answer_fact",
            required_relation=required_relation,
            required_relation_goal_id=required_goal_id,
        )

    def _resolve_unique_promoted_answer(
        self,
        *,
        store: TaskFactStore,
        answer_requirement: str,
    ) -> EvidenceAnswerResolution:
        """Resolve only values that passed a strict promotion or trusted handler."""

        accepted_methods = {
            "grounded_answer_value_promotion",
            "trusted_handler",
            "deterministic_adapter",
        }
        grouped: dict[str, dict[str, Any]] = {}
        for fact in store.verifiable_answer_facts():
            if fact.polarity != "positive":
                continue
            if fact.extraction_method not in accepted_methods:
                continue
            if normalize_text(str(fact.qualifiers.get("answer_binding") or "")).casefold() != "direct":
                continue
            compatible, _ = self.answer_bound_validator.value_compatible(
                requirement=answer_requirement,
                value=fact.object,
            )
            if not compatible:
                continue
            answer = normalize_text(fact.object)
            key = normalize_for_exact(answer)
            if not key:
                continue
            bucket = grouped.setdefault(key, {"answer": answer, "fact_ids": []})
            bucket["fact_ids"].append(fact.fact_id)

        if not grouped:
            return EvidenceAnswerResolution(
                status="not_applicable",
                reason="task_has_no_required_relation_or_promoted_answer",
            )
        if len(grouped) > 1:
            return EvidenceAnswerResolution(
                status="conflict",
                conflicting_values=[str(item["answer"]) for item in grouped.values()],
                reason="multiple_promoted_answer_values",
            )
        selected = next(iter(grouped.values()))
        return EvidenceAnswerResolution(
            status="resolved",
            answer=str(selected["answer"]),
            supporting_fact_ids=list(dict.fromkeys(selected["fact_ids"])),
            reason="unique_promoted_answer_fact",
        )

    @staticmethod
    def _fact_store(payload: Mapping[str, Any]) -> TaskFactStore:
        existing = payload.get("_fact_store")
        if isinstance(existing, TaskFactStore):
            return existing
        serialized = payload.get("fact_store")
        return TaskFactStore.from_dict(
            serialized if isinstance(serialized, dict) else None
        )


__all__ = ["EvidenceAnswerResolution", "EvidenceAnswerResolver"]
