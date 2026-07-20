from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from utils.network_utils import normalize_text
from tools.evidence.fact_extraction.models import EvidenceFact
from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)

from ..query.relation_plan import RelationPlan


@dataclass(frozen=True)
class DirectEvidenceContract:
    """Authorize one grounded span for final-answer evidence conversion."""

    goal_id: str
    answer_span: str
    context: str
    document_id: str
    source_title: str
    url: str
    answer_requirement: str
    fact_id: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    polarity: str = "positive"
    grounding_status: str = ""
    contract_method: str = "semantic_fact"
    origin_fact_id: str = ""
    scope_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeEvidenceContract:
    """Authorize one grounded span for resolving an intermediate relation goal."""

    goal_id: str
    bridge_span: str
    context: str
    document_id: str
    source_title: str
    url: str
    next_goal_id: str
    fact_id: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    polarity: str = "positive"
    grounding_status: str = ""
    contract_method: str = "semantic_fact"
    origin_fact_id: str = ""
    scope_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedEvidenceSpan:
    """Record why a classified span did not receive downstream authority."""

    span: str
    role: str
    document_id: str
    reason: str
    goal_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRoleContracts:
    """Keep mutually exclusive Direct, Bridge, and rejected span contracts."""

    direct: list[DirectEvidenceContract] = field(default_factory=list)
    bridge: list[BridgeEvidenceContract] = field(default_factory=list)
    unsupported: list[RejectedEvidenceSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct": [item.to_dict() for item in self.direct],
            "bridge": [item.to_dict() for item in self.bridge],
            "unsupported": [item.to_dict() for item in self.unsupported],
        }


class EvidenceRoleContractBuilder:
    """Convert grounded classifier roles into non-overlapping authority contracts."""

    def __init__(
        self,
        *,
        fact_goal_binding_validator: FactGoalBindingValidator | None = None,
    ) -> None:
        self.fact_goal_binding_validator = (
            fact_goal_binding_validator or FactGoalBindingValidator()
        )

    def build(
        self,
        *,
        question: str,
        answer_requirement: str,
        answer_target: str,
        relation_plan: RelationPlan | None,
        document_id: str,
        source_title: str,
        url: str,
        text: str,
        direct_spans: Iterable[str] = (),
        bridge_spans: Iterable[str] = (),
        span_assignments: Iterable[dict[str, Any]] | None = None,
    ) -> EvidenceRoleContracts:
        source_text = normalize_text(text)
        requirement = (
            normalize_text(answer_requirement)
            or normalize_text(answer_target)
            or normalize_text(question)
        )
        plan = relation_plan or RelationPlan()
        if span_assignments is not None:
            return self._build_goal_assigned(
                requirement=requirement,
                plan=plan,
                document_id=document_id,
                source_title=source_title,
                url=url,
                source_text=source_text,
                span_assignments=span_assignments,
            )
        active_goal = plan.active_goal
        final_goal = plan.goals[-1] if plan.goals else None
        next_goal = self._next_pending_goal(plan)

        direct: list[DirectEvidenceContract] = []
        bridge: list[BridgeEvidenceContract] = []
        unsupported: list[RejectedEvidenceSpan] = []
        direct_keys: set[str] = set()

        for span in self._dedupe(direct_spans):
            context = self._grounded_context(source_text, span)
            if not context:
                unsupported.append(
                    RejectedEvidenceSpan(span, "DIRECT", document_id, "span_not_grounded")
                )
                continue
            direct.append(
                DirectEvidenceContract(
                    goal_id=final_goal.goal_id if final_goal is not None else "",
                    answer_span=span,
                    context=context,
                    document_id=document_id,
                    source_title=normalize_text(source_title),
                    url=normalize_text(url),
                    answer_requirement=requirement,
                )
            )
            direct_keys.add(span.casefold())

        for span in self._dedupe(bridge_spans):
            if span.casefold() in direct_keys:
                unsupported.append(
                    RejectedEvidenceSpan(
                        span,
                        "BRIDGE",
                        document_id,
                        "direct_contract_precedence",
                    )
                )
                continue
            context = self._grounded_context(source_text, span)
            if not context:
                unsupported.append(
                    RejectedEvidenceSpan(span, "BRIDGE", document_id, "span_not_grounded")
                )
                continue
            if active_goal is None or next_goal is None:
                unsupported.append(
                    RejectedEvidenceSpan(
                        span,
                        "BRIDGE",
                        document_id,
                        "missing_active_or_next_goal",
                    )
                )
                continue
            bridge.append(
                BridgeEvidenceContract(
                    goal_id=active_goal.goal_id,
                    bridge_span=span,
                    context=context,
                    document_id=document_id,
                    source_title=normalize_text(source_title),
                    url=normalize_text(url),
                    next_goal_id=next_goal.goal_id,
                )
            )
        return EvidenceRoleContracts(
            direct=direct,
            bridge=bridge,
            unsupported=unsupported,
        )

    def build_from_grounded_facts(
        self,
        *,
        question: str,
        answer_requirement: str,
        answer_target: str,
        relation_plan: RelationPlan | None,
        document_id: str,
        source_title: str,
        url: str,
        text: str,
        facts: Iterable[EvidenceFact | dict[str, Any]],
    ) -> EvidenceRoleContracts:
        """Promote already grounded, answer-bound facts into direct contracts.

        This entry point is intentionally independent from span finalization. A
        semantic extractor may recover a precise answer value even when the
        classifier span is wider than the value or the finalizer rejects that
        wider span. The fact still has to pass grounding, answer binding, and
        relation-goal binding before it receives evidence authority.
        """

        source_text = normalize_text(text)
        requirement = (
            normalize_text(answer_requirement)
            or normalize_text(answer_target)
            or normalize_text(question)
        )
        plan = relation_plan or RelationPlan()
        goals = {goal.goal_id: goal for goal in plan.goals}
        direct: list[DirectEvidenceContract] = []
        unsupported: list[RejectedEvidenceSpan] = []
        seen: set[tuple[str, str, str]] = set()

        for value in facts:
            fact = value if isinstance(value, EvidenceFact) else EvidenceFact.from_dict(value)
            span = normalize_text(fact.object).strip(" \"'`.,;:")
            if (
                fact.grounding_status != "grounded"
                or fact.role != "ANSWER_SUPPORT"
                or fact.qualifiers.get("answer_binding") != "direct"
            ):
                continue
            if not span:
                unsupported.append(
                    RejectedEvidenceSpan(
                        "",
                        "ANSWER_SUPPORT",
                        document_id,
                        "direct_fact_object_empty",
                        fact.goal_id,
                    )
                )
                continue

            goal_id = normalize_text(fact.goal_id)
            matching_goal = goals.get(goal_id)
            if plan.goals and matching_goal is None:
                unsupported.append(
                    RejectedEvidenceSpan(
                        span,
                        "ANSWER_SUPPORT",
                        document_id,
                        "direct_fact_goal_unbound",
                        goal_id,
                    )
                )
                continue
            if matching_goal is not None:
                binding = self.fact_goal_binding_validator.validate(
                    fact=fact,
                    goal=matching_goal,
                    effective_subjects=self.fact_goal_binding_validator.effective_subjects(
                        plan,
                        matching_goal,
                    ),
                    answer_role=(
                        matching_goal.target
                        if matching_goal == plan.goals[-1]
                        else ""
                    ),
                )
                if not binding.bound:
                    unsupported.append(
                        RejectedEvidenceSpan(
                            span,
                            "ANSWER_SUPPORT",
                            document_id,
                            f"direct_fact_{binding.status}",
                            goal_id,
                        )
                    )
                    continue

            grounded_span = self._restore_source_span(source_text, span)
            if not grounded_span:
                unsupported.append(
                    RejectedEvidenceSpan(
                        span,
                        "ANSWER_SUPPORT",
                        document_id,
                        "direct_fact_value_not_grounded",
                        goal_id,
                    )
                )
                continue
            context = self._fact_context(fact.to_dict(), source_text)
            if not context:
                context = self._grounded_context(source_text, grounded_span)
            if not context:
                unsupported.append(
                    RejectedEvidenceSpan(
                        grounded_span,
                        "ANSWER_SUPPORT",
                        document_id,
                        "direct_fact_context_not_grounded",
                        goal_id,
                    )
                )
                continue

            key = (goal_id, grounded_span.casefold(), fact.fact_id)
            if key in seen:
                continue
            seen.add(key)
            direct.append(
                DirectEvidenceContract(
                    goal_id=goal_id,
                    answer_span=grounded_span,
                    context=context,
                    document_id=normalize_text(document_id),
                    source_title=normalize_text(source_title),
                    url=normalize_text(url),
                    answer_requirement=requirement,
                    **self._fact_contract_fields(fact.to_dict()),
                )
            )

        return EvidenceRoleContracts(
            direct=direct,
            bridge=[],
            unsupported=unsupported,
        )

    def _build_goal_assigned(
        self,
        *,
        requirement: str,
        plan: RelationPlan,
        document_id: str,
        source_title: str,
        url: str,
        source_text: str,
        span_assignments: Iterable[dict[str, Any]],
    ) -> EvidenceRoleContracts:
        valid_goal_ids = {goal.goal_id for goal in plan.goals}
        direct: list[DirectEvidenceContract] = []
        bridge: list[BridgeEvidenceContract] = []
        unsupported: list[RejectedEvidenceSpan] = []
        direct_keys: set[tuple[str, str]] = set()
        seen: set[tuple[str, str, str]] = set()

        normalized: list[tuple[str, str, str, str, dict[str, Any]]] = []
        for item in span_assignments:
            if not isinstance(item, dict) or not bool(item.get("accepted", False)):
                continue
            role = normalize_text(str(item.get("role", ""))).upper()
            goal_id = normalize_text(str(item.get("goal_id", "")))
            original = normalize_text(
                str(item.get("original_text") or item.get("text") or "")
            ).strip(" \"'`.,;:")
            finalized = normalize_text(
                str(item.get("finalized_text") or original)
            ).strip(" \"'`.,;:")
            key = (goal_id, role, original.casefold())
            if not original or key in seen:
                continue
            semantic_fact = self._grounded_fact(
                list(item.get("semantic_facts") or []),
                role=role,
                goal_id=goal_id,
            )
            normalized.append((role, goal_id, original, finalized, semantic_fact))
            seen.add(key)

        for role, goal_id, original, finalized, semantic_fact in normalized:
            if role not in {"ANSWER_SUPPORT", "BRIDGE"}:
                continue
            if plan.goals and not goal_id:
                unsupported.append(
                    RejectedEvidenceSpan(
                        original,
                        role,
                        document_id,
                        "missing_goal_id",
                    )
                )
                continue
            if goal_id and goal_id not in valid_goal_ids:
                unsupported.append(
                    RejectedEvidenceSpan(
                        original,
                        role,
                        document_id,
                        "unknown_goal_id",
                        goal_id,
                    )
                )
                continue
            contract_span = self._fact_span(semantic_fact, original, source_text)
            context = self._fact_context(semantic_fact, source_text)
            if not context:
                context = self._grounded_context(source_text, contract_span)
            if not context and finalized != original:
                context = self._grounded_context(source_text, finalized)
            if not context:
                unsupported.append(
                    RejectedEvidenceSpan(
                        original,
                        role,
                        document_id,
                        "span_not_grounded",
                        goal_id,
                    )
                )
                continue
            if role == "ANSWER_SUPPORT":
                if not semantic_fact:
                    unsupported.append(
                        RejectedEvidenceSpan(
                            original,
                            role,
                            document_id,
                            "missing_grounded_answer_fact",
                            goal_id,
                        )
                    )
                    continue
                if not self._is_answer_bound_fact(semantic_fact):
                    unsupported.append(
                        RejectedEvidenceSpan(
                            original,
                            role,
                            document_id,
                            "fact_not_answer_bound",
                            goal_id,
                        )
                    )
                    continue
                matching_goal = next(
                    (goal for goal in plan.goals if goal.goal_id == goal_id),
                    None,
                )
                if matching_goal is not None:
                    binding = self.fact_goal_binding_validator.validate(
                        fact=semantic_fact,
                        goal=matching_goal,
                        effective_subjects=self.fact_goal_binding_validator.effective_subjects(
                            plan,
                            matching_goal,
                        ),
                        answer_role=(
                            matching_goal.target
                            if matching_goal == plan.goals[-1]
                            else ""
                        ),
                    )
                    if not binding.bound:
                        unsupported.append(
                            RejectedEvidenceSpan(
                                original,
                                role,
                                document_id,
                                f"fact_{binding.status}",
                                goal_id,
                            )
                        )
                        continue
                direct.append(
                    DirectEvidenceContract(
                        goal_id=goal_id,
                        answer_span=contract_span,
                        context=context,
                        document_id=document_id,
                        source_title=normalize_text(source_title),
                        url=normalize_text(url),
                        answer_requirement=requirement,
                        **self._fact_contract_fields(semantic_fact),
                    )
                )
                direct_keys.add((goal_id, contract_span.casefold()))

        for role, goal_id, original, finalized, semantic_fact in normalized:
            if role != "BRIDGE":
                continue
            bridge_candidate = self._fact_span(semantic_fact, finalized or original, source_text)
            if (goal_id, bridge_candidate.casefold()) in direct_keys:
                unsupported.append(
                    RejectedEvidenceSpan(
                        original,
                        role,
                        document_id,
                        "direct_contract_precedence",
                        goal_id,
                    )
                )
                continue
            if plan.goals and not goal_id:
                continue
            if goal_id and goal_id not in valid_goal_ids:
                continue
            next_goal = self._next_pending_goal_after(plan, goal_id)
            if not plan.goals or next_goal is None:
                unsupported.append(
                    RejectedEvidenceSpan(
                        original,
                        role,
                        document_id,
                        "missing_active_or_next_goal",
                        goal_id,
                    )
                )
                continue
            bridge_text = bridge_candidate
            context = self._fact_context(semantic_fact, source_text)
            if not context:
                context = self._grounded_context(source_text, original)
            if not context and bridge_text != original:
                context = self._grounded_context(source_text, bridge_text)
            if not context:
                continue
            bridge.append(
                BridgeEvidenceContract(
                    goal_id=goal_id,
                    bridge_span=bridge_text,
                    context=context,
                    document_id=document_id,
                    source_title=normalize_text(source_title),
                    url=normalize_text(url),
                    next_goal_id=next_goal.goal_id,
                    **self._fact_contract_fields(semantic_fact),
                )
            )
        return EvidenceRoleContracts(
            direct=direct,
            bridge=bridge,
            unsupported=unsupported,
        )

    def _grounded_fact(
        self,
        values: list[Any],
        *,
        role: str,
        goal_id: str,
    ) -> dict[str, Any]:
        for value in values:
            if not isinstance(value, dict):
                continue
            if normalize_text(str(value.get("grounding_status") or "")) != "grounded":
                continue
            fact_role = normalize_text(str(value.get("role") or "")).upper()
            fact_goal = normalize_text(str(value.get("goal_id") or ""))
            if fact_role != role or (goal_id and fact_goal and fact_goal != goal_id):
                continue
            return dict(value)
        return {}

    def _fact_span(
        self,
        fact: dict[str, Any],
        fallback: str,
        source_text: str,
    ) -> str:
        candidate = normalize_text(str(fact.get("object") or ""))
        if candidate and candidate.casefold() in source_text.casefold():
            return candidate
        return normalize_text(fallback)

    @staticmethod
    def _restore_source_span(source_text: str, span: str) -> str:
        cleaned = normalize_text(span)
        if not source_text or not cleaned:
            return ""
        index = source_text.casefold().find(cleaned.casefold())
        if index < 0:
            return ""
        return source_text[index : index + len(cleaned)]

    def _fact_context(self, fact: dict[str, Any], source_text: str) -> str:
        fact_context = normalize_text(str(fact.get("context") or ""))
        if fact_context and fact_context.casefold() in source_text.casefold():
            return fact_context
        spans = [
            normalize_text(str(item))
            for item in list(fact.get("evidence_spans") or [])[:2]
            if normalize_text(str(item))
        ]
        if not spans or any(span.casefold() not in source_text.casefold() for span in spans):
            return ""
        return normalize_text(" ".join(spans))

    @staticmethod
    def _fact_contract_fields(fact: dict[str, Any]) -> dict[str, Any]:
        if not fact:
            return {}
        qualifiers = {
            str(key): str(value)
            for key, value in dict(fact.get("qualifiers") or {}).items()
        }
        return {
            "fact_id": normalize_text(str(fact.get("fact_id") or "")),
            "subject": normalize_text(str(fact.get("subject") or "")),
            "relation": normalize_text(str(fact.get("relation") or "")),
            "object": normalize_text(str(fact.get("object") or "")),
            "qualifiers": qualifiers,
            "polarity": normalize_text(str(fact.get("polarity") or "positive")),
            "grounding_status": normalize_text(
                str(fact.get("grounding_status") or "")
            ),
            "contract_method": normalize_text(
                qualifiers.get("contract_method") or "semantic_fact"
            ),
            "origin_fact_id": normalize_text(
                qualifiers.get("origin_fact_id") or ""
            ),
            "scope_status": normalize_text(
                qualifiers.get("scope_status") or ""
            ),
        }

    @staticmethod
    def _is_answer_bound_fact(fact: dict[str, Any]) -> bool:
        qualifiers = dict(fact.get("qualifiers") or {})
        return bool(
            normalize_text(str(fact.get("subject") or ""))
            and normalize_text(str(fact.get("relation") or ""))
            and normalize_text(str(fact.get("object") or ""))
            and normalize_text(str(fact.get("grounding_status") or ""))
            == "grounded"
            and normalize_text(str(fact.get("role") or "")).upper()
            == "ANSWER_SUPPORT"
            and normalize_text(str(qualifiers.get("answer_binding") or ""))
            == "direct"
        )

    def _next_pending_goal(self, plan: RelationPlan):
        active = plan.active_goal
        if active is None:
            return None
        active_index = next(
            (index for index, goal in enumerate(plan.goals) if goal.goal_id == active.goal_id),
            -1,
        )
        return next(
            (goal for goal in plan.goals[active_index + 1 :] if goal.state == "pending"),
            None,
        )

    def _next_pending_goal_after(self, plan: RelationPlan, goal_id: str):
        goal_index = next(
            (index for index, goal in enumerate(plan.goals) if goal.goal_id == goal_id),
            -1,
        )
        if goal_index < 0:
            return None
        return next(
            (goal for goal in plan.goals[goal_index + 1 :] if goal.state != "resolved"),
            None,
        )

    def _grounded_context(self, text: str, span: str, *, max_chars: int = 520) -> str:
        cleaned_span = normalize_text(span)
        index = text.casefold().find(cleaned_span.casefold())
        if not cleaned_span or index < 0:
            return ""
        start = max(0, index - max_chars // 2)
        end = min(len(text), index + len(cleaned_span) + max_chars // 2)
        return normalize_text(text[start:end])

    def _dedupe(self, values: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value).strip(" \"'`.,;:")
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            output.append(cleaned)
            seen.add(key)
        return output


__all__ = [
    "BridgeEvidenceContract",
    "DirectEvidenceContract",
    "EvidenceRoleContractBuilder",
    "EvidenceRoleContracts",
    "RejectedEvidenceSpan",
]
