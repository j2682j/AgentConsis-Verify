from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from utils.network_utils import normalize_text

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedEvidenceSpan:
    """Record why a classified span did not receive downstream authority."""

    span: str
    role: str
    document_id: str
    reason: str

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
        direct_spans: Iterable[str],
        bridge_spans: Iterable[str],
    ) -> EvidenceRoleContracts:
        source_text = normalize_text(text)
        requirement = (
            normalize_text(answer_requirement)
            or normalize_text(answer_target)
            or normalize_text(question)
        )
        plan = relation_plan or RelationPlan()
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
