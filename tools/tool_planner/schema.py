from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCandidate:
    tool_name: str
    capability: str
    priority_hint: str = ""
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolPlanStep:
    tool_name: str
    purpose: str = ""
    depends_on: list[str] = field(default_factory=list)
    expected_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandlerPlan:
    tool_name: str = "deterministic_handler"
    handler_name: str = ""
    reason: str = ""
    required_inputs: list[str] = field(default_factory=list)
    available_inputs: dict[str, Any] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    status: str = "not_applicable"
    next_action_hint: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolNeed:
    need_type: str
    required_capabilities: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolPlan:
    requires_tools: bool = False
    tool_needs: list[ToolNeed] = field(default_factory=list)
    tool_sequence: list[ToolPlanStep] = field(default_factory=list)
    handler_plans: list[HandlerPlan] = field(default_factory=list)
    stop_condition: str = ""
    planner_source: str = "system"
    validation_errors: list[str] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_needs"] = [need.to_dict() for need in self.tool_needs]
        payload["tool_sequence"] = [step.to_dict() for step in self.tool_sequence]
        payload["handler_plans"] = [handler_plan.to_dict() for handler_plan in self.handler_plans]
        return payload


@dataclass
class ToolPlanResult:
    candidate_tools: list[ToolCandidate]
    raw_planner_reply: str
    parsed_plan: ToolPlan
    validated_plan: ToolPlan
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_tools": [candidate.to_dict() for candidate in self.candidate_tools],
            "raw_planner_reply": self.raw_planner_reply,
            "parsed_plan": self.parsed_plan.to_dict(),
            "validated_plan": self.validated_plan.to_dict(),
            "fallback_used": self.fallback_used,
        }


__all__ = [
    "HandlerPlan",
    "ToolCandidate",
    "ToolNeed",
    "ToolPlan",
    "ToolPlanResult",
    "ToolPlanStep",
]
