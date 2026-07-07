from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolRequestPayload:
    """
    Normalized tool request emitted by a Stage1 agent.
    """

    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    reasoning_step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1StructuredOutput:
    """
    Normalized Stage1 output used after parsing and repair.
    """

    reasoning_steps: list[str] = field(default_factory=list)
    final_answer: str = ""
    confidence: float | None = None
    used_evidence_ids: list[str] = field(default_factory=list)
    answer_type: str = "unknown"
    tool_request: ToolRequestPayload | None = None
    weights: list[int] = field(default_factory=list)

    def reasoning_text(self) -> str:
        return "\n".join(step.strip() for step in self.reasoning_steps if step.strip())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_request"] = (
            self.tool_request.to_dict()
            if isinstance(self.tool_request, ToolRequestPayload)
            else None
        )
        return payload


@dataclass
class Stage1ValidationResult:
    """
    Validation labels for Stage1 structured output.
    """

    schema_valid: bool
    eligible_for_winner: bool
    schema_errors: list[str] = field(default_factory=list)
    validity_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "Stage1StructuredOutput",
    "Stage1ValidationResult",
    "ToolRequestPayload",
]
