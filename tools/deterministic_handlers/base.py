from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .schema import HandlerIOContract, SCHEMA_VERSION


@dataclass
class HandlerInput:
    question: str
    attachment: dict[str, Any] = field(default_factory=dict)
    attachment_result: str = ""
    search_result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        parts = [
            self.question,
            self.attachment_result,
            self.search_result,
        ]
        return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())

    def adapted_inputs(self) -> dict[str, Any]:
        value = self.metadata.get("adapted_inputs") if isinstance(self.metadata, dict) else None
        return dict(value) if isinstance(value, dict) else {}


@dataclass
class HandlerMatch:
    handler_name: str
    matched: bool
    confidence: float
    reason: str
    handler_role: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandlerResult:
    handler_name: str
    status: str
    answer: str = ""
    evidence_text: str = ""
    structured_result: dict[str, Any] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    error: str = ""
    confidence: float = 0.0
    next_action_hint: str = ""
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_type: str = "intermediate_value"
    semantic_role: str = ""
    supporting_inputs: list[str] = field(default_factory=list)
    output_schema_version: str = SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(str(self.answer or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def no_match(cls, *, matches: list[HandlerMatch] | None = None) -> "HandlerResult":
        best_missing = []
        best_handler = ""
        if matches:
            best = next((match for match in matches if match.missing_inputs), matches[0])
            best_missing = list(best.missing_inputs)
            best_handler = best.handler_name
        return cls(
            handler_name=best_handler,
            status="no_match",
            missing_inputs=best_missing,
            structured_result={
                "matches": [match.to_dict() for match in matches or []],
            },
            error="no deterministic handler matched",
            next_action_hint=(
                "Recover missing deterministic inputs: " + ", ".join(best_missing)
                if best_missing
                else ""
            ),
        )

    @classmethod
    def missing(
        cls,
        *,
        handler_name: str,
        missing_inputs: list[str],
        structured_result: dict[str, Any] | None = None,
        next_action_hint: str = "",
        input_summary: dict[str, Any] | None = None,
    ) -> "HandlerResult":
        return cls(
            handler_name=handler_name,
            status="missing_inputs",
            missing_inputs=list(missing_inputs),
            structured_result=structured_result or {},
            error="missing required deterministic handler inputs",
            next_action_hint=next_action_hint,
            input_summary=input_summary or {},
        )

    @classmethod
    def error_result(cls, *, handler_name: str, error: str) -> "HandlerResult":
        return cls(handler_name=handler_name, status="error", error=error)

    @classmethod
    def missing_handler(
        cls,
        *,
        required_handler_role: str,
        matches: list[HandlerMatch] | None = None,
    ) -> "HandlerResult":
        return cls(
            handler_name="",
            status="missing_handler",
            missing_inputs=["handler:" + str(required_handler_role or "unknown")],
            structured_result={
                "required_handler_role": required_handler_role,
                "matches": [match.to_dict() for match in matches or []],
            },
            error=f"no deterministic handler registered for role: {required_handler_role}",
            next_action_hint=(
                "No deterministic handler is registered for this task role; "
                "do not use a generic handler as fallback."
            ),
        )


class DeterministicHandler(Protocol):
    name: str
    capability_description: str
    supported_attachment_types: set[str]
    handler_role: str
    supported_answer_roles: set[str]
    supported_task_roles: set[str]
    input_schema: HandlerIOContract
    output_schema: HandlerIOContract

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        ...

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        ...

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        ...


def render_handler_evidence(result: HandlerResult) -> str:
    if not result.ok:
        return ""
    if result.output_type != "final_answer":
        return (
            "Intermediate deterministic result:\n"
            f"Handler: {result.handler_name}\n"
            f"Status: {result.status}\n"
            f"Output type: {result.output_type}\n"
            f"Semantic role: {result.semantic_role or 'unspecified'}\n"
            f"Value: {result.answer}\n"
            "Instruction: this is not a final answer; use only as intermediate trace."
        )
    return (
        "Deterministic handler evidence:\n"
        f"Handler: {result.handler_name}\n"
        f"Status: {result.status}\n"
        f"Output type: {result.output_type}\n"
        f"Semantic role: {result.semantic_role or 'final_answer'}\n"
        f"Answer: {result.answer}\n"
        f"Confidence: {result.confidence}\n"
        f"Input summary: {result.input_summary}\n"
        "Instruction: prefer this exact deterministic result for closed-world computation tasks."
    )


__all__ = [
    "DeterministicHandler",
    "HandlerInput",
    "HandlerMatch",
    "HandlerResult",
    "render_handler_evidence",
]
