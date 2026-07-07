from __future__ import annotations

from typing import Any, Iterable

from .schema import HandlerIOContract, HandlerInputField, HandlerOutputField


def input_field(
    name: str,
    type_: str,
    required: bool = True,
    description: str = "",
    source: str = "question",
) -> HandlerInputField:
    return HandlerInputField(
        name=name,
        type=type_,
        required=required,
        description=description,
        source=source,
    )


def output_field(
    name: str,
    type_: str,
    required: bool = True,
    description: str = "",
) -> HandlerOutputField:
    return HandlerOutputField(
        name=name,
        type=type_,
        required=required,
        description=description,
    )


def io_contract(
    handler_name: str,
    inputs: Iterable[HandlerInputField],
    outputs: Iterable[HandlerOutputField],
    *,
    supported_attachment_types: Iterable[str] = (),
    examples: Iterable[dict[str, Any]] = (),
) -> HandlerIOContract:
    return HandlerIOContract(
        handler_name=handler_name,
        input_fields=list(inputs),
        output_fields=list(outputs),
        supported_attachment_types=set(supported_attachment_types),
        examples=list(examples),
    )


def default_outputs() -> list[HandlerOutputField]:
    return [
        output_field("answer", "str", True, "Deterministic final answer."),
        output_field("task_type", "str", True, "Normalized deterministic task type."),
        output_field("calculation_trace", "dict", False, "Compact computation trace."),
    ]


__all__ = ["default_outputs", "input_field", "io_contract", "output_field"]
