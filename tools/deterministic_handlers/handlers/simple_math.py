from __future__ import annotations

from typing import Any

from tools.deterministic_solver.handlers.math_handler import MathHandler

from ..base import HandlerInput, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class SimpleMathRouterHandler:
    name = "simple_math"
    capability_description = (
        "Exact arithmetic, percentages, sums, averages, medians, maximums, minimums, "
        "and simple numeric expressions when all numbers are present in the question or evidence."
    )
    supported_attachment_types: set[str] = set()
    routing_terms = {"calculate", "compute", "sum", "average", "mean", "median", "percentage", "percent", "arithmetic"}
    input_schema = io_contract(
        name,
        [
            input_field("question", "str", True, "Complete numeric expression or arithmetic question.", "question|evidence"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        self.handler = MathHandler()

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        return {"question": handler_input.combined_text() or handler_input.question}

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        result = self.handler.solve(str(inputs.get("question", "")))
        if not result.used_deterministic_solver:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["complete_numeric_expression"],
                structured_result=result.to_dict(),
            )
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=str(result.answer_text or result.answer),
            evidence_text="",
            structured_result=result.to_dict(),
            confidence=float(result.confidence or 0.0),
            output_type="final_answer",
            semantic_role=str(result.task_type or "simple_math_answer"),
            supporting_inputs=[str(inputs.get("question", ""))[:240]],
        )


__all__ = ["SimpleMathRouterHandler"]
