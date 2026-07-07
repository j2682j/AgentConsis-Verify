from __future__ import annotations

from decimal import Decimal
import re
from statistics import median
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class NumericReasoningRouterHandler:
    name = "numeric_reasoning"
    capability_description = (
        "Compute exact numeric reasoning tasks such as percentage change, ratio, "
        "difference, range, median, average, rank, and rounding from inline numbers."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    routing_terms = {"percentage", "percent", "ratio", "difference", "range", "median", "average", "round", "rank"}
    input_schema = io_contract(
        name,
        [
            input_field("numbers", "list[Decimal]", True, "Numbers used by the computation.", "question|attachment|search"),
            input_field("operation", "str", True, "Numeric operation such as ratio, difference, median, or average.", "question"),
            input_field("round_places", "int", False, "Decimal places for rounding.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        numbers = self._numbers(handler_input.combined_text())
        operation = self._operation(handler_input.question)
        missing = []
        if not numbers:
            missing.append("numbers")
        if not operation:
            missing.append("numeric_operation")
        if operation in {"percentage_change", "ratio", "difference", "range"} and len(numbers) < 2:
            missing.append("two_numbers")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.92 if not missing else 0.3,
            reason="numbers_and_numeric_operation_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        return {
            "question": handler_input.question,
            "numbers": self._numbers(handler_input.combined_text()),
            "operation": self._operation(handler_input.question),
            "round_places": self._round_places(handler_input.question),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        numbers = list(inputs.get("numbers") or [])
        operation = str(inputs.get("operation") or "")
        if not numbers:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["numbers"],
                structured_result={"operation": operation},
            )
        if operation == "percentage_change":
            value = (numbers[1] - numbers[0]) / numbers[0] * Decimal("100")
            answer = self._format(value)
        elif operation == "ratio":
            answer = f"{self._format(numbers[0])}:{self._format(numbers[1])}"
        elif operation == "difference":
            answer = self._format(abs(numbers[1] - numbers[0]))
        elif operation == "range":
            answer = self._format(max(numbers) - min(numbers))
        elif operation == "median":
            answer = self._format(Decimal(str(median(numbers))))
        elif operation in {"average", "mean"}:
            answer = self._format(sum(numbers) / Decimal(len(numbers)))
        elif operation == "sum":
            answer = self._format(sum(numbers))
        elif operation == "round":
            places = int(inputs.get("round_places") or 0)
            answer = str(round(float(numbers[0]), places))
            if places == 0:
                answer = answer.split(".", 1)[0]
        else:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["numeric_operation"],
                structured_result={"numbers": [str(item) for item in numbers]},
            )
        structured = {
            "task_type": f"numeric_{operation}",
            "operation": operation,
            "numbers": [str(item) for item in numbers],
            "round_places": inputs.get("round_places"),
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Task: numeric_{operation}\n"
                f"Numbers: {structured['numbers']}\n"
                f"Answer: {answer}\n"
                "Instruction: prefer this exact deterministic result for closed-world numeric tasks."
            ),
            structured_result=structured,
            confidence=0.94,
        )

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        if "percentage change" in lowered or "percent change" in lowered:
            return "percentage_change"
        if "ratio" in lowered:
            return "ratio"
        if "difference" in lowered:
            return "difference"
        if "range" in lowered:
            return "range"
        for operation in ("median", "average", "mean", "sum"):
            if re.search(rf"\b{operation}\b", lowered):
                return operation
        if "round" in lowered:
            return "round"
        return ""

    def _round_places(self, question: str) -> int:
        lowered = str(question or "").lower()
        match = re.search(r"(\d+)\s+decimal", lowered)
        return int(match.group(1)) if match else 0

    def _numbers(self, text: str) -> list[Decimal]:
        result: list[Decimal] = []
        for item in re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", text or ""):
            try:
                result.append(Decimal(item.rstrip("%").replace(",", "")))
            except Exception:
                continue
        return result

    def _format(self, value: Decimal) -> str:
        if value == value.to_integral():
            return str(value.quantize(Decimal("1")))
        return format(value.normalize(), "f")


__all__ = ["NumericReasoningRouterHandler"]
