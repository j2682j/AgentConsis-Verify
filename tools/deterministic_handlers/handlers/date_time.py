from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field


class DateTimeRouterHandler:
    name = "date_time"
    capability_description = (
        "Compute exact date and time answers, including date difference, weekday, "
        "duration, earliest date, latest date, and ordering dates."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    routing_terms = {"date", "dates", "day", "days", "weekday", "elapsed", "between", "earliest", "latest"}
    input_schema = io_contract(
        name,
        [
            input_field("dates", "list[datetime]", True, "Date values needed for the operation.", "question|attachment|search"),
            input_field("operation", "str", True, "Date operation such as difference, weekday, earliest, or latest.", "question"),
        ],
        [
            *default_outputs(),
            output_field("dates", "list[str]", False, "Normalized dates used by the handler."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    DATE_PATTERNS = (
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    )

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        dates = self._dates(handler_input.combined_text())
        operation = self._operation(handler_input.question)
        missing = []
        if not dates:
            missing.append("date_values")
        if not operation:
            missing.append("date_operation")
        if operation == "date_difference" and len(dates) < 2:
            missing.append("two_dates")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.94 if not missing else 0.35,
            reason="date_values_and_operation_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        text = handler_input.combined_text()
        return {
            "question": handler_input.question,
            "dates": self._dates(text),
            "operation": self._operation(handler_input.question),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        dates = list(inputs.get("dates") or [])
        operation = str(inputs.get("operation") or "")
        if not dates:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["date_values"],
                structured_result={"operation": operation},
            )
        if operation == "date_difference":
            if len(dates) < 2:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["two_dates"],
                    structured_result={"dates": [item.isoformat() for item in dates]},
                )
            answer = str(abs((dates[1] - dates[0]).days))
            task_type = "date_difference_days"
        elif operation == "weekday":
            answer = dates[0].strftime("%A")
            task_type = "date_weekday"
        elif operation == "earliest":
            answer = min(dates).strftime("%Y-%m-%d")
            task_type = "date_earliest"
        elif operation == "latest":
            answer = max(dates).strftime("%Y-%m-%d")
            task_type = "date_latest"
        else:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["date_operation"],
                structured_result={"dates": [item.isoformat() for item in dates]},
            )
        structured = {
            "task_type": task_type,
            "operation": operation,
            "dates": [item.isoformat() for item in dates],
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Task: {task_type}\n"
                f"Dates: {structured['dates']}\n"
                f"Answer: {answer}\n"
                "Instruction: prefer this exact deterministic result for closed-world date tasks."
            ),
            structured_result=structured,
            confidence=0.95,
            output_type="final_answer",
            semantic_role=task_type,
            supporting_inputs=structured["dates"],
        )

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        if re.search(r"\b(days?|elapsed|difference|between)\b", lowered) and "weekday" not in lowered:
            return "date_difference"
        if "weekday" in lowered or "day of the week" in lowered:
            return "weekday"
        if "earliest" in lowered or "first" in lowered:
            return "earliest"
        if "latest" in lowered or "last" in lowered or "most recent" in lowered:
            return "latest"
        return ""

    def _dates(self, text: str) -> list[datetime]:
        values: list[datetime] = []
        candidates = set()
        candidates.update(
            re.findall(r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", text or "")
        )
        candidates.update(
            re.findall(r"\b[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\b", text or "")
        )
        candidates.update(re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text or ""))
        candidates.update(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text or ""))
        for candidate in candidates:
            for pattern in self.DATE_PATTERNS:
                try:
                    values.append(datetime.strptime(candidate, pattern))
                    break
                except ValueError:
                    continue
        return sorted(values)


__all__ = ["DateTimeRouterHandler"]
