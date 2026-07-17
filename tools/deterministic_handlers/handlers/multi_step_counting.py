from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class MultiStepCountingRouterHandler:
    name = "multi_step_counting"
    handler_role = "multi_step_counting"
    capability_description = (
        "Solve small closed-world counting tasks that combine group counts, per-group quantities, "
        "and totals from text or attachment context."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    supported_task_roles: set[str] = {"multi_step_counting"}
    supported_answer_roles: set[str] = {"count", "number"}
    input_schema = io_contract(
        name,
        [
            input_field("counting_factors", "list[dict]", True, "Group counts and per-group quantities.", "question|attachment"),
            input_field("operation", "str", True, "Counting operation such as sum of products.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        missing = []
        if not inputs["terms"]:
            missing.append("counting_factors")
        if not inputs["operation"]:
            missing.append("counting_operation")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.9 if not missing else 0.25,
            reason="counting_terms_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        text = handler_input.combined_text()
        terms = self._sum_product_terms(text)
        if not terms:
            terms = self._each_terms(text)
        operation = "sum_products" if terms else ""
        return {"question": handler_input.question, "terms": terms, "operation": operation}

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        terms = list(inputs.get("terms") or [])
        operation = str(inputs.get("operation") or "")
        if not terms or not operation:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=[
                    item for item, value in (("counting_factors", terms), ("counting_operation", operation)) if not value
                ],
                next_action_hint="Provide explicit group counts and per-group quantities.",
            )
        total = sum(int(term["count"]) * int(term["per_item"]) for term in terms)
        structured = {
            "task_type": "multi_step_counting",
            "operation": operation,
            "terms": terms,
            "total": total,
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=str(total),
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Counting terms: {terms}\n"
                f"Answer: {total}\n"
                "Instruction: use this result only if the task asks for this total count."
            ),
            structured_result=structured,
            confidence=0.91,
            output_type="final_answer",
            semantic_role="multi_step_counting_answer",
            supporting_inputs=[str(term) for term in terms],
        )

    def _sum_product_terms(self, text: str) -> list[dict[str, Any]]:
        lowered = str(text or "").lower()
        # Handles compact forms such as "3 adults with 2 bags each and 5 kids with 1 bag each".
        pattern = re.compile(
            r"(\d+)\s+([a-z][a-z -]{0,30}?)\s+(?:with|having|carry|carrying|gets?|each(?:\s+has)?)\s+(\d+)\s+([a-z][a-z -]{0,30}?)(?:\s+each|\b)",
            re.IGNORECASE,
        )
        terms = []
        for match in pattern.finditer(lowered):
            terms.append(
                {
                    "count": int(match.group(1)),
                    "group": self._clean_label(match.group(2)),
                    "per_item": int(match.group(3)),
                    "item": self._clean_label(match.group(4)),
                }
            )
        return terms

    def _each_terms(self, text: str) -> list[dict[str, Any]]:
        lowered = str(text or "").lower()
        # Handles "each of the 4 teams has 6 players".
        terms = []
        for match in re.finditer(
            r"each\s+of\s+the\s+(\d+)\s+([a-z][a-z -]{0,30}?)\s+(?:has|have|gets?|contains?)\s+(\d+)\s+([a-z][a-z -]{0,30}?)(?:[.,;?]|$)",
            lowered,
            re.IGNORECASE,
        ):
            terms.append(
                {
                    "count": int(match.group(1)),
                    "group": self._clean_label(match.group(2)),
                    "per_item": int(match.group(3)),
                    "item": self._clean_label(match.group(4)),
                }
            )
        return terms

    def _clean_label(self, value: str) -> str:
        text = " ".join(str(value or "").split())
        text = re.sub(r"\b(each|and|with|having|has|have|gets|contains?)\b", "", text).strip()
        return " ".join(text.split())


__all__ = ["MultiStepCountingRouterHandler"]
