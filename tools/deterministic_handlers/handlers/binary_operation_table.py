from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field


class BinaryOperationTableRouterHandler:
    """Evaluate algebraic properties from a finite binary-operation table."""

    name = "binary_operation_table"
    handler_role = "binary_operation_reasoning"
    capability_description = (
        "Evaluate commutativity and report exact counterexamples from a square finite "
        "binary-operation or Cayley table."
    )
    supported_attachment_types: set[str] = {".txt", ".md", ".csv", ".tsv", ".json"}
    supported_task_roles: set[str] = {
        "binary_operation_reasoning",
        "table_reasoning",
    }
    supported_answer_roles: set[str] = {
        "set_elements",
        "list",
        "boolean",
    }
    routing_terms = {
        "commutative",
        "commutativity",
        "binary",
        "operation",
        "cayley",
        "counterexample",
    }
    input_schema = io_contract(
        name,
        [
            input_field("symbols", "list[str]", True, "Elements labeling both table axes.", "question|attachment"),
            input_field("operation_table", "dict[str,dict[str,str]]", True, "Square operation table.", "question|attachment"),
            input_field("property", "str", True, "Algebraic property to evaluate.", "question"),
            input_field("answer_format", "str", False, "Requested final-answer formatting.", "question"),
        ],
        [
            *default_outputs(),
            output_field("counterexamples", "list[dict]", False, "Pairs that violate the property."),
            output_field("involved_elements", "list[str]", False, "Elements appearing in counterexamples."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        missing = []
        if not inputs["symbols"] or not inputs["operation_table"]:
            missing.append("operation_table")
        if not inputs["property"]:
            missing.append("algebraic_property")
        return HandlerMatch(
            handler_name=self.name,
            handler_role=self.handler_role,
            matched=not missing,
            confidence=0.99 if not missing else 0.25,
            reason="square_binary_operation_table_and_property",
            missing_inputs=missing,
            required_inputs=["symbols", "operation_table", "property"],
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        adapted = handler_input.adapted_inputs()
        symbols = [str(value) for value in list(adapted.get("symbols") or [])]
        table = dict(adapted.get("operation_table") or {})
        if not symbols or not table:
            symbols, table = self._parse_markdown_table(handler_input.question)
        if not symbols or not table:
            symbols, table = self._parse_markdown_table(handler_input.attachment_result)
        return {
            "symbols": symbols,
            "operation_table": table,
            "property": str(adapted.get("property") or self._property(handler_input.question)),
            "answer_format": self._answer_format(handler_input.question),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        symbols = [str(value) for value in list(inputs.get("symbols") or [])]
        table = dict(inputs.get("operation_table") or {})
        property_name = str(inputs.get("property") or "").strip().lower()
        validation = self._validate_table(symbols, table)
        if not validation["valid"]:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=list(validation["missing"]),
                structured_result={"validation": validation},
                next_action_hint="Provide a square operation table with identical row and column labels.",
            )
        if property_name != "commutative":
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["supported_algebraic_property"],
                structured_result={"property": property_name},
                next_action_hint="The current handler supports commutativity checks.",
            )

        counterexamples: list[dict[str, Any]] = []
        involved: set[str] = set()
        for left_index, left in enumerate(symbols):
            for right in symbols[left_index + 1 :]:
                forward = str(table[left][right])
                reverse = str(table[right][left])
                if forward == reverse:
                    continue
                involved.update((left, right))
                counterexamples.append(
                    {
                        "pair": [left, right],
                        "forward": forward,
                        "reverse": reverse,
                        "forward_expression": f"{left}*{right}={forward}",
                        "reverse_expression": f"{right}*{left}={reverse}",
                    }
                )

        ordered_elements = sorted(involved, key=lambda value: (value.casefold(), value))
        answer = ", ".join(ordered_elements)
        structured = {
            "task_type": "binary_operation_commutativity",
            "property": property_name,
            "symbols": symbols,
            "counterexamples": counterexamples,
            "involved_elements": ordered_elements,
            "is_commutative": not counterexamples,
            "validation": validation,
            "calculation_trace": {
                "unordered_pair_count": len(symbols) * (len(symbols) - 1) // 2,
                "counterexample_count": len(counterexamples),
            },
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            structured_result=structured,
            confidence=1.0,
            output_type="final_answer",
            semantic_role="binary_operation_counterexample_elements",
            supporting_inputs=[self._render_table(symbols, table)],
        )

    def _parse_markdown_table(self, text: str) -> tuple[list[str], dict[str, dict[str, str]]]:
        rows: list[list[str]] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if line.count("|") < 2:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells or all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                continue
            rows.append(cells)
        for start, header in enumerate(rows):
            if len(header) < 3:
                continue
            symbols = [cell for cell in header[1:] if cell]
            if len(symbols) < 2:
                continue
            candidate_rows = rows[start + 1 : start + 1 + len(symbols)]
            if len(candidate_rows) != len(symbols):
                continue
            if any(len(row) < len(symbols) + 1 for row in candidate_rows):
                continue
            row_labels = [row[0] for row in candidate_rows]
            if row_labels != symbols:
                continue
            table = {
                row[0]: {
                    symbol: row[index + 1]
                    for index, symbol in enumerate(symbols)
                }
                for row in candidate_rows
            }
            return symbols, table
        return [], {}

    def _validate_table(
        self,
        symbols: list[str],
        table: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        missing: list[str] = []
        if len(symbols) < 2 or len(set(symbols)) != len(symbols):
            missing.append("unique_symbols")
        if set(table) != set(symbols):
            missing.append("matching_row_labels")
        for symbol in symbols:
            row = table.get(symbol)
            if not isinstance(row, dict) or set(row) != set(symbols):
                missing.append(f"complete_row:{symbol}")
        return {
            "valid": not missing,
            "missing": list(dict.fromkeys(missing)),
            "symbol_count": len(symbols),
            "square": not missing,
        }

    def _property(self, question: str) -> str:
        lowered = str(question or "").casefold()
        if "commutative" in lowered or "commutativity" in lowered:
            return "commutative"
        return ""

    def _answer_format(self, question: str) -> str:
        lowered = str(question or "").casefold()
        if "comma separated" in lowered or "comma-separated" in lowered:
            return "comma_separated"
        return "plain_text"

    def _render_table(
        self,
        symbols: list[str],
        table: dict[str, dict[str, str]],
    ) -> str:
        lines = ["|*|" + "|".join(symbols) + "|"]
        lines.extend(
            "|" + symbol + "|" + "|".join(table[symbol][column] for column in symbols) + "|"
            for symbol in symbols
        )
        return "\n".join(lines)


__all__ = ["BinaryOperationTableRouterHandler"]
