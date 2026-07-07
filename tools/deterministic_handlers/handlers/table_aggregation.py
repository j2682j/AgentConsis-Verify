from __future__ import annotations

from decimal import Decimal
import re
from statistics import median
from typing import Any

from ..base import HandlerInput, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field
from .common import attachment_path, normalize_text, parse_inline_delimited_rows, read_delimited_rows


class TableAggregationRouterHandler:
    name = "table_aggregation"
    capability_description = (
        "Perform exact CSV or table aggregation, filtering, count, max, min, sum, average, "
        "mean, median, unique count, duplicate count, and simple row/column lookup."
    )
    supported_attachment_types: set[str] = {".csv", ".tsv", ".txt"}
    routing_terms = {"table", "spreadsheet", "csv", "filter", "count", "average", "mean", "median", "sum", "max", "min"}
    input_schema = io_contract(
        name,
        [
            input_field("rows", "list[list[str]]", True, "Raw table rows.", "attachment|question"),
            input_field("operation", "str", True, "Aggregation operation.", "question"),
            input_field("target_column", "str", False, "Numeric or categorical target column.", "question|attachment"),
            input_field("condition", "dict", False, "Optional filter condition.", "question|attachment"),
        ],
        [
            *default_outputs(),
            output_field("matched_row_count", "int", False, "Number of filtered rows."),
            output_field("headers", "list[str]", False, "Detected table headers."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        rows = self._attachment_rows(handler_input) or parse_inline_delimited_rows(handler_input.combined_text())
        operation = self._operation(handler_input.question)
        records = self._records(rows)
        target_column = self._target_column(handler_input.question, list(records[0]) if records else [])
        filtered, condition = self._filter_records(handler_input.question, records)
        return {
            "rows": rows,
            "records": records,
            "filtered": filtered,
            "operation": operation,
            "target_column": target_column,
            "condition": condition,
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        records = list(inputs.get("records") or [])
        filtered = list(inputs.get("filtered") or records)
        operation = str(inputs.get("operation", "") or "")
        target_column = str(inputs.get("target_column", "") or "")
        if not records:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["table_rows"],
                structured_result={"rows": inputs.get("rows", [])[:5]},
            )
        if operation in {"count", "how_many"}:
            return self._result("table_filtered_count", str(len(filtered)), inputs)
        if operation in {"max", "min", "sum", "average", "mean", "median"}:
            if not target_column:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["target_column"],
                    structured_result={"headers": list(records[0]) if records else []},
                )
            values = [
                value
                for record in filtered
                if (value := self._decimal(record.get(target_column, ""))) is not None
            ]
            if not values:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["numeric_values"],
                    structured_result={"target_column": target_column},
                )
            if operation == "max":
                answer = max(values)
            elif operation == "min":
                answer = min(values)
            elif operation == "sum":
                answer = sum(values)
            elif operation == "median":
                answer = Decimal(str(median(values)))
            else:
                answer = sum(values) / Decimal(len(values))
            return self._result(f"table_{operation}", self._format_decimal(answer), inputs)
        return HandlerResult.missing(
            handler_name=self.name,
            missing_inputs=["supported_table_operation"],
            structured_result={
                "operation": operation,
                "headers": list(records[0]) if records else [],
            },
        )

    def _attachment_rows(self, handler_input: HandlerInput) -> list[list[str]]:
        path = attachment_path(handler_input.attachment)
        if path and path.suffix.lower() in {".csv", ".tsv"}:
            return read_delimited_rows(path)
        return []

    def _records(self, rows: list[list[str]]) -> list[dict[str, str]]:
        if len(rows) < 2:
            return []
        headers = [normalize_text(header) for header in rows[0]]
        return [
            {
                header: normalize_text(row[index]) if index < len(row) else ""
                for index, header in enumerate(headers)
            }
            for row in rows[1:]
        ]

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        if "how many" in lowered:
            return "count"
        for operation in ("average", "mean", "median", "sum", "max", "min", "count"):
            if re.search(rf"\b{operation}\b", lowered):
                return operation
        if "maximum" in lowered or "largest" in lowered:
            return "max"
        if "minimum" in lowered or "smallest" in lowered:
            return "min"
        return ""

    def _target_column(self, question: str, headers: list[str]) -> str:
        lowered = str(question or "").lower()
        for header in sorted(headers, key=len, reverse=True):
            if header and re.search(rf"\b{re.escape(header.lower())}\b", lowered):
                condition_only = re.search(
                    rf"\b{re.escape(header.lower())}\b\s*(?:>=|<=|!=|=|>|<)",
                    lowered,
                )
                if not condition_only:
                    return header
        return ""

    def _filter_records(
        self,
        question: str,
        records: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], dict[str, str]]:
        if not records:
            return [], {}
        for header in sorted(records[0], key=len, reverse=True):
            match = re.search(
                rf"\b{re.escape(header)}\b\s*(>=|<=|!=|=|>|<)\s*[\"']?([^,;.\"']+)",
                question or "",
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            operator = match.group(1)
            target = normalize_text(match.group(2))
            filtered = [
                record
                for record in records
                if self._compare(record.get(header, ""), operator, target)
            ]
            return filtered, {"column": header, "operator": operator, "value": target}
        return records, {}

    def _compare(self, value: str, operator: str, target: str) -> bool:
        left_number = self._decimal(value)
        right_number = self._decimal(target)
        left: Any = left_number if left_number is not None and right_number is not None else value.lower()
        right: Any = right_number if left_number is not None and right_number is not None else target.lower()
        return {
            "=": left == right,
            "!=": left != right,
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
        }[operator]

    def _decimal(self, value: Any) -> Decimal | None:
        try:
            return Decimal(normalize_text(value).replace(",", ""))
        except Exception:
            return None

    def _format_decimal(self, value: Decimal) -> str:
        if value == value.to_integral():
            return str(value.quantize(Decimal("1")))
        return format(value.normalize(), "f")

    def _result(self, task_type: str, answer: str, inputs: dict[str, Any]) -> HandlerResult:
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            structured_result={
                "task_type": task_type,
                "condition": dict(inputs.get("condition") or {}),
                "target_column": inputs.get("target_column", ""),
                "matched_row_count": len(inputs.get("filtered") or []),
                "rows": list(inputs.get("filtered") or [])[:10],
            },
        )


__all__ = ["TableAggregationRouterHandler"]
