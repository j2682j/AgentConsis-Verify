from __future__ import annotations

from decimal import Decimal
import re
from statistics import median
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field
from .common import (
    attachment_path,
    normalize_text,
    parse_inline_delimited_rows,
    read_delimited_rows,
)


class TableExactRouterHandler:
    name = "table_exact_operations"
    capability_description = (
        "Perform exact table, spreadsheet, CSV, or markdown-table operations including "
        "multi-condition filtering, cell lookup, sorting, group by, count, sum, average, "
        "mean, median, unique count, and duplicate count."
    )
    supported_attachment_types: set[str] = {".csv", ".tsv", ".txt", ".xlsx", ".json"}
    routing_terms = {
        "table",
        "spreadsheet",
        "csv",
        "row",
        "column",
        "cell",
        "where",
        "filter",
        "group",
        "unique",
        "duplicate",
        "median",
        "average",
        "sum",
        "sort",
    }
    input_schema = io_contract(
        name,
        [
            input_field("rows", "list[list[str]]", True, "Raw table rows.", "attachment|question"),
            input_field("operation", "str", True, "Exact table operation.", "question"),
            input_field("target_column", "str", False, "Column used for aggregation or lookup.", "question|attachment"),
            input_field("filters", "list[dict]", False, "Exact row filters inferred from the question.", "question|attachment"),
            input_field("group_column", "str", False, "Column used for grouping.", "question|attachment"),
            input_field("sort_spec", "dict", False, "Sort column and direction.", "question|attachment"),
        ],
        [
            *default_outputs(),
            output_field("matched_row_count", "int", False, "Number of rows used after filtering."),
            output_field("headers", "list[str]", False, "Detected table headers."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        missing: list[str] = []
        if not inputs["records"]:
            missing.append("table_rows")
        if not inputs["operation"]:
            missing.append("table_operation")
        confidence = 0.95 if not missing else 0.35
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=confidence,
            reason="table_rows_and_operation_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        rows = self._attachment_rows(handler_input) or parse_inline_delimited_rows(
            handler_input.combined_text()
        )
        records = self._records(rows)
        headers = list(records[0]) if records else []
        question = handler_input.question
        filters = self._filters(question, headers)
        filtered = self._apply_filters(records, filters)
        operation = self._operation(question)
        group_column = self._group_column(question, headers)
        target_column = self._target_column(question, headers, filters, group_column)
        sort_spec = self._sort_spec(question, headers)
        ordinal = self._ordinal(question)
        return {
            "rows": rows,
            "records": records,
            "filtered": filtered,
            "filters": filters,
            "operation": operation,
            "target_column": target_column,
            "group_column": group_column,
            "sort_spec": sort_spec,
            "ordinal": ordinal,
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        records = list(inputs.get("records") or [])
        filtered = list(inputs.get("filtered") or [])
        operation = str(inputs.get("operation") or "")
        target_column = str(inputs.get("target_column") or "")
        if not records:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["table_rows"],
                structured_result={"rows": inputs.get("rows", [])[:5]},
                next_action_hint="Use attachment_reader or provide CSV/markdown table rows.",
            )
        if not operation:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["table_operation"],
                structured_result={"headers": list(records[0])},
                next_action_hint="Ask for a count, sum, average, median, unique count, lookup, sort, or group by operation.",
            )

        if operation == "row_lookup":
            return self._row_lookup(inputs)
        if operation == "sort_nth":
            return self._sort_nth(inputs)
        if operation == "group_count":
            return self._group_count(inputs)
        if operation in {"count", "how_many"}:
            return self._result("table_filtered_count", str(len(filtered)), inputs)
        if operation in {"unique_count", "duplicate_count"}:
            if not target_column:
                return self._missing_target(inputs)
            values = [record.get(target_column, "") for record in filtered]
            if operation == "unique_count":
                answer = str(len({value for value in values if value != ""}))
                return self._result("table_unique_count", answer, inputs)
            counts: dict[str, int] = {}
            for value in values:
                if value:
                    counts[value] = counts.get(value, 0) + 1
            answer = str(sum(1 for count in counts.values() if count > 1))
            return self._result("table_duplicate_count", answer, inputs)
        if operation in {"max", "min", "sum", "average", "mean", "median"}:
            if not target_column:
                return self._missing_target(inputs)
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
                    next_action_hint="Choose a numeric target column.",
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
            structured_result={"operation": operation, "headers": list(records[0])},
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
            if any(str(cell).strip() for cell in row)
        ]

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        if re.search(r"\bgroup(?:ed)?\s+by\b", lowered):
            return "group_count"
        if re.search(r"\b(sort|order|rank|top|bottom)\b", lowered) and self._ordinal(lowered) is not None:
            return "sort_nth"
        if re.search(r"\b(row|record)\b", lowered) and re.search(r"\bwhere\b", lowered):
            return "row_lookup"
        if "duplicate" in lowered:
            return "duplicate_count"
        if "unique" in lowered or "distinct" in lowered:
            return "unique_count"
        if "how many" in lowered:
            return "count"
        for operation in ("average", "mean", "median", "sum", "max", "min", "count"):
            if re.search(rf"\b{operation}\b", lowered):
                return operation
        if "maximum" in lowered or "largest" in lowered or "highest" in lowered:
            return "max"
        if "minimum" in lowered or "smallest" in lowered or "lowest" in lowered:
            return "min"
        return ""

    def _filters(self, question: str, headers: list[str]) -> list[dict[str, str]]:
        filters: list[dict[str, str]] = []
        for header in sorted(headers, key=len, reverse=True):
            aliases = self._header_aliases(header)
            alias_pattern = "|".join(re.escape(alias) for alias in aliases)
            pattern = (
                rf"\b(?:{alias_pattern})\b\s*(>=|<=|!=|=|>|<|is|equals?)\s*"
                rf"[\"']?([^,;.\"']+)"
            )
            for match in re.finditer(pattern, question or "", flags=re.IGNORECASE):
                operator = match.group(1).lower()
                value = self._clean_filter_value(match.group(2))
                filters.append(
                    {
                        "column": header,
                        "operator": "=" if operator in {"is", "equal", "equals"} else operator,
                        "value": value,
                    }
                )
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, str]] = []
        for item in filters:
            key = (item["column"], item["operator"], item["value"].casefold())
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _clean_filter_value(self, value: str) -> str:
        cleaned = normalize_text(value).strip(" ?!")
        cleaned = re.split(
            r"\s+(?:and|or)\s+[A-Za-z0-9_ -]+\s*(?:>=|<=|!=|=|>|<|is|equals?)\s*",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return normalize_text(cleaned).strip(" ?!")

    def _apply_filters(
        self,
        records: list[dict[str, str]],
        filters: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        filtered = records
        for item in filters:
            filtered = [
                record
                for record in filtered
                if self._compare(
                    record.get(item["column"], ""),
                    item["operator"],
                    item["value"],
                )
            ]
        return filtered

    def _compare(self, value: str, operator: str, target: str) -> bool:
        left_number = self._decimal(value)
        right_number = self._decimal(target)
        if left_number is not None and right_number is not None:
            left: Any = left_number
            right: Any = right_number
        else:
            left = normalize_text(value).casefold()
            right = normalize_text(target).casefold()
        return {
            "=": left == right,
            "!=": left != right,
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
        }[operator]

    def _target_column(
        self,
        question: str,
        headers: list[str],
        filters: list[dict[str, str]],
        group_column: str,
    ) -> str:
        filter_columns = {item["column"] for item in filters}
        lowered = str(question or "").lower()
        for header in sorted(headers, key=len, reverse=True):
            if header == group_column or header in filter_columns:
                continue
            if self._header_in_text(header, lowered):
                return header
        numeric_headers = [
            header
            for header in headers
            if header not in filter_columns and header != group_column
        ]
        return numeric_headers[-1] if len(numeric_headers) == 1 else ""

    def _group_column(self, question: str, headers: list[str]) -> str:
        match = re.search(r"\bgroup(?:ed)?\s+by\s+([A-Za-z0-9_ -]+)", question or "", re.IGNORECASE)
        if not match:
            return ""
        return self._best_header(match.group(1), headers)

    def _sort_spec(self, question: str, headers: list[str]) -> dict[str, Any]:
        lowered = str(question or "").lower()
        if not re.search(r"\b(sort|order|rank|top|bottom|highest|lowest|largest|smallest)\b", lowered):
            return {}
        descending = bool(
            re.search(r"\b(descending|top|highest|largest|maximum|max)\b", lowered)
        )
        ascending = bool(
            re.search(r"\b(ascending|bottom|lowest|smallest|minimum|min)\b", lowered)
        )
        direction = "asc" if ascending and not descending else "desc"
        for header in sorted(headers, key=len, reverse=True):
            if self._header_in_text(header, lowered):
                return {"column": header, "direction": direction}
        return {}

    def _ordinal(self, question: str) -> int | None:
        lowered = str(question or "").lower()
        mapping = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
        for word, index in mapping.items():
            if re.search(rf"\b{word}\b", lowered):
                return index
        match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", lowered)
        return int(match.group(1)) - 1 if match else None

    def _row_lookup(self, inputs: dict[str, Any]) -> HandlerResult:
        filtered = list(inputs.get("filtered") or [])
        target_column = str(inputs.get("target_column") or "")
        if not filtered:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["matching_row"],
                structured_result=self._structured(inputs),
            )
        if target_column:
            return self._result("table_row_lookup", filtered[0].get(target_column, ""), inputs)
        return self._result("table_row_lookup", str(filtered[0]), inputs)

    def _sort_nth(self, inputs: dict[str, Any]) -> HandlerResult:
        filtered = list(inputs.get("filtered") or [])
        sort_spec = dict(inputs.get("sort_spec") or {})
        ordinal = inputs.get("ordinal")
        target_column = str(inputs.get("target_column") or "")
        if not filtered or not sort_spec or ordinal is None:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["sortable_rows", "sort_column", "ordinal"],
                structured_result=self._structured(inputs),
            )
        column = str(sort_spec.get("column") or "")
        reverse = sort_spec.get("direction") == "desc"
        sorted_rows = sorted(
            filtered,
            key=lambda row: self._sort_key(row.get(column, "")),
            reverse=reverse,
        )
        index = int(ordinal)
        if not 0 <= index < len(sorted_rows):
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["ordinal_in_range"],
                structured_result={"row_count": len(sorted_rows), "ordinal": ordinal},
            )
        row = sorted_rows[index]
        answer = row.get(target_column or column, "")
        return self._result("table_sort_nth", answer, {**inputs, "sorted_rows": sorted_rows[:10]})

    def _group_count(self, inputs: dict[str, Any]) -> HandlerResult:
        filtered = list(inputs.get("filtered") or [])
        group_column = str(inputs.get("group_column") or "")
        if not group_column:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["group_column"],
                structured_result=self._structured(inputs),
            )
        counts: dict[str, int] = {}
        for record in filtered:
            key = record.get(group_column, "")
            counts[key] = counts.get(key, 0) + 1
        answer = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
        return self._result("table_group_count", answer, {**inputs, "groups": counts})

    def _missing_target(self, inputs: dict[str, Any]) -> HandlerResult:
        records = list(inputs.get("records") or [])
        return HandlerResult.missing(
            handler_name=self.name,
            missing_inputs=["target_column"],
            structured_result={"headers": list(records[0]) if records else []},
            next_action_hint="Specify the column to aggregate or count.",
        )

    def _result(self, task_type: str, answer: str, inputs: dict[str, Any]) -> HandlerResult:
        structured = self._structured(inputs)
        structured["task_type"] = task_type
        evidence_text = (
            "Deterministic handler evidence:\n"
            f"Handler: {self.name}\n"
            f"Task: {task_type}\n"
            f"Filters: {structured.get('filters', [])}\n"
            f"Matched rows: {structured.get('matched_row_count', 0)}\n"
            f"Answer: {answer}\n"
            "Instruction: prefer this exact deterministic result for closed-world computation tasks."
        )
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=str(answer),
            evidence_text=evidence_text,
            structured_result=structured,
            confidence=0.96,
            output_type="final_answer",
            semantic_role=task_type,
            supporting_inputs=self._supporting_inputs(structured),
        )

    def _structured(self, inputs: dict[str, Any]) -> dict[str, Any]:
        filtered = list(inputs.get("filtered") or [])
        return {
            "operation": inputs.get("operation", ""),
            "target_column": inputs.get("target_column", ""),
            "group_column": inputs.get("group_column", ""),
            "filters": list(inputs.get("filters") or []),
            "sort_spec": dict(inputs.get("sort_spec") or {}),
            "ordinal": inputs.get("ordinal"),
            "matched_row_count": len(filtered),
            "rows": filtered[:10],
            "groups": dict(inputs.get("groups") or {}),
            "sorted_rows": list(inputs.get("sorted_rows") or []),
        }

    def _supporting_inputs(self, structured: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for key in ("operation", "target_column", "group_column", "sort_spec", "filters"):
            value = structured.get(key)
            if value:
                result.append(f"{key}={value}")
        for row in list(structured.get("rows") or [])[:3]:
            result.append(f"row={row}")
        return result[:12]

    def _best_header(self, text: str, headers: list[str]) -> str:
        normalized = normalize_text(text).casefold()
        normalized_tokens = set(re.findall(r"[a-z0-9]+", normalized))
        best = ""
        best_score = 0.0
        for header in headers:
            header_norm = header.casefold()
            header_tokens = set(re.findall(r"[a-z0-9]+", header_norm))
            score = 1.0 if header_norm in normalized else 0.0
            if header_tokens:
                score = max(score, len(header_tokens & normalized_tokens) / len(header_tokens))
            if score > best_score:
                best = header
                best_score = score
        return best if best_score >= 0.5 else ""

    def _header_in_text(self, header: str, lowered_text: str) -> bool:
        return any(
            re.search(rf"\b{re.escape(alias)}\b", lowered_text, re.IGNORECASE)
            for alias in self._header_aliases(header)
        )

    def _header_aliases(self, header: str) -> list[str]:
        aliases = {header, header.lower(), header.replace("_", " "), header.replace("-", " ")}
        aliases.add(re.sub(r"[_-]+", " ", header.lower()))
        return [alias for alias in aliases if alias.strip()]

    def _sort_key(self, value: Any) -> Any:
        number = self._decimal(value)
        return number if number is not None else normalize_text(value).casefold()

    def _decimal(self, value: Any) -> Decimal | None:
        text = normalize_text(value).replace(",", "").replace("%", "")
        try:
            return Decimal(text)
        except Exception:
            return None

    def _format_decimal(self, value: Decimal) -> str:
        if value == value.to_integral():
            return str(value.quantize(Decimal("1")))
        return format(value.normalize(), "f")


__all__ = ["TableExactRouterHandler"]
