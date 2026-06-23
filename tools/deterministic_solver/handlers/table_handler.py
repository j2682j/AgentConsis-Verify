"""表格 deterministic handler。"""

from __future__ import annotations

import csv
import io
import re
from decimal import Decimal
from statistics import median
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, format_decimal, lower_text


class TableHandler:
    """處理簡單表格儲存格抽取。"""

    def solve(self, question: str, *, attachment_context: str | None = None, table_data: Any = None, **_: Any) -> DeterministicSolverResult:
        """
        ??? deterministic ?????????
        
        Args:
            - ????????????
        
        Returns:
            - DeterministicSolverResult ????????
        """
        lowered = lower_text(question)
        if not any(term in lowered for term in ["table", "spreadsheet", "cell", "row", "column", "csv"]):
            return DeterministicSolverResult.miss("table")
        rows = self._rows_from_table_data(table_data) or self._parse_rows(attachment_context or question)
        if not rows:
            return DeterministicSolverResult.miss("table", "no parseable table")
        aggregate = self._aggregate_or_filter(question, rows)
        if aggregate is not None:
            return aggregate
        cell = self._extract_cell_by_row_column(question, rows)
        if cell is None:
            return DeterministicSolverResult.miss("table", "no supported cell reference")
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type="table_cell_extraction",
            answer=cell,
            answer_text=cell,
            confidence=0.86,
            evidence={"rows": rows[:5]},
        )

    def _rows_from_table_data(self, table_data: Any) -> list[list[str]]:
        """由結構化 table_data 取得列資料。"""
        if not table_data:
            return []
        if isinstance(table_data, list) and all(isinstance(row, list) for row in table_data):
            return [[clean_text(cell) for cell in row] for row in table_data]
        if isinstance(table_data, list) and all(isinstance(row, dict) for row in table_data):
            headers = list(table_data[0].keys())
            return [headers] + [[clean_text(row.get(header, "")) for header in headers] for row in table_data]
        return []

    def _parse_rows(self, text: str) -> list[list[str]]:
        """從 markdown 或 CSV 文字解析表格列。"""
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        markdown_rows = [line for line in lines if "|" in line]
        if markdown_rows:
            rows = []
            for line in markdown_rows:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if cells and not all(re.fullmatch(r"-+", cell.replace(" ", "")) for cell in cells):
                    rows.append(cells)
            return rows
        if "," in text:
            try:
                return [[clean_text(cell) for cell in row] for row in csv.reader(io.StringIO(text)) if row]
            except Exception:
                return []
        return []

    def _extract_cell_by_row_column(self, question: str, rows: list[list[str]]) -> str | None:
        """依 row/column 或 header 描述抽取儲存格。"""
        natural_question = next((line for line in str(question or "").splitlines() if "|" not in line and line.strip()), question)
        lowered = lower_text(natural_question)
        row_match = re.search(r"row\s+(\d+)", lowered)
        col_match = re.search(r"column\s+(\d+)", lowered)
        if row_match and col_match:
            row_idx = int(row_match.group(1)) - 1
            col_idx = int(col_match.group(1)) - 1
            if 0 <= row_idx < len(rows) and 0 <= col_idx < len(rows[row_idx]):
                return rows[row_idx][col_idx]
        headers = rows[0] if rows else []
        for col_idx, header in enumerate(headers):
            if header and header.lower() in lowered:
                for row in rows[1:]:
                    if row and row[0].lower() in lowered and col_idx < len(row):
                        return row[col_idx]
        return None

    def _aggregate_or_filter(
        self,
        question: str,
        rows: list[list[str]],
    ) -> DeterministicSolverResult | None:
        lowered = lower_text(question)
        if not any(
            term in lowered
            for term in (
                "filter",
                "where",
                "count",
                "how many",
                "sum",
                "average",
                "mean",
                "median",
                "unique",
                "duplicate",
            )
        ):
            return None
        records = self._records(rows)
        if not records:
            return None
        filtered, condition = self._filter_records(question, records)
        headers = list(records[0])

        if "duplicate" in lowered:
            header = self._target_header(question, headers)
            if not header:
                return None
            counts: dict[str, int] = {}
            for record in filtered:
                value = record.get(header, "")
                counts[value] = counts.get(value, 0) + 1
            answer = str(sum(1 for count in counts.values() if count > 1))
            return self._table_result("table_duplicate_count", answer, filtered, condition, header)

        if "unique" in lowered:
            header = self._target_header(question, headers)
            if not header:
                return None
            answer = str(len({record.get(header, "") for record in filtered}))
            return self._table_result("table_unique_count", answer, filtered, condition, header)

        operation = next(
            (
                name
                for name in ("average", "mean", "median", "sum")
                if name in lowered
            ),
            "",
        )
        if not operation and ("count" in lowered or "how many" in lowered):
            return self._table_result(
                "table_filtered_count",
                str(len(filtered)),
                filtered,
                condition,
                "",
            )
        if not operation:
            return None

        header = self._target_header(question, headers)
        if not header:
            return None
        values = [
            value
            for record in filtered
            if (value := self._decimal(record.get(header, ""))) is not None
        ]
        if not values:
            return None
        if operation == "sum":
            result = sum(values)
            task_type = "table_sum"
        elif operation == "median":
            result = Decimal(str(median(values)))
            task_type = "table_median"
        else:
            result = sum(values) / Decimal(len(values))
            task_type = "table_average"
        return self._table_result(
            task_type,
            format_decimal(result),
            filtered,
            condition,
            header,
        )

    def _records(self, rows: list[list[str]]) -> list[dict[str, str]]:
        if len(rows) < 2:
            return []
        headers = [clean_text(header) for header in rows[0]]
        return [
            {
                header: clean_text(row[index]) if index < len(row) else ""
                for index, header in enumerate(headers)
            }
            for row in rows[1:]
        ]

    def _filter_records(
        self,
        question: str,
        records: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], dict[str, str]]:
        headers = list(records[0])
        for header in sorted(headers, key=len, reverse=True):
            match = re.search(
                rf"\b{re.escape(header)}\b\s*(>=|<=|!=|=|>|<)\s*[\"']?([^,;.\"']+)",
                question,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            operator = match.group(1)
            target = clean_text(match.group(2))
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

    def _target_header(self, question: str, headers: list[str]) -> str:
        lowered = lower_text(question)
        for header in sorted(headers, key=len, reverse=True):
            if re.search(rf"\b{re.escape(header.lower())}\b", lowered):
                condition_only = re.search(
                    rf"\b{re.escape(header.lower())}\b\s*(?:>=|<=|!=|=|>|<)",
                    lowered,
                )
                if not condition_only:
                    return header
        return ""

    def _decimal(self, value: Any) -> Decimal | None:
        try:
            return Decimal(clean_text(value).replace(",", ""))
        except Exception:
            return None

    def _table_result(
        self,
        task_type: str,
        answer: str,
        records: list[dict[str, str]],
        condition: dict[str, str],
        target_column: str,
    ) -> DeterministicSolverResult:
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type=task_type,
            answer=answer,
            answer_text=answer,
            confidence=0.94,
            evidence={
                "matched_row_count": len(records),
                "condition": condition,
                "target_column": target_column,
                "rows": records[:10],
            },
        )
