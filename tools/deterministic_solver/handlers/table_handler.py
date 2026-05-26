"""表格 deterministic handler。"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, lower_text


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
