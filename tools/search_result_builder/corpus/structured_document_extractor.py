from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StructuredDocumentUnit:
    """
    保存文件中可獨立建立 passage 的結構化單位。

    Args:
     - text: 單位的完整文字內容。
     - unit_type: paragraph 或 table_row。
     - section: 單位所屬章節名稱。
     - table_id: 表格識別名稱。
     - row_index: 表格資料列索引。

    Returns:
     - StructuredDocumentUnit: 可交由 cleaner 與 chunker 處理的文件單位。
    """

    text: str
    unit_type: str = "paragraph"
    section: str = ""
    table_id: str = ""
    row_index: int = -1


class StructuredDocumentExtractor:
    """
    在文件扁平化前保留 HTML 或抽取文字中的表格列關係。

    Args:
     - max_table_rows: 每份文件最多保留的表格資料列數量。

    Returns:
     - StructuredDocumentExtractor: 結構化文件單位抽取器。
    """

    _SECTION_RE = re.compile(
        r"^(Metadata|Structured Data|Headings|Content|Tables|Lists|Captions):\s*$",
        flags=re.IGNORECASE,
    )
    _TABLE_RE = re.compile(r"^Table\s+(.+?)\s*$", flags=re.IGNORECASE)

    def __init__(self, *, max_table_rows: int = 80) -> None:
        self.max_table_rows = max(1, max_table_rows)

    def extract(self, content: str) -> list[StructuredDocumentUnit]:
        """
        將原始文件轉成一般文字與表格列單位。

        Args:
         - content: HTML 或 PageContentFetcher 產生的結構化文字。

        Returns:
         - list[StructuredDocumentUnit]: 表格列優先、一般內容在後的文件單位。
        """
        raw = str(content or "")
        if not raw.strip():
            return []

        html_units, html_body = self._extract_html(raw)
        source = html_body if html_units else raw
        text_units, body = self._extract_marked_tables(source)
        table_units = [*html_units, *text_units]
        units = list(table_units)
        if body.strip():
            units.append(StructuredDocumentUnit(text=body, unit_type="paragraph"))
        if not units:
            units.append(StructuredDocumentUnit(text=raw, unit_type="paragraph"))
        return self._deduplicate(units)

    def _extract_html(
        self,
        content: str,
    ) -> tuple[list[StructuredDocumentUnit], str]:
        if "<table" not in content.casefold():
            return [], content
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return [], content

        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")

        units: list[StructuredDocumentUnit] = []
        row_count = 0
        for table_index, table in enumerate(soup.find_all("table"), start=1):
            rows = self._html_rows(table)
            if not rows:
                table.decompose()
                continue
            headers, data_rows = self._headers_and_rows(rows)
            section = self._nearest_heading(table)
            for index, row in enumerate(data_rows, start=1):
                if row_count >= self.max_table_rows:
                    break
                units.append(
                    self._table_unit(
                        headers=headers,
                        row=row,
                        section=section,
                        table_id=f"Table {table_index}",
                        row_index=index,
                    )
                )
                row_count += 1
            table.decompose()
            if row_count >= self.max_table_rows:
                break
        return units, str(soup)

    def _extract_marked_tables(
        self,
        content: str,
    ) -> tuple[list[StructuredDocumentUnit], str]:
        lines = content.splitlines()
        units: list[StructuredDocumentUnit] = []
        body_lines: list[str] = []
        in_tables = False
        table_id = ""
        table_lines: list[str] = []
        row_count = 0

        def flush_table() -> None:
            nonlocal row_count
            if not table_lines or row_count >= self.max_table_rows:
                table_lines.clear()
                return
            rows = [self._split_row(line) for line in table_lines]
            rows = [row for row in rows if row]
            headers, data_rows = self._headers_and_rows(rows)
            for index, row in enumerate(data_rows, start=1):
                if row_count >= self.max_table_rows:
                    break
                units.append(
                    self._table_unit(
                        headers=headers,
                        row=row,
                        section="Tables",
                        table_id=table_id or "Table",
                        row_index=index,
                    )
                )
                row_count += 1
            table_lines.clear()

        for line in lines:
            stripped = line.strip()
            section_match = self._SECTION_RE.match(stripped)
            if section_match:
                section = section_match.group(1).casefold()
                if in_tables and section != "tables":
                    flush_table()
                    in_tables = False
                if section == "tables":
                    in_tables = True
                    continue
                body_lines.append(line)
                continue
            if not in_tables:
                body_lines.append(line)
                continue
            table_match = self._TABLE_RE.match(stripped)
            if table_match:
                flush_table()
                table_id = f"Table {table_match.group(1)}"
                continue
            if stripped:
                table_lines.append(stripped)

        if in_tables:
            flush_table()
        return units, "\n".join(body_lines)

    def _html_rows(self, table: object) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"])
            ]
            cells = [self._clean_cell(cell) for cell in cells]
            if any(cells):
                rows.append(cells)
        return rows

    def _nearest_heading(self, table: object) -> str:
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        if heading is None:
            return ""
        return self._clean_cell(heading.get_text(" ", strip=True))

    def _split_row(self, line: str) -> list[str]:
        if "|" not in line:
            return []
        return [
            self._clean_cell(cell)
            for cell in line.strip(" |").split("|")
        ]

    def _headers_and_rows(
        self,
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        if not rows:
            return [], []
        headers = rows[0]
        data_rows = rows[1:]
        if not data_rows:
            generated = [f"Column {index}" for index in range(1, len(headers) + 1)]
            return generated, [headers]
        return headers, data_rows

    def _table_unit(
        self,
        *,
        headers: list[str],
        row: list[str],
        section: str,
        table_id: str,
        row_index: int,
    ) -> StructuredDocumentUnit:
        width = max(len(headers), len(row))
        padded_headers = [*headers, *[f"Column {i}" for i in range(len(headers) + 1, width + 1)]]
        padded_row = [*row, *[""] * (width - len(row))]
        parts = []
        if section:
            parts.append(f"Section: {section}")
        parts.extend(
            [
                table_id,
                "Columns: " + " | ".join(padded_headers),
                "Row: " + " | ".join(padded_row),
            ]
        )
        return StructuredDocumentUnit(
            text="\n".join(parts),
            unit_type="table_row",
            section=section,
            table_id=table_id,
            row_index=row_index,
        )

    def _clean_cell(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _deduplicate(
        self,
        units: list[StructuredDocumentUnit],
    ) -> list[StructuredDocumentUnit]:
        result: list[StructuredDocumentUnit] = []
        seen: set[str] = set()
        for unit in units:
            key = re.sub(r"\W+", " ", unit.text.casefold()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(unit)
        return result


__all__ = ["StructuredDocumentExtractor", "StructuredDocumentUnit"]
