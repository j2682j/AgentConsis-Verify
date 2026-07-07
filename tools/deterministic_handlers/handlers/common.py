from __future__ import annotations

import csv
import io
from pathlib import Path
import re
from typing import Any


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def attachment_path(attachment: dict[str, Any]) -> Path | None:
    for key in ("file_path", "path"):
        value = attachment.get(key) if isinstance(attachment, dict) else None
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    return None


def attachment_extension(attachment: dict[str, Any]) -> str:
    if not isinstance(attachment, dict):
        return ""
    extension = str(attachment.get("extension", "") or "").strip().lower()
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    path = attachment_path(attachment)
    return path.suffix.lower() if path else ""


def read_delimited_rows(path: Path) -> list[list[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return []
    try:
        return [
            [normalize_text(cell) for cell in row]
            for row in csv.reader(io.StringIO(text), delimiter=delimiter)
            if row
        ]
    except Exception:
        return []


def parse_inline_delimited_rows(text: str) -> list[list[str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    markdown_rows = [line for line in lines if "|" in line]
    if markdown_rows:
        rows = []
        for line in markdown_rows:
            cells = [normalize_text(cell) for cell in line.strip("|").split("|")]
            if cells and not all(re.fullmatch(r"-+", cell.replace(" ", "")) for cell in cells):
                rows.append(cells)
        return rows
    csv_lines = [line for line in lines if "," in line]
    if csv_lines:
        try:
            return [
                [normalize_text(cell) for cell in row]
                for row in csv.reader(io.StringIO("\n".join(csv_lines)))
                if row
            ]
        except Exception:
            return []
    return []


def extract_quoted_or_word_pair(question: str) -> tuple[str, str]:
    quoted = re.findall(r'"([^"]+)"|' + r"'([^']+)'", question or "")
    values = [left or right for left, right in quoted if left or right]
    if len(values) >= 2:
        return values[0], values[1]
    match = re.search(
        r"\bfrom\s+([A-Za-z0-9_. -]+?)\s+to\s+([A-Za-z0-9_. -]+?)(?:[?.;,]|$)",
        question or "",
        re.IGNORECASE,
    )
    if match:
        return normalize_text(match.group(1)), normalize_text(match.group(2))
    return "", ""


__all__ = [
    "attachment_extension",
    "attachment_path",
    "extract_quoted_or_word_pair",
    "normalize_text",
    "parse_inline_delimited_rows",
    "read_delimited_rows",
]
