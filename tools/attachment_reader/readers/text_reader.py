from __future__ import annotations

import csv
import json
from pathlib import Path

from ..formatters import format_rows
from ..models import AttachmentReaderConfig


TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".jsonld", ".xml", ".pdb"}


class TextAttachmentReader:
    """TextAttachmentReader 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """初始化物件與必要狀態。"""
        self.config = config

    def read(self, file_path: Path, extension: str) -> str:
        """read 的主要實作。"""
        if extension == ".csv":
            return self._read_csv(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if extension in {".json", ".jsonld"}:
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                return text
        return text

    def _read_csv(self, file_path: Path) -> str:
        """_read_csv 的內部輔助實作。"""
        rows: list[list[str]] = []
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if idx >= self.config.max_table_rows:
                    break
                rows.append([str(cell) for cell in row])
        if not rows:
            return "(empty csv)"
        return format_rows(
            "CSV",
            rows,
            truncated=len(rows) >= self.config.max_table_rows,
            max_rows=self.config.max_table_rows,
        )

