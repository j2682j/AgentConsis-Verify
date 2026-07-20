from __future__ import annotations

from typing import Any

from parsers.json_parse import try_parse_json

from .models import AttachmentStrategy


class AttachmentStrategyParser:
    """
    解析並輕量修復附件策略 JSON。

    Args:
     - allowed_handlers: 系統允許的 handler 名稱或 handler role。

    Returns:
     - AttachmentStrategyParser: 將模型輸出轉成 AttachmentStrategy 的解析器。
    """

    def __init__(self, allowed_handlers: set[str] | None = None) -> None:
        self.allowed_handlers = {
            self._normalize_handler(value)
            for value in (allowed_handlers or set())
            if self._normalize_handler(value)
        }

    def parse(self, raw_reply: str) -> AttachmentStrategy:
        parsed = try_parse_json(raw_reply)
        if not isinstance(parsed, dict):
            return AttachmentStrategy(missing_inputs=["invalid_strategy_json"])
        return self.from_dict(parsed)

    def from_dict(self, data: dict[str, Any]) -> AttachmentStrategy:
        normalized_handler = self._normalize_handler(data.get("required_handler"))
        if self.allowed_handlers and normalized_handler not in self.allowed_handlers:
            normalized_handler = ""

        return AttachmentStrategy(
            information_need=str(data.get("information_need") or "").strip(),
            required_handler=normalized_handler,
            required_inputs=self._string_list(data.get("required_inputs") or [])[:6],
            expected_answer=str(data.get("expected_answer") or "").strip(),
            needs_search=bool(data.get("needs_search", False)),
            missing_inputs=self._string_list(data.get("missing_inputs") or [])[:6],
            next_capability=str(data.get("next_capability") or "").strip().lower(),
        )

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []
        return [str(item).strip() for item in values if str(item).strip()]

    def _normalize_handler(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = text.replace("-", "_").replace(" ", "_")
        aliases = {
            "spreadsheet": "table_reasoning",
            "spreadsheet_table_solver": "table_reasoning",
            "table": "table_reasoning",
            "table_solver": "table_reasoning",
            "excel": "table_reasoning",
            "csv": "table_reasoning",
            "pdf": "text_extraction",
            "pdf_text_solver": "text_extraction",
            "text": "text_extraction",
            "ocr": "text_extraction",
            "math": "numeric_arithmetic",
            "calculator": "numeric_arithmetic",
            "numeric": "numeric_arithmetic",
            "unit": "unit_conversion",
            "units": "unit_conversion",
            "graph": "graph_search",
            "route": "graph_search",
            "boggle": "boggle_dfs",
            "word_grid": "boggle_dfs",
            "date": "date_time",
            "time": "date_time",
            "coordinate": "coordinate_distance",
            "distance": "coordinate_distance",
            "list": "list_operation",
            "logic": "logic_equivalence",
            "boolean": "logic_equivalence",
            "probability": "probability_simulation",
            "odds": "probability_simulation",
            "random": "probability_simulation",
            "counting": "multi_step_counting",
            "count": "multi_step_counting",
            "chess": "chess_tactics",
            "archive": "text_extraction",
            "zip": "text_extraction",
            "image": "text_extraction",
            "vision": "text_extraction",
        }
        return aliases.get(text, text)


__all__ = ["AttachmentStrategyParser"]
