from __future__ import annotations

from typing import Any

from .json_parse import try_parse_json
from .stage1_reply_parser import Stage1ReplyParser
from score.answer_validator import AnswerValidator


class ToolRequestParser:
    """
    解析 Stage1 tool-use 模式的 Agent 回覆，區分 tool_request、final_answer 與 invalid。

    Args:
        - 無。

    Returns:
        - ToolRequestParser: 可解析 tool-use trajectory 回覆的 parser。
    """

    def __init__(self) -> None:
        self.validator = AnswerValidator()
        self.stage1_parser = Stage1ReplyParser()

    def parse(self, raw_reply: str) -> dict[str, Any]:
        """
        解析單次 tool-use 回覆，優先處理 JSON，再 fallback 到 Stage1ReplyParser。

        Args:
            - raw_reply: Agent 回傳的原始文字。

        Returns:
            - dict[str, Any]: type 為 tool_request、final_answer 或 invalid 的標準化結果。
        """
        parsed = try_parse_json(raw_reply)
        if isinstance(parsed, dict):
            reply_type = str(parsed.get("type", "")).strip().lower()
            if reply_type == "tool_request":
                return {
                    "type": "tool_request",
                    "reasoning_step": str(parsed.get("reasoning_step", "") or "").strip(),
                    "tool_name": str(parsed.get("tool_name", "") or "").strip(),
                    "tool_args": parsed.get("tool_args") if isinstance(parsed.get("tool_args"), dict) else {},
                    "reasoning": "",
                    "final_answer": "",
                }
            if reply_type == "final_answer":
                final_answer = self.validator.clean(parsed.get("final_answer", ""))
                if not self.validator.is_valid(final_answer):
                    return self._invalid()
                return {
                    "type": "final_answer",
                    "reasoning_step": "",
                    "tool_name": "",
                    "tool_args": {},
                    "reasoning": str(parsed.get("reasoning", "") or "").strip(),
                    "final_answer": final_answer,
                }

        try:
            fallback = self.stage1_parser.parse(raw_reply, expected_weight_count=0)
        except Exception:
            return self._invalid()

        final_answer = self.validator.clean(fallback.get("final_answer", ""))
        reasoning = str(fallback.get("reasoning", "") or "").strip()
        if self.validator.is_valid(final_answer):
            return {
                "type": "final_answer",
                "reasoning_step": "",
                "tool_name": "",
                "tool_args": {},
                "reasoning": reasoning,
                "final_answer": final_answer,
            }

        return self._invalid()

    def _invalid(self) -> dict[str, Any]:
        """
        建立標準 invalid parse result。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 空欄位的 invalid 結果。
        """
        return {
            "type": "invalid",
            "reasoning_step": "",
            "tool_name": "",
            "tool_args": {},
            "reasoning": "",
            "final_answer": "",
        }


__all__ = ["ToolRequestParser"]
