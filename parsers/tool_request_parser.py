from __future__ import annotations

from typing import Any

from .json_parse import try_parse_json
from .stage1_output_parser import Stage1OutputParser
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
        self.structured_parser = Stage1OutputParser(answer_validator=self.validator)

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
                tool_args = (
                    dict(parsed.get("tool_args"))
                    if isinstance(parsed.get("tool_args"), dict)
                    else {}
                )
                if parsed.get("missing_information") and "missing_information" not in tool_args:
                    tool_args["missing_information"] = str(
                        parsed.get("missing_information") or ""
                    ).strip()
                return {
                    "type": "tool_request",
                    "reasoning_step": str(parsed.get("reasoning_step", "") or "").strip(),
                    "tool_name": str(parsed.get("tool_name", "") or "").strip(),
                    "tool_args": tool_args,
                    "reasoning": "",
                    "final_answer": "",
                    "structured_output": {},
                    "schema_valid": True,
                    "schema_errors": [],
                    "repair_applied": False,
                    "repair_actions": [],
                    "eligible_for_winner": False,
                    "validity_labels": ["tool_request_pending"],
                }
            if reply_type == "final_answer":
                structured = self.structured_parser.parse(raw_reply, expected_weight_count=0)
                return self._build_final_answer_payload(structured)

        try:
            fallback = self.structured_parser.parse(raw_reply, expected_weight_count=0)
        except Exception:
            return self._invalid()

        final_answer = self.validator.clean(fallback.get("final_answer", ""))
        if final_answer:
            return self._build_final_answer_payload(fallback)

        return self._invalid()

    def _build_final_answer_payload(
        self,
        structured: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one lossless final-answer contract for every parse path."""

        return {
            "type": "final_answer",
            "reasoning_step": "",
            "tool_name": "",
            "tool_args": {},
            "reasoning": str(structured.get("reasoning", "") or "").strip(),
            "final_answer": self.validator.clean(structured.get("final_answer", "")),
            "reasoning_parse": dict(structured.get("reasoning_parse") or {}),
            "structured_output": dict(structured.get("structured_output") or {}),
            "schema_valid": bool(structured.get("schema_valid")),
            "schema_errors": list(structured.get("schema_errors") or []),
            "repair_applied": bool(structured.get("repair_applied")),
            "repair_actions": list(structured.get("repair_actions") or []),
            "eligible_for_winner": bool(structured.get("eligible_for_winner")),
            "validity_labels": list(structured.get("validity_labels") or []),
        }

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
            "reasoning_parse": {},
            "structured_output": {},
            "schema_valid": False,
            "schema_errors": ["invalid_tool_reply"],
            "repair_applied": False,
            "repair_actions": [],
            "eligible_for_winner": False,
            "validity_labels": ["invalid_tool_reply"],
        }


__all__ = ["ToolRequestParser"]
