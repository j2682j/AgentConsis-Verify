from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from parsers.json_parse import try_parse_json
from score.answer_validator import AnswerValidator


@dataclass
class SelfReviewResult:
    """
    Agent self-review 回覆的解析結果。

    Args:
     - reply_type: final_answer、tool_request 或 invalid。
     - answer: review 後的最終答案。
     - tool_name: 要求呼叫的工具名稱。
     - tool_args: 工具參數。
     - raw_reply: Agent 原始輸出。
     - parse_completed: 是否成功解析。

    Returns:
     - SelfReviewResult: Stage1Runner 可用於覆蓋 Agent summary 的結果。
    """

    reply_type: str = "invalid"
    answer: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    raw_reply: str = ""
    parse_completed: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.reply_type,
            "answer": self.answer,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "raw_reply": self.raw_reply,
            "parse_completed": self.parse_completed,
            "error": self.error,
        }


class SelfReviewParser:
    """
    解析 Stage1 self-review 的 JSON-only 回覆。

    Args:
     - answer_validator: 清理與驗證 final answer。

    Returns:
     - SelfReviewParser: 提供 parse(raw_reply) 方法。
    """

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def parse(self, raw_reply: str) -> SelfReviewResult:
        raw = str(raw_reply or "").strip()
        parsed = try_parse_json(raw)
        if isinstance(parsed, dict):
            reply_type = str(parsed.get("type", "") or "").strip().lower()
            if reply_type == "final_answer":
                answer = self.answer_validator.clean(str(parsed.get("answer", "") or ""))
                return SelfReviewResult(
                    reply_type="final_answer",
                    answer=answer,
                    raw_reply=raw,
                    parse_completed=bool(answer and self.answer_validator.is_valid(answer)),
                    error="" if answer else "empty_answer",
                )
            if reply_type == "tool_request":
                tool_args = parsed.get("tool_args")
                if not isinstance(tool_args, dict):
                    tool_args = parsed.get("arguments")
                if not isinstance(tool_args, dict):
                    tool_args = {}
                tool_name = str(parsed.get("tool_name", "") or "").strip()
                return SelfReviewResult(
                    reply_type="tool_request",
                    tool_name=tool_name,
                    tool_args=dict(tool_args),
                    raw_reply=raw,
                    parse_completed=bool(tool_name),
                    error="" if tool_name else "empty_tool_name",
                )

        answer = self._extract_answer_v2(raw)
        if answer:
            answer = self.answer_validator.clean(answer)
            return SelfReviewResult(
                reply_type="final_answer",
                answer=answer,
                raw_reply=raw,
                parse_completed=bool(self.answer_validator.is_valid(answer)),
                error="fallback_answer_extraction",
            )
        return SelfReviewResult(
            raw_reply=raw,
            parse_completed=False,
            error="invalid_self_review_reply",
        )

    def _extract_answer(self, raw: str) -> str:
        patterns = [
            r'(?is)"answer"\s*:\s*"([^"]+)"',
            r"(?im)^\s*final\s+answer\s*[:：]\s*(.+?)\s*$",
            r"(?im)^\s*(?:\*\*)?\s*final\s+answer\s*(?:\*\*)?\s*$\s*(.+?)\s*$",
            r"(?im)^\s*(?:#{1,6}\s*)?final\s+answer\s*$\s*(.+?)\s*$",
            r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                return str(match.group(1) or "").strip()
        return ""

    def _extract_answer_v2(self, raw: str) -> str:
        patterns = [
            r'(?is)"answer"\s*:\s*"([^"]+)"',
            r"(?im)^\s*final\s+answer\s*[:\uFF1A]\s*(.+?)\s*$",
            r"(?im)^\s*answer\s*[:\uFF1A]\s*(.+?)\s*$",
            r"(?im)^\s*(?:\*\*)?\s*final\s+answer\s*(?:\*\*)?\s*$\s*(.+?)\s*$",
            r"(?im)^\s*(?:#{1,6}\s*)?final\s+answer\s*$\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                answer = str(match.group(1) or "").strip()
                answer = re.sub(r"^\s*[-*]\s+", "", answer).strip()
                return answer
        return ""


__all__ = ["SelfReviewParser", "SelfReviewResult"]
