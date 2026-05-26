from __future__ import annotations

import re
from typing import Any

from .json_parse import try_parse_json
from score.answer_validator import AnswerValidator


class Stage1ReplyParser:
    """
    解析一般 Stage1 Agent 回覆，抽出 reasoning、final_answer 與 weights。

    Args:
        - validator: 驗證 final answer 格式與有效性的 AnswerValidator。

    Returns:
        - Stage1ReplyParser: 可解析 Stage1 raw reply 的 parser。
    """

    def __init__(self, validator: AnswerValidator | None = None) -> None:
        self.validator = validator or AnswerValidator()

    def parse(self, reply: str, expected_weight_count: int = 0) -> dict[str, Any]:
        """
        解析 Stage1 raw reply，支援 JSON 與 REASONING/FINAL_ANSWER 文字格式。

        Args:
            - reply: Agent 回傳的原始文字。
            - expected_weight_count: 預期 weights 數量，0 表示不需要 weights。

        Returns:
            - dict[str, Any]: 包含 reasoning、final_answer、weights、parse_completed 與 parse_error。
        """
        if not str(reply or "").strip():
            raise ValueError("Empty stage1 reply.")

        parsed_json = try_parse_json(reply)
        if isinstance(parsed_json, dict):
            return self._parse_json_reply(parsed_json, expected_weight_count)

        reasoning = self.extract_reasoning(reply)
        final_answer = self.extract_final_answer(reply)
        weights = self.extract_weights(reply, expected_weight_count)
        if not final_answer:
            raise ValueError("Missing or invalid FINAL_ANSWER in stage1 reply.")

        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "weights": weights,
            "parse_completed": True,
            "parse_error": None,
        }

    def _parse_json_reply(
        self,
        parsed: dict[str, Any],
        expected_weight_count: int,
    ) -> dict[str, Any]:
        """
        解析已轉成 dict 的 Stage1 JSON 回覆。

        Args:
            - parsed: 已解析的 JSON dict。
            - expected_weight_count: 預期 weights 數量。

        Returns:
            - dict[str, Any]: 標準化後的 Stage1 parsed result。
        """
        reply_type = str(parsed.get("type", "") or "").strip().lower()
        if reply_type == "tool_request":
            raise ValueError("Tool request cannot be parsed as final answer.")

        final_answer = self._first_present(
            parsed,
            ["final_answer", "correct_answer", "answer", "final", "result", "output"],
        )
        final_answer = self.validator.clean(final_answer)
        if not self.validator.is_valid(final_answer):
            raise ValueError(f"Invalid final answer: {final_answer!r}")

        reasoning = str(parsed.get("reasoning", "") or "").strip()
        weights = self._normalize_weights(parsed.get("weights"), expected_weight_count)
        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "weights": weights,
            "parse_completed": True,
            "parse_error": None,
        }

    def extract_reasoning(self, reply: str) -> str:
        """
        從 Stage1 raw reply 中抽取 reasoning 區塊。

        Args:
            - reply: Agent 回傳的原始文字。

        Returns:
            - str: 抽取出的 reasoning 文字；無法抽取時回傳 fallback 文字。
        """
        patterns = [
            r"REASONING\s*=\s*(.*?)(?=\n\s*FINAL[_ ]ANSWER\s*[:=]|\n\s*WEIGHTS\s*=|$)",
            r"reasoning\s*[:=]\s*(.*?)(?=\n\s*FINAL[_ ]ANSWER\s*[:=]|\n\s*WEIGHTS\s*=|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, reply or "", re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        lines = self._nonempty_lines(reply)
        filtered = [
            line
            for line in lines
            if not re.match(r"(FINAL[_ ]ANSWER|ANSWER|WEIGHTS)\s*[:=]", line, re.IGNORECASE)
        ]
        return "\n".join(filtered).strip()

    def extract_final_answer(self, reply: str) -> str:
        """
        從 Stage1 raw reply 中抽取 final answer，並通過 AnswerValidator 檢查。

        Args:
            - reply: Agent 回傳的原始文字。

        Returns:
            - str: 合法 final answer；無合法答案時回傳空字串。
        """
        patterns = [
            r"FINAL[_ ]ANSWER\s*[:=]\s*(.+)",
            r"FINAL ANSWER\s*:\s*(.+)",
            r"ANSWER\s*:\s*(.+)",
            r"\\boxed\{([^{}]+)\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, reply or "", re.IGNORECASE)
            if not match:
                continue
            candidate = self.validator.clean(match.group(1))
            if self.validator.is_valid(candidate):
                return candidate

        lines = self._nonempty_lines(reply)
        if not lines:
            return ""
        last_line = lines[-1]
        if re.match(r"WEIGHTS\s*=", last_line, re.IGNORECASE) and len(lines) >= 2:
            last_line = lines[-2]
        candidate = self.validator.clean(last_line)
        if self._looks_like_short_answer(candidate) and self.validator.is_valid(candidate):
            return candidate
        return ""

    def extract_weights(self, reply: str, expected_weight_count: int = 0) -> list[int]:
        """
        從 Stage1 raw reply 中抽取 WEIGHTS 欄位並正規化到 1 到 5。

        Args:
            - reply: Agent 回傳的原始文字。
            - expected_weight_count: 預期 weights 數量。

        Returns:
            - list[int]: 正規化後的 weights；缺失或數量不符時回傳 fallback weights。
        """
        if expected_weight_count <= 0:
            return []

        match = re.search(r"WEIGHTS\s*=\s*\[([^\]]*)\]", reply or "", re.IGNORECASE | re.DOTALL)
        if not match:
            return self.fallback_weights(expected_weight_count)
        raw_items = [item.strip() for item in match.group(1).split(",") if item.strip()]
        weights: list[int] = []
        for item in raw_items:
            try:
                value = float(item)
            except ValueError:
                return self.fallback_weights(expected_weight_count)
            mapped = int(round(1 + value * 4)) if 0.0 <= value <= 1.0 else int(round(value))
            weights.append(max(1, min(5, mapped)))

        if len(weights) != expected_weight_count:
            return self.fallback_weights(expected_weight_count)
        return weights

    def fallback_weights(self, expected_weight_count: int) -> list[int]:
        """
        產生 Stage1 weights fallback 值。

        Args:
            - expected_weight_count: 預期 weights 數量。

        Returns:
            - list[int]: 長度為 expected_weight_count、值皆為 3 的預設 weights。
        """
        return [3] * max(0, expected_weight_count)

    def _normalize_weights(self, weights: Any, expected_weight_count: int) -> list[int]:
        """
        將 JSON weights 欄位正規化到整數分數範圍。

        Args:
            - weights: JSON 中的 weights 欄位。
            - expected_weight_count: 預期 weights 數量。

        Returns:
            - list[int]: 正規化後的 weights。
        """
        if expected_weight_count <= 0:
            return []
        if weights is None:
            return self.fallback_weights(expected_weight_count)
        if not isinstance(weights, list):
            raise TypeError("'weights' must be a list.")

        normalized: list[int] = []
        for item in weights:
            try:
                value = float(item)
            except (TypeError, ValueError):
                return self.fallback_weights(expected_weight_count)
            mapped = int(round(1 + value * 4)) if 0.0 <= value <= 1.0 else int(round(value))
            normalized.append(max(1, min(5, mapped)))
        if len(normalized) != expected_weight_count:
            return self.fallback_weights(expected_weight_count)
        return normalized

    def _first_present(self, data: dict[str, Any], keys: list[str]) -> Any:
        """
        依序從 dict 中取出第一個存在的欄位值。

        Args:
            - data: 要查找的 dict。
            - keys: 欄位優先順序。

        Returns:
            - Any: 第一個存在欄位的值；不存在時回傳 None。
        """
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _nonempty_lines(self, text: str) -> list[str]:
        """
        將文字切成非空白行。

        Args:
            - text: 原始文字。

        Returns:
            - list[str]: 去除前後空白後的非空白行。
        """
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]

    def _looks_like_short_answer(self, text: str, max_chars: int = 160) -> bool:
        """
        判斷文字是否像短 final answer，可作為 fallback answer。

        Args:
            - text: 候選文字。
            - max_chars: 允許的最大字元數。

        Returns:
            - bool: 若文字短且不含明顯結構符號則回傳 True。
        """
        candidate = str(text or "").strip()
        if not candidate:
            return False
        if len(candidate) > max_chars:
            return False
        if re.search(r"[{}\[\]`]", candidate):
            return False
        return True


__all__ = ["Stage1ReplyParser"]
