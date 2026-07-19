from __future__ import annotations

import json
import re
from typing import Any


class AnswerValidator:
    """
    驗證與清理 final answer，避免工具呼叫、拒答文字或過長解釋進入 scoring。

    Args:
        - 無。

    Returns:
        - AnswerValidator: 可用於 clean、is_valid 與各類 final answer 檢查的驗證器。
    """

    REFUSAL_PATTERNS = (
        r"\binformation unavailable\b",
        r"\bnot available\b",
        r"\bcannot be determined\b",
        r"\bcannot determine\b",
        r"\bcould not be identified\b",
        r"\bcould not find\b",
        r"\binsufficient (data|information|evidence)\b",
        r"\bmissing image data\b",
        r"\bunknown\b",
        r"\bnot provided\b",
        r"\bnot present in the provided search results\b",
        r"\bcannot determine\b",
        r"\bcan't determine\b",
        r"\bcan not determine\b",
        r"\bnot enough (data|information|evidence)\b",
        r"\bunknown\b",
        r"\bno answer\b",
        r"\bneed_more_evidence\b",
        r"\bneed more evidence\b",
        r"\bnone\b",
        r"^n/?a$",
    )

    UNCERTAINTY_PATTERNS = (
        r"\bappears to be\b",
        r"\blikely\b",
        r"\bprobably\b",
        r"\bI think\b",
        r"\bbased on limited information\b",
        r"\bthe available evidence suggests\b"
    )

    TOOL_KEYS = {"tool_name", "tool_args", "arguments", "name", "function", "type"}
    TOOL_TYPES = {"tool_request", "function_call", "tool_call"}

    def clean(self, answer: Any) -> str:
        """
        清理 final answer 的外層格式，例如 code fence、引號、markdown bold 與欄位前綴。

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - str: 清理後的答案文字。
        """
        candidate = "" if answer is None else str(answer)
        candidate = candidate.strip()
        if not candidate:
            return ""

        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

        if (
            (candidate.startswith('"') and candidate.endswith('"'))
            or (candidate.startswith("'") and candidate.endswith("'"))
        ):
            candidate = candidate[1:-1].strip()

        candidate = re.sub(r"^\*\*(.*?)\*\*$", r"\1", candidate, flags=re.DOTALL).strip()
        candidate = re.sub(r"^\*+\s*", "", candidate).strip()
        candidate = re.sub(r"\s*\*+$", "", candidate).strip()
        candidate = re.sub(r"\*+(?=[\s.,;:!?]*$)", "", candidate).strip()
        candidate = re.sub(r"`+(?=[\s.,;:!?]*$)", "", candidate).strip()
        candidate = re.sub(r"^FINAL[_ ]ANSWER\s*[:=]\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^ANSWER\s*[:=]\s*", "", candidate, flags=re.IGNORECASE).strip()
        boolean_match = re.fullmatch(r"(yes|no)\.?", candidate, re.IGNORECASE)
        if boolean_match:
            return boolean_match.group(1).lower()
        candidate = " ".join(candidate.split()) if "\n" not in candidate else candidate.strip()
        return candidate.strip()

    def is_valid(self, answer: Any, *, answer_type: str = "") -> bool:
        """
        判斷候選 final answer 是否可進入 voting、scoring 與 winner selection。

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - bool: 若答案非空、非工具呼叫、非拒答且格式合理則回傳 True。
        """
        candidate = self.clean(answer)
        if not candidate:
            return False
        if candidate in {"}", "{", "]", "[", ")", "(", "$", "$$", "```"}:
            return False
        if self.is_tool_call_like(candidate):
            return False
        if self.is_refusal_like(candidate):
            return False
        if self.is_too_verbose(candidate, answer_type=answer_type):
            return False
        if re.search(r"(?:REASONING|WEIGHTS)\s*=", candidate, re.IGNORECASE):
            return False
        if re.fullmatch(r"F?I?INAL_?ANSWE?R?", candidate, re.IGNORECASE):
            return False
        if re.fullmatch(r"[\W_]+", candidate):
            return False
        return True

    def is_tool_call_like(self, answer: Any) -> bool:
        """
        判斷候選答案是否像 tool call、tool request 或 function call JSON。

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - bool: 若答案看起來是工具呼叫而不是最終答案則回傳 True。
        """
        candidate = self.clean(answer)
        if not candidate:
            return False

        parsed = self._parse_json_fragment(candidate)
        if parsed is not None:
            return self._json_looks_like_tool_call(parsed)

        lowered = candidate.lower()
        if '"type"' in lowered and "tool_request" in lowered:
            return True
        if '"tool_name"' in lowered or '"tool_args"' in lowered:
            return True
        if re.search(r"\b(search|python_calculator)\s*\(", lowered):
            return True
        return False

    def is_refusal_like(self, answer: Any) -> bool:
        """
        判斷候選答案是否屬於 None、unknown、insufficient data 等拒答或無答案文字。

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - bool: 若答案看起來是拒答或無答案則回傳 True。
        """
        candidate = self.clean(answer).strip().lower()
        if not candidate:
            return True
        return any(re.search(pattern, candidate, re.IGNORECASE) for pattern in self.REFUSAL_PATTERNS)

    def is_too_verbose(self, answer: Any, *, answer_type: str = "") -> bool:
        """
        判斷 final answer 是否過長或包含過多行，避免長段解釋污染短答案任務。

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - bool: 若答案超過長度或行數限制則回傳 True。
        """
        candidate = self.clean(answer)
        if self._is_compact_list_answer(candidate, answer_type=answer_type):
            return False
        if len(candidate) > 50:
            return True
        if len(candidate.splitlines()) > 1:
            return True
        return False

    def _is_compact_list_answer(self, answer: str, *, answer_type: str = "") -> bool:
        """Accept long enumerations while continuing to reject prose explanations."""
        candidate = str(answer or "").strip()
        if not candidate or len(candidate.splitlines()) != 1 or len(candidate) > 1000:
            return False
        parts = [part.strip() for part in re.split(r"[,;]", candidate)]
        if len(parts) < 2 or any(not part or len(part) > 64 for part in parts):
            return False
        if any(re.search(r"[.!?](?:\s|$)", part) for part in parts):
            return False
        declared_list = str(answer_type or "").strip().lower() == "list"
        compact_items = all(len(part.split()) <= 8 for part in parts)
        return declared_list or compact_items

    def _parse_json_fragment(self, text: str) -> Any | None:
        """
        嘗試從文字中解析完整或片段 JSON。

        Args:
            - text: 可能包含 JSON 的文字。

        Returns:
            - Any | None: 解析成功的 JSON 物件；失敗時回傳 None。
        """
        candidate = text.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        start_positions = [pos for pos in (candidate.find("{"), candidate.find("[")) if pos >= 0]
        if not start_positions:
            return None
        start = min(start_positions)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end < start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _json_looks_like_tool_call(self, value: Any) -> bool:
        """
        判斷已解析 JSON 是否符合工具呼叫或 function call 形狀。

        Args:
            - value: 已解析的 JSON 值。

        Returns:
            - bool: 若 JSON 結構看起來是工具呼叫則回傳 True。
        """
        if isinstance(value, list):
            return bool(value) and all(self._json_looks_like_tool_call(item) for item in value)

        if not isinstance(value, dict):
            return False

        value_type = str(value.get("type", "") or "").strip().lower()
        if value_type in self.TOOL_TYPES:
            return True

        keys = set(value)
        if {"name", "arguments"}.issubset(keys):
            return True
        if {"tool_name", "tool_args"}.issubset(keys):
            return True
        return bool(keys & self.TOOL_KEYS) and not {"final_answer", "reasoning"} & keys
    

    def is_uncertain(self, answer: str) -> bool:
        """
        判斷候選答案是否表達不確定性，例如包含 "not sure"、"uncertain" 等字樣

        Args:
            - answer: 任意型別的候選 final answer。

        Returns:
            - bool: 若答案看起來表達不確定性則回傳 True。
        """
        candidate = self.clean(answer).strip().lower()
        if not candidate:
            return True
        return any(re.search(pattern, candidate, re.IGNORECASE) for pattern in self.UNCERTAINTY_PATTERNS)

        
    def question_allow_refusal(self, question: str) -> bool:
        """
        判斷題目是否允許拒答，若題目中包含 "if you don't know"、"if you are unsure" 等字樣則視為允許拒答。

        Args:
            - question: 任意型別的題目文字。

        Returns:
            - bool: 若題目中包含允許拒答的提示則回傳 True。
        """
        candidate = self.clean(question).strip().lower()
        if not candidate:
            return False
        refusal_clues = [
            r"if you don't know",
            r"if you are unsure",
            r"if you cannot determine",
            r"if you cannot answer",
            r"if the answer is unknown",
            r"if the information is unavailable",
        ]
        return any(re.search(pattern, candidate, re.IGNORECASE) for pattern in refusal_clues)



__all__ = ["AnswerValidator"]
