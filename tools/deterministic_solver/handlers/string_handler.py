from __future__ import annotations

import re
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, first_quoted_value, lower_text


class StringHandler:
    """StringHandler 類別，封裝此模組的資料結構與服務邏輯。"""

    def solve(self, question: str, **_: Any) -> DeterministicSolverResult:
        """
        ??? deterministic ?????????
        
        Args:
            - ????????????
        
        Returns:
            - DeterministicSolverResult ????????
        """
        text = clean_text(question)
        lowered = lower_text(text)
        value = first_quoted_value(text) or self._value_after_colon(text)
        if not value:
            return DeterministicSolverResult.miss("string")

        if any(term in lowered for term in ["uppercase", "upper case", "capital letters"]):
            return self._result("string_uppercase", value.upper(), value)
        if any(term in lowered for term in ["lowercase", "lower case"]):
            return self._result("string_lowercase", value.lower(), value)
        if "title case" in lowered:
            return self._result("string_titlecase", value.title(), value)
        if "reverse" in lowered:
            return self._result("string_reverse", value[::-1], value)
        if "remove spaces" in lowered or "without spaces" in lowered:
            return self._result("string_remove_spaces", re.sub(r"\s+", "", value), value)
        if "character count" in lowered or "how many characters" in lowered or "count characters" in lowered:
            count_spaces = "including spaces" in lowered
            target = value if count_spaces else value.replace(" ", "")
            return self._result("string_character_count", str(len(target)), value)
        if "word count" in lowered or "how many words" in lowered or "count words" in lowered:
            return self._result("string_word_count", str(len(value.split())), value)

        return DeterministicSolverResult.miss("string")

    def _value_after_colon(self, text: str) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if ":" not in text:
            return ""
        return text.split(":", 1)[1].strip(" .")

    def _result(self, task_type: str, answer: str, source: str) -> DeterministicSolverResult:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type=task_type,
            answer=answer,
            answer_text=answer,
            confidence=0.92,
            evidence={"source_text": source},
        )

