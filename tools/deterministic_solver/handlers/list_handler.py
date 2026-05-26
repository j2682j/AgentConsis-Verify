"""清單 deterministic handler。"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, lower_text, split_items


class ListHandler:
    """處理清單排序、計數與第 N 項抽取。"""

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
        items = self._extract_items(text)
        if not items:
            return DeterministicSolverResult.miss("list")
        if "count" in lowered or "how many items" in lowered:
            return self._result("list_count", str(len(items)), items)
        ordinal = self._extract_ordinal(lowered)
        if ordinal is not None and 0 <= ordinal < len(items):
            return self._result("list_nth_item", items[ordinal], items)
        if "sort" in lowered or "order" in lowered or "alphabetically" in lowered:
            reverse = any(term in lowered for term in ["descending", "reverse order", "z to a", "largest to smallest"])
            return self._result("list_sort", ", ".join(self._sort_items(items, reverse=reverse)), items)
        return DeterministicSolverResult.miss("list")

    def _extract_items(self, text: str) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        tail = text.split(":", 1)[1] if ":" in text else ""
        items = split_items(tail)
        if len(items) <= 1:
            numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
            items = numbers if len(numbers) >= 2 else []
        return [item.strip(" .") for item in items if item.strip(" .")]

    def _sort_items(self, items: list[str], *, reverse: bool) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", item) for item in items):
            return [item for _, item in sorted(((Decimal(item), item) for item in items), reverse=reverse)]
        return sorted(items, key=lambda item: item.lower(), reverse=reverse)

    def _extract_ordinal(self, lowered: str) -> int | None:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        mapping = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
        for word, index in mapping.items():
            if word in lowered:
                return index
        match = re.search(r"(\d+)(?:st|nd|rd|th)", lowered)
        return int(match.group(1)) - 1 if match else None

    def _result(self, task_type: str, answer: str, items: list[str]) -> DeterministicSolverResult:
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
            confidence=0.88,
            evidence={"items": items},
        )
