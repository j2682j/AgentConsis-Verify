"""單位換算 deterministic handler。"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, format_decimal, lower_text


class UnitHandler:
    """處理常見線性單位換算。"""

    LINEAR_FACTORS = {
        ("mm", "cm"): Decimal("0.1"),
        ("cm", "mm"): Decimal("10"),
        ("cm", "m"): Decimal("0.01"),
        ("m", "cm"): Decimal("100"),
        ("m", "km"): Decimal("0.001"),
        ("km", "m"): Decimal("1000"),
        ("g", "kg"): Decimal("0.001"),
        ("kg", "g"): Decimal("1000"),
        ("mg", "g"): Decimal("0.001"),
        ("g", "mg"): Decimal("1000"),
        ("sec", "min"): Decimal("0.01666666666666666666666666667"),
        ("min", "sec"): Decimal("60"),
        ("hour", "minute"): Decimal("60"),
        ("minute", "hour"): Decimal("0.01666666666666666666666666667"),
    }
    UNIT_ALIASES = {
        "millimeter": "mm",
        "millimeters": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "meter": "m",
        "meters": "m",
        "kilometer": "km",
        "kilometers": "km",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "seconds": "sec",
        "second": "sec",
        "minutes": "minute",
        "hours": "hour",
    }

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
        if "convert" not in lowered and "conversion" not in lowered:
            return DeterministicSolverResult.miss("unit")
        match = re.search(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*([a-z]+)\s+(?:to|in)\s+([a-z]+)", lowered)
        if not match:
            return DeterministicSolverResult.miss("unit")
        value = Decimal(match.group(1).replace(",", ""))
        source_unit = self._normalize_unit(match.group(2))
        target_unit = self._normalize_unit(match.group(3))
        converted: Decimal | None = None
        if (source_unit, target_unit) in self.LINEAR_FACTORS:
            converted = value * self.LINEAR_FACTORS[(source_unit, target_unit)]
        elif source_unit in {"c", "celsius"} and target_unit in {"f", "fahrenheit"}:
            converted = value * Decimal("9") / Decimal("5") + Decimal("32")
        elif source_unit in {"f", "fahrenheit"} and target_unit in {"c", "celsius"}:
            converted = (value - Decimal("32")) * Decimal("5") / Decimal("9")
        if converted is None:
            return DeterministicSolverResult.miss("unit", f"unsupported conversion: {source_unit} to {target_unit}")
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type="unit_conversion",
            answer=converted,
            answer_text=format_decimal(converted),
            confidence=0.9,
            evidence={"source_value": str(value), "source_unit": source_unit, "target_unit": target_unit},
        )

    def _normalize_unit(self, unit: str) -> str:
        """正規化單位名稱。"""
        return self.UNIT_ALIASES.get(unit, unit)
