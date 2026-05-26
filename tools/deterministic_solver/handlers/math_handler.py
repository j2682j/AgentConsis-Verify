"""數學 deterministic handler。"""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from statistics import median
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, extract_numbers, format_decimal, lower_text


class MathHandler:
    """處理封閉世界的簡單數學題。"""

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
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
        numbers = extract_numbers(text)

        percent = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%\s+of\s+([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", lowered)
        if percent:
            value = Decimal(percent.group(2).replace(",", "")) * Decimal(percent.group(1)) / Decimal("100")
            return self._result("math_percent", value, {"operation": "percent_of"})

        if any(term in lowered for term in ["average", "mean"]) and len(numbers) >= 2:
            return self._result("math_average", sum(numbers) / Decimal(len(numbers)), {"numbers": [str(item) for item in numbers]})
        if "median" in lowered and len(numbers) >= 2:
            return self._result("math_median", Decimal(str(median(numbers))), {"numbers": [str(item) for item in numbers]})
        if any(term in lowered for term in ["sum", "total", "add"]) and len(numbers) >= 2:
            return self._result("math_sum", sum(numbers), {"numbers": [str(item) for item in numbers]})
        if any(term in lowered for term in ["largest", "maximum", "max"]) and numbers:
            return self._result("math_max", max(numbers), {"numbers": [str(item) for item in numbers]})
        if any(term in lowered for term in ["smallest", "minimum", "min"]) and numbers:
            return self._result("math_min", min(numbers), {"numbers": [str(item) for item in numbers]})

        expression = self._extract_expression(text)
        if expression:
            value = Decimal(str(self._safe_eval(expression)))
            return self._result("math_expression", self._apply_rounding(value, lowered), {"expression": expression})

        return DeterministicSolverResult.miss("math")

    def _result(self, task_type: str, value: Decimal, evidence: dict[str, Any]) -> DeterministicSolverResult:
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
            answer=value,
            answer_text=format_decimal(value),
            confidence=0.9,
            evidence=evidence,
        )

    def _extract_expression(self, text: str) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        candidate = re.sub(r"[^0-9+\-*/().^ ]", " ", text).replace("^", "**")
        candidate = re.sub(r"\s+", "", candidate)
        if not re.search(r"\d", candidate) or not re.search(r"[+\-*/]", candidate):
            return ""
        return candidate

    def _safe_eval(self, expression: str) -> float:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        node = ast.parse(expression, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPERATORS:
            return self.OPERATORS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.OPERATORS:
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        raise ValueError("不支援的算式")

    def _apply_rounding(self, value: Decimal, lowered_question: str) -> Decimal:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        match = re.search(r"round(?:ed)?\s+to\s+(\d+)\s+decimal", lowered_question)
        if not match:
            return value
        places = int(match.group(1))
        quantum = Decimal("1") if places <= 0 else Decimal("1").scaleb(-places)
        return value.quantize(quantum, rounding=ROUND_HALF_UP)
