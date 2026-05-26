"""安全數學計算工具。"""

import ast
import math
import operator
from typing import Any, Dict

from .base import Tool


class CalculatorTool(Tool):
    """
    使用 Python AST 安全計算基本數學表達式。

    Args:
        - 無。

    Returns:
        - CalculatorTool: 可被 ToolManager 執行的 python_calculator 工具。
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        """
        初始化 calculator 工具名稱與描述。

        Args:
            - 無。

        Returns:
            - None。
        """
        super().__init__(
            name="python_calculator",
            description="Evaluate safe mathematical expressions such as 2+3*4, sqrt(16), or sin(pi/2).",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行數學表達式計算。

        Args:
            - parameters: 包含 input 或 expression 的工具參數。

        Returns:
            - str: 計算結果文字；失敗時回傳錯誤訊息。
        """
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "缺少數學表達式"

        try:
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body)
            return str(result)
        except Exception as exc:
            return f"計算失敗: {exc}"

    def _eval_node(self, node):
        """
        遞迴計算 AST node，僅允許白名單運算子與函式。

        Args:
            - node: Python AST node。

        Returns:
            - Any: node 計算結果。
        """
        if isinstance(node, ast.Constant):
            return node.value
        ast_num = getattr(ast, "Num", None)
        if ast_num is not None and isinstance(node, ast_num):
            return node.n
        if isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left),
                self._eval_node(node.right),
            )
        if isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            raise ValueError(f"不支援的函式: {func_name}")
        if isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            raise ValueError(f"不支援的名稱: {node.id}")
        raise ValueError(f"不支援的語法: {type(node).__name__}")

    def get_parameters(self):
        """
        回傳 calculator 工具參數 schema。

        Args:
            - 無。

        Returns:
            - list[ToolParameter]: calculator 需要的 input 參數定義。
        """
        from .base import ToolParameter

        return [
            ToolParameter(
                name="input",
                type="string",
                description="Mathematical expression to evaluate.",
                required=True,
            )
        ]


def calculate(expression: str) -> str:
    """
    使用 CalculatorTool 計算單一數學表達式。

    Args:
        - expression: 要計算的數學表達式。

    Returns:
        - str: 計算結果文字。
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})
