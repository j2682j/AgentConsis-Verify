from __future__ import annotations

import json
from typing import Any

from .base import Tool, ToolParameter
from .deterministic_solver import DeterministicSolver


class DeterministicSolverTool(Tool):
    """
    將 DeterministicSolver 包裝成 ToolManager 可執行的工具。

    Args:
        - 無。

    Returns:
        - DeterministicSolverTool: 可解決封閉型計算、字串、列表、表格與單位題的工具。
    """

    def __init__(self) -> None:
        """
        初始化 deterministic_solver 工具與內部 solver。

        Args:
            - 無。

        Returns:
            - None。
        """
        super().__init__(
            name="deterministic_solver",
            description="Solve deterministic closed-world math, string, list, table, and unit tasks.",
            capabilities={
                "list.count",
                "list.nth",
                "list.sort",
                "conversion.sexagesimal",
                "geometry.coordinate_distance",
                "graph.shortest_path",
                "graph.station_count",
                "graph.traversal",
                "grid.word_search",
                "math.arithmetic",
                "math.statistics",
                "string.count",
                "string.transform",
                "table.cell_lookup",
                "table.filter",
                "table.statistics",
                "unit.linear_conversion",
                "unit.temperature_conversion",
            },
            deterministic=True,
        )
        self.solver = DeterministicSolver()

    def run(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        執行 deterministic solver 並回傳標準 dict 結果。

        Args:
            - parameters: 包含 input/question、attachment_context、table_data 等欄位的參數。

        Returns:
            - dict[str, Any]: DeterministicSolverResult.to_dict() 的結果。
        """
        result = self.solver.solve(
            str(parameters.get("input") or parameters.get("question") or ""),
            attachment_context=parameters.get("attachment_context"),
            table_data=parameters.get("table_data"),
            best_candidate=parameters.get("best_candidate"),
        )
        return result.to_dict()

    def get_parameters(self) -> list[ToolParameter]:
        """
        回傳 deterministic_solver 工具參數 schema。

        Args:
            - 無。

        Returns:
            - list[ToolParameter]: deterministic_solver 需要的 input 參數定義。
        """
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Question or deterministic task to solve.",
                required=True,
            )
        ]


def solve_deterministic(question: str, **kwargs: Any) -> str:
    """
    以便利函式形式呼叫 DeterministicSolverTool。

    Args:
        - question: 要嘗試 deterministic 解題的問題。
        - kwargs: 額外傳入 tool 的 attachment_context、table_data 等參數。

    Returns:
        - str: JSON 字串格式的 solver 結果。
    """
    tool = DeterministicSolverTool()
    return json.dumps(tool.run({"input": question, **kwargs}), ensure_ascii=False)
