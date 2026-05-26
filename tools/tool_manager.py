from __future__ import annotations

from typing import Any

from .registry import ToolRegistry


class ToolManager:
    """
    管理本地工具註冊、啟用狀態、執行結果正規化與工具使用 trace。

    Args:
        - 無。

    Returns:
        - ToolManager: 可供 EvidenceRunner 與 Stage1TrajectoryRunner 呼叫的工具管理器。
    """

    def __init__(self) -> None:
        """
        初始化工具 registry、啟用工具集合與 trace 紀錄，並註冊預設工具。

        Args:
            - 無。

        Returns:
            - None。
        """
        self.registry = ToolRegistry()
        self.tools: dict[str, Any] = {}
        self.enabled_tools: set[str] = set()
        self.tool_traces: list[dict[str, Any]] = []
        self.register_default_tools()

    def register_tool(self, tool: Any, auto_expand: bool = True) -> None:
        """
        註冊單一工具到 ToolManager 與底層 ToolRegistry。

        Args:
            - tool: 具有 name 與 run 方法的工具物件。
            - auto_expand: 是否允許 registry 展開 expandable tool。

        Returns:
            - None。
        """
        self.tools[tool.name] = tool
        self.registry.register_tool(tool, auto_expand=auto_expand)

    def register_default_tools(self) -> None:
        """
        註冊系統預設工具，例如 calculator、deterministic_solver 與 search。

        Args:
            - 無。

        Returns:
            - None。
        """
        from .calculator import CalculatorTool
        from .deterministic_solver_tool import DeterministicSolverTool

        calculator = CalculatorTool()
        self.register_tool(calculator)
        self.enabled_tools.add(calculator.name)

        deterministic_solver = DeterministicSolverTool()
        self.register_tool(deterministic_solver)
        self.enabled_tools.add(deterministic_solver.name)

        try:
            from .search_tool import SearchTool

            search_tool = SearchTool()
            self.register_tool(search_tool)
            self.enabled_tools.add(search_tool.name)
        except Exception as exc:
            print(f"[WARN] SearchTool initialization failed: {exc}")

    def set_enabled_tools(self, tool_names: list[str] | set[str]) -> None:
        """
        設定目前允許執行的工具名稱集合。

        Args:
            - tool_names: 要啟用的工具名稱清單或集合。

        Returns:
            - None。
        """
        self.enabled_tools = set(tool_names)

    def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        agent_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        """
        執行指定工具並回傳標準化工具結果，同時記錄 tool trace。

        Args:
            - tool_name: 要執行的工具名稱。
            - parameters: 傳給工具的參數。
            - agent_id: 發起工具呼叫的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - dict[str, Any]: 包含 ok、tool_name、output_text、raw_result 與 error 的工具結果。
        """
        if tool_name not in self.enabled_tools:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "raw_result": None,
                "error": f"tool '{tool_name}' is not enabled",
            }
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        tool = self.tools.get(tool_name) or self.registry.get_tool(tool_name)
        if tool is None:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "raw_result": None,
                "error": f"tool '{tool_name}' not found",
            }
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        try:
            raw = tool.run(parameters)
            result = self.normalize_result(tool_name, raw)
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }

        self._record_trace(tool_name, parameters, result, agent_id, stage)
        return result

    def normalize_result(self, tool_name: str, raw_result: Any) -> dict[str, Any]:
        """
        將工具原始輸出轉成系統統一的工具結果格式。

        Args:
            - tool_name: 工具名稱。
            - raw_result: 工具原始回傳值。

        Returns:
            - dict[str, Any]: 標準化後的工具結果。
        """
        if isinstance(raw_result, dict):
            return {
                "ok": True,
                "tool_name": tool_name,
                "output_text": str(raw_result),
                "raw_result": raw_result,
                "error": None,
            }
        return {
            "ok": True,
            "tool_name": tool_name,
            "output_text": str(raw_result),
            "raw_result": raw_result,
            "error": None,
        }

    def _record_trace(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        agent_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        """
        記錄一次工具呼叫的輸入、輸出、Agent 與階段資訊。

        Args:
            - tool_name: 工具名稱。
            - parameters: 工具呼叫參數。
            - result: 標準化工具結果。
            - agent_id: 發起工具呼叫的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - None。
        """
        self.tool_traces.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "agent_id": agent_id,
                "stage": stage,
                "ok": result.get("ok", False),
                "output_text": result.get("output_text", ""),
                "error": result.get("error"),
            }
        )
