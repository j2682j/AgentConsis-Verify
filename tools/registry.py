"""工具註冊表。"""

from typing import Any, Callable, Optional

from .base import Tool


class ToolRegistry:
    """
    保存可用工具與簡單函式工具，提供查詢、執行與描述輸出。

    Args:
        - 無。

    Returns:
        - ToolRegistry: 可註冊 Tool 或 function 的工具註冊表。
    """

    def __init__(self) -> None:
        """
        初始化工具與函式註冊表。

        Args:
            - 無。

        Returns:
            - None。
        """
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True) -> None:
        """
        註冊 Tool 物件，必要時展開 expandable tool。

        Args:
            - tool: 要註冊的 Tool 物件。
            - auto_expand: 是否自動展開 expandable tool 的子工具。

        Returns:
            - None。
        """
        if auto_expand and hasattr(tool, "expandable") and tool.expandable:
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        print(f"[WARN] tool '{sub_tool.name}' already registered; overwriting.")
                    self._tools[sub_tool.name] = sub_tool
                print(f"[OK] tool '{tool.name}' expanded into {len(expanded_tools)} tools.")
                return

        if tool.name in self._tools:
            print(f"[WARN] tool '{tool.name}' already registered; overwriting.")

        self._tools[tool.name] = tool
        print(f"[OK] tool '{tool.name}' registered.")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str],
    ) -> None:
        """
        註冊簡單字串輸入、字串輸出的函式工具。

        Args:
            - name: 函式工具名稱。
            - description: 函式工具用途描述。
            - func: 可執行的函式。

        Returns:
            - None。
        """
        if name in self._functions:
            print(f"[WARN] function '{name}' already registered; overwriting.")

        self._functions[name] = {
            "description": description,
            "func": func,
        }
        print(f"[OK] function '{name}' registered.")

    def unregister(self, name: str) -> None:
        """
        移除指定工具或函式工具。

        Args:
            - name: 要移除的工具或函式名稱。

        Returns:
            - None。
        """
        if name in self._tools:
            del self._tools[name]
            print(f"[INFO] tool '{name}' unregistered.")
        elif name in self._functions:
            del self._functions[name]
            print(f"[INFO] function '{name}' unregistered.")
        else:
            print(f"[WARN] Tool or function '{name}' not found.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        依名稱取得 Tool 物件。

        Args:
            - name: 工具名稱。

        Returns:
            - Optional[Tool]: 找到時回傳工具物件，否則回傳 None。
        """
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable[[str], str]]:
        """
        依名稱取得函式工具。

        Args:
            - name: 函式工具名稱。

        Returns:
            - Optional[Callable[[str], str]]: 找到時回傳函式，否則回傳 None。
        """
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        以簡單文字輸入執行指定工具或函式工具。

        Args:
            - name: 工具或函式工具名稱。
            - input_text: 傳入工具的文字輸入。

        Returns:
            - str: 工具執行結果文字；失敗時回傳錯誤訊息。
        """
        if name in self._tools:
            tool = self._tools[name]
            try:
                return tool.run({"input": input_text})
            except Exception:
                return "工具執行失敗"

        if name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception:
                return "函式工具執行失敗"

        return f"[ERROR] Tool or function '{name}' was not found."

    def get_tools_description(self) -> str:
        """
        取得目前註冊工具與函式工具的文字描述。

        Args:
            - 無。

        Returns:
            - str: 工具描述清單；沒有工具時回傳提示文字。
        """
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "No tools registered."

    def list_tools(self) -> list[str]:
        """
        列出所有已註冊工具與函式工具名稱。

        Args:
            - 無。

        Returns:
            - list[str]: 工具名稱清單。
        """
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """
        取得所有已註冊 Tool 物件。

        Args:
            - 無。

        Returns:
            - list[Tool]: 工具物件清單。
        """
        return list(self._tools.values())

    def find_by_capability(self, capability: str) -> list[Tool]:
        required = str(capability or "").strip().lower()
        if not required:
            return []
        return [
            tool
            for tool in self._tools.values()
            if any(
                self._capability_matches(registered, required)
                for registered in getattr(tool, "capabilities", set())
            )
        ]

    def capability_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for tool in self._tools.values():
            for capability in sorted(getattr(tool, "capabilities", set())):
                index.setdefault(capability, []).append(tool.name)
        return index

    def tool_metadata(self, name: str) -> dict[str, Any]:
        tool = self.get_tool(name)
        return tool.capability_metadata() if tool is not None else {}

    def _capability_matches(self, registered: str, required: str) -> bool:
        registered_key = str(registered).strip().lower()
        if registered_key == required:
            return True
        if registered_key.endswith(".*"):
            return required.startswith(registered_key[:-1])
        return False

    def clear(self) -> None:
        """
        清空所有工具與函式工具註冊。

        Args:
            - 無。

        Returns:
            - None。
        """
        self._tools.clear()
        self._functions.clear()
        print("[OK] Global tool registry cleared.")


global_registry = ToolRegistry()
