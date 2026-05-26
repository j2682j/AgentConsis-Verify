from __future__ import annotations

import json
from threading import Lock
from typing import Any


class ToolCache:
    """
    提供單一任務內的工具結果快取，避免 Stage1 重複執行相同工具參數。

    Args:
        - 無。

    Returns:
        - ToolCache: 可依 tool_name 與 tool_args 快取工具結果的物件。
    """

    def __init__(self) -> None:
        """
        初始化快取儲存區與執行緒鎖。

        Args:
            - 無。

        Returns:
            - None。
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def get_or_execute(
        self,
        *,
        tool_manager: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        agent_id: str,
        stage: str,
    ) -> dict[str, Any]:
        """
        依工具名稱與參數查詢快取，未命中時透過 ToolManager 執行工具。

        Args:
            - tool_manager: 具有 execute_tool 方法的工具管理器。
            - tool_name: 要執行或查詢的工具名稱。
            - tool_args: 工具參數。
            - agent_id: 發起工具呼叫的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - dict[str, Any]: 工具結果，並包含 cache_hit 欄位。
        """
        key = self._cache_key(tool_name, tool_args)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            return result

        if tool_manager is None or not hasattr(tool_manager, "execute_tool"):
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "raw_result": None,
                "error": "tool_manager with execute_tool is not available",
                "cache_hit": False,
            }
        else:
            result = dict(
                tool_manager.execute_tool(
                    tool_name,
                    tool_args,
                    agent_id=agent_id,
                    stage=stage,
                )
            )
            result["cache_hit"] = False

        with self._lock:
            self._cache[key] = dict(result)
        return result

    def _cache_key(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """
        建立穩定的工具快取 key。

        Args:
            - tool_name: 工具名稱。
            - tool_args: 工具參數。

        Returns:
            - str: JSON 序列化後的快取 key。
        """
        return json.dumps(
            {"tool_name": tool_name, "tool_args": tool_args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


__all__ = ["ToolCache"]
