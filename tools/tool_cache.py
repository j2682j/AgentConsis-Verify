from __future__ import annotations

import json
from threading import Event, Lock
from typing import Any

from .tool_result import failure_result


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
        self._inflight: dict[str, Event] = {}
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
            inflight = self._inflight.get(key)
            if cached is None and inflight is None:
                inflight = Event()
                self._inflight[key] = inflight
                is_owner = True
            else:
                is_owner = False
        if cached is not None:
            return self._duplicate_result(cached)
        if not is_owner:
            assert inflight is not None
            inflight.wait()
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                return self._duplicate_result(cached)

        try:
            if tool_manager is None or not hasattr(tool_manager, "execute_tool"):
                result = failure_result(
                    tool_name,
                    status="fatal",
                    error_code="tool_manager_unavailable",
                    error_message="tool_manager with execute_tool is not available",
                )
            else:
                result = dict(
                    tool_manager.execute_tool(
                        tool_name,
                        tool_args,
                        agent_id=agent_id,
                        stage=stage,
                    )
                )
        except Exception as exc:
            result = failure_result(
                tool_name,
                status="retryable_failure",
                error_code="tool_manager_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                retryable=True,
                retry_hint="Change the arguments or choose another tool before retrying.",
            )
        finally:
            result["cache_hit"] = False
            result["duplicate_request"] = False

        with self._lock:
            self._cache[key] = dict(result)
            event = self._inflight.pop(key, None)
            if event is not None:
                event.set()
        return result

    def _duplicate_result(self, cached: dict[str, Any]) -> dict[str, Any]:
        result = dict(cached)
        result["cache_hit"] = True
        result["duplicate_request"] = True
        if cached.get("ok") and cached.get("evidence_valid"):
            result["status"] = "already_available"
            result["retryable"] = False
            result["retry_hint"] = "Use the existing tool result; do not request it again."
            return result

        previous_error = str(
            cached.get("error_message") or cached.get("error") or "previous request failed"
        )
        result.update(
            {
                "ok": False,
                "status": "duplicate_blocked",
                "error_code": "duplicate_failed_request",
                "error_message": previous_error,
                "error": previous_error,
                "retryable": False,
                "retry_hint": "Change the arguments or choose another tool capability.",
                "evidence_valid": False,
            }
        )
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
