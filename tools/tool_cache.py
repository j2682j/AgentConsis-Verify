from __future__ import annotations

import json
import re
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
        self._request_count = 0
        self._execution_count = 0
        self._cache_hit_count = 0
        self._duplicate_request_count = 0
        self._retryable_failure_count = 0

    def get_or_execute(
        self,
        *,
        tool_manager: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        agent_id: str,
        stage: str,
        runtime_context: dict[str, Any] | None = None,
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
        key = self._cache_key(
            tool_name,
            tool_args,
            runtime_context=runtime_context,
        )
        with self._lock:
            self._request_count += 1
            cached = self._cache.get(key)
            inflight = self._inflight.get(key)
            if cached is None and inflight is None:
                inflight = Event()
                self._inflight[key] = inflight
                is_owner = True
            else:
                is_owner = False
        if cached is not None:
            self._record_cache_hit()
            return self._duplicate_result(cached)
        if not is_owner:
            assert inflight is not None
            inflight.wait()
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                self._record_cache_hit()
                return self._duplicate_result(cached)

        try:
            with self._lock:
                self._execution_count += 1
            if tool_manager is None or not hasattr(tool_manager, "execute_tool"):
                result = failure_result(
                    tool_name,
                    status="fatal",
                    error_code="tool_manager_unavailable",
                    error_message="tool_manager with execute_tool is not available",
                )
            else:
                execute_kwargs = {
                    "agent_id": agent_id,
                    "stage": stage,
                }
                if runtime_context is not None:
                    execute_kwargs["runtime_context"] = runtime_context
                result = dict(
                    tool_manager.execute_tool(
                        tool_name,
                        tool_args,
                        **execute_kwargs,
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
            if bool(result.get("retryable")):
                self._retryable_failure_count += 1
            else:
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

    def snapshot(self) -> dict[str, int]:
        """Return task-local cache counters for reports and regression tests."""

        with self._lock:
            return {
                "request_count": self._request_count,
                "execution_count": self._execution_count,
                "cache_hit_count": self._cache_hit_count,
                "duplicate_request_count": self._duplicate_request_count,
                "retryable_failure_count": self._retryable_failure_count,
                "cached_entry_count": len(self._cache),
                "inflight_count": len(self._inflight),
            }

    def _record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hit_count += 1
            self._duplicate_request_count += 1

    def _cache_key(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """
        建立穩定的工具快取 key。

        Args:
            - tool_name: 工具名稱。
            - tool_args: 工具參數。

        Returns:
            - str: JSON 序列化後的快取 key。
        """
        attachment_fingerprint = ""
        if runtime_context:
            workspace = runtime_context.get("attachment_workspace")
            attachment_fingerprint = str(
                getattr(workspace, "fingerprint", "") or ""
            ).strip()
        return json.dumps(
            {
                "tool_name": str(tool_name or "").strip().casefold(),
                "tool_args": self._canonicalize(
                    tool_args,
                    search_mode=str(tool_name or "").strip().casefold() == "search",
                ),
                "attachment_fingerprint": attachment_fingerprint,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _canonicalize(self, value: Any, *, search_mode: bool = False) -> Any:
        if isinstance(value, dict):
            return {
                str(key).strip(): self._canonicalize(
                    item,
                    search_mode=search_mode,
                )
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [
                self._canonicalize(item, search_mode=search_mode)
                for item in value
            ]
        if isinstance(value, str):
            normalized = re.sub(r"\s+", " ", value).strip()
            return normalized.casefold() if search_mode else normalized
        return value


__all__ = ["ToolCache"]
