from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ToolExecutionResult:
    """
    統一工具執行結果，並保留既有 ToolManager dict 欄位。

    Args:
        - ok: 工具是否成功完成可用工作。
        - tool_name: 工具名稱。
        - status: success、partial、retryable_failure、unsupported、fatal、
          already_available 或 duplicate_blocked。
        - output_text: 可提供給 Agent 的文字結果。
        - raw_result: 工具原始回傳值。
        - error_code: 穩定的錯誤代碼。
        - error_message: 人類可讀錯誤。
        - retryable: 是否可在修改參數後重試。
        - retry_hint: 下一步建議。
        - evidence_valid: 結果是否包含可支持答案的證據。

    Returns:
        - ToolExecutionResult: 標準化工具狀態。
    """

    ok: bool
    tool_name: str
    status: str
    output_text: str = ""
    raw_result: Any = None
    error_code: str = ""
    error_message: str = ""
    retryable: bool = False
    retry_hint: str = ""
    evidence_valid: bool = False
    cache_hit: bool = False
    duplicate_request: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["error"] = self.error_message or None
        return result


def failure_result(
    tool_name: str,
    *,
    status: str,
    error_code: str,
    error_message: str,
    retryable: bool = False,
    retry_hint: str = "",
    raw_result: Any = None,
) -> dict[str, Any]:
    return ToolExecutionResult(
        ok=False,
        tool_name=tool_name,
        status=status,
        raw_result=raw_result,
        error_code=error_code,
        error_message=error_message,
        retryable=retryable,
        retry_hint=retry_hint,
        evidence_valid=False,
    ).to_dict()


__all__ = ["ToolExecutionResult", "failure_result"]
