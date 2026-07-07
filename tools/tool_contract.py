from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .tool_result import ToolExecutionResult


@dataclass(frozen=True)
class ToolCapabilitySpec:
    """
    描述工具可以處理的能力、輸入與輸出。

    Args:
     - tool_name: 工具名稱。
     - capabilities: 工具宣告支援的能力。
     - input_types: 工具可接受的輸入型態。
     - output_types: 工具可產生的輸出型態。
     - required_inputs: 執行工具所需的必要輸入。
     - optional_inputs: 工具可接受的額外輸入。
     - priority: capability matching 時的排序優先度。

    Returns:
     - ToolCapabilitySpec: 給 registry 使用的工具能力宣告。

    """

    tool_name: str
    capabilities: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ToolCapabilitySpec", "ToolExecutionResult"]
