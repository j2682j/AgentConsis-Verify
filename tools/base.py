from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameter:
    """
    描述工具可接受的單一參數欄位。

    Args:
        - name: 參數名稱。
        - type: 參數型別描述。
        - description: 參數用途說明。
        - required: 是否為必填參數。

    Returns:
        - ToolParameter: 工具參數 metadata。
    """

    name: str
    type: str
    description: str = ""
    required: bool = False


class Tool:
    """
    定義所有本地工具的共同介面。

    Args:
        - name: 工具名稱。
        - description: 工具用途描述。

    Returns:
        - Tool: 可被 ToolManager 註冊與執行的工具基底物件。
    """

    def __init__(self, *, name: str, description: str = "") -> None:
        """
        ??????????????
        
        Args:
            - ????????????
        
        Returns:
            - None?
        """
        self.name = name
        self.description = description

    def run(self, parameters: dict[str, Any]) -> Any:
        """
        執行工具主要功能，子類別必須實作。

        Args:
            - parameters: 工具執行所需參數。

        Returns:
            - Any: 工具執行結果。
        """
        raise NotImplementedError

    def get_parameters(self) -> list[ToolParameter]:
        """
        回傳工具支援的參數 schema。

        Args:
            - 無。

        Returns:
            - list[ToolParameter]: 工具參數定義清單。
        """
        return []
