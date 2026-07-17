from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RelationEvidence:
    """
    保存由原始 passage 綁定到 active relation goal 的關係證據。

    Args:
     - goal_id: 被此證據解析的 relation goal。
     - subject: 在上下文中被確認的關係主體。
     - relation: 與 goal 對齊的自然語言關係。
     - object: 從 grounded span 取得的 relation object。
     - context: 同時包含 subject 與 object 的原文上下文。
     - document_id: 原始 corpus passage id。

    Returns:
     - RelationEvidence: 可供 goal resolver 更新狀態的證據記錄。
    """

    goal_id: str
    subject: str
    relation: str
    object: str
    context: str
    document_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["RelationEvidence"]
