from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttachmentStrategy:
    """
    描述附件中需要取得的資訊與唯一的下一個處理動作。

    Args:
     - information_need: 原始問題要求從附件取得的資訊。
     - required_handler: 下一個要執行的 handler 名稱或角色。
     - required_inputs: handler 執行前必須存在的輸入。
     - expected_answer: 以自然語言描述的答案需求。
     - needs_search: 附件不足時是否需要外部搜尋。
     - missing_inputs: 目前仍缺少的輸入。

    Returns:
     - AttachmentStrategy: 可供 executor 驗證與執行的精簡策略。
    """

    information_need: str = ""
    required_handler: str = ""
    required_inputs: list[str] = field(default_factory=list)
    expected_answer: str = ""
    needs_search: bool = False
    missing_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttachmentStrategyResult:
    """
    保存附件解析、策略、handler 結果與一次修正的完整狀態。

    Args:
     - strategy: 初始附件策略。
     - revised_strategy: handler 失敗後的一次修正策略。
     - final_answer_candidate: 保留欄位，目前 reviewer 不直接產生答案。
     - attachment_context: 通用解析後的附件文字。
     - solver_context: 通過信任驗證的 handler 證據。
     - attachment_profile: 附件能力與結構摘要。
     - parsed_payload: 後端可重用的解析內容與來源資訊。
     - tool_usage: 解析、規劃、handler 與驗證紀錄。
     - metadata: 路由與執行摘要。

    Returns:
     - AttachmentStrategyResult: 交由 EvidenceRunner 使用的附件結果。
    """

    strategy: AttachmentStrategy = field(default_factory=AttachmentStrategy)
    revised_strategy: AttachmentStrategy | None = None
    final_answer_candidate: str = ""
    attachment_context: str = ""
    solver_context: str = ""
    attachment_profile: dict[str, Any] = field(default_factory=dict)
    parsed_payload: dict[str, Any] = field(default_factory=dict)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    reader_status: str = "failed"
    strategy_status: str = "not_required"
    handler_status: str = "not_required"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        parsed_payload = dict(self.parsed_payload or {})
        content = str(parsed_payload.pop("content", "") or "")
        typed_counts = {
            "text_block_count": len(parsed_payload.get("text_blocks") or []),
            "table_count": len(parsed_payload.get("tables") or []),
            "list_count": len(parsed_payload.get("lists") or []),
            "coordinate_count": len(parsed_payload.get("coordinates") or []),
            "relation_count": len(parsed_payload.get("relations") or []),
            "visual_block_count": len(parsed_payload.get("visual_blocks") or []),
        }
        parsed_payload_summary = {
            "schema_version": str(parsed_payload.get("schema_version") or "1.0"),
            "reader": str(parsed_payload.get("reader") or ""),
            "reader_metadata": dict(parsed_payload.get("reader_metadata") or {}),
            "native_metadata": dict(parsed_payload.get("native_metadata") or {}),
            "provenance": dict(parsed_payload.get("provenance") or {}),
            **typed_counts,
            "content_character_count": len(content),
        }
        return {
            "strategy": self.strategy.to_dict(),
            "revised_strategy": (
                self.revised_strategy.to_dict() if self.revised_strategy else None
            ),
            "final_answer_candidate": self.final_answer_candidate,
            "attachment_context": self.attachment_context,
            "solver_context": self.solver_context,
            "attachment_profile": dict(self.attachment_profile or {}),
            "parsed_payload": parsed_payload_summary,
            "tool_usage": list(self.tool_usage or []),
            "reader_status": self.reader_status,
            "strategy_status": self.strategy_status,
            "handler_status": self.handler_status,
            "metadata": dict(self.metadata or {}),
        }


__all__ = ["AttachmentStrategy", "AttachmentStrategyResult"]
