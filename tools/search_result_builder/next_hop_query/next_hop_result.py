from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NextHopQueryResult:
    """
    保存下一跳查詢組合結果與追蹤資訊。

    Args:
        - query: 組合後的下一跳查詢。
        - kept_question_tokens: 從原始問題保留的查詢 tokens。
        - kept_evidence_tokens: 從 bridge evidence 保留的 spans。
        - fallback_used: 是否由恢復流程產生結果。
        - metadata: 查詢組合與驗證的追蹤資訊。

    Returns:
        - NextHopQueryResult: 下一跳查詢結果。
    """

    query: str
    kept_question_tokens: list[str] = field(default_factory=list)
    kept_evidence_tokens: list[str] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = ["NextHopQueryResult"]
