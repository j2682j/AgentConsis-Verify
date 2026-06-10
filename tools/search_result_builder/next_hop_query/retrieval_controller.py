from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CandidateAnswer, EvidenceItem


@dataclass
class RetrievalDecision:
    """
    保存 retrieval control 對是否需要 next-hop search 的判斷。

    Args:
        - need_next_hop: 是否需要再做下一跳搜尋。
        - reason: 判斷原因。
        - confidence: retrieval control 的信心；目前有 evidence 時為 1。
        - missing_info: 缺少的資訊類型。

    Returns:
        - RetrievalDecision: next-hop search 決策。
    """

    need_next_hop: bool
    reason: str
    confidence: float = 0.0
    missing_info: list[str] = field(default_factory=list)


class RetrievalController:
    """
    根據 evidence 是否存在，判斷是否需要 next-hop retrieval。

    Args:
        - min_candidate_support: 候選答案需要的最低支撐數。

    Returns:
        - RetrievalController: retrieval sufficiency controller。
    """

    def __init__(self, *, min_candidate_support: int = 1) -> None:
        self.min_candidate_support = min_candidate_support

    def assess(
        self,
        *,
        evidence_items: list[EvidenceItem],
        candidates: list[CandidateAnswer],
    ) -> RetrievalDecision:
        """
        判斷目前 evidence 是否足夠進入後續流程。

        Args:
            - evidence_items: SourceAnalysis 輸出的 evidence。
            - candidates: 可選候選答案，目前通常為空。

        Returns:
            - RetrievalDecision: 是否需要 next-hop search。
        """
        if not evidence_items:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_evidence",
                confidence=0.0,
                missing_info=["evidence"],
            )

        supported_candidates = [
            candidate
            for candidate in candidates
            if candidate.support_count >= self.min_candidate_support
        ]
        if candidates and not supported_candidates:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_supported_candidate",
                confidence=1.0,
                missing_info=["candidate_answer"],
            )

        return RetrievalDecision(
            need_next_hop=False,
            reason="sufficient_evidence",
            confidence=1.0,
            missing_info=[],
        )


__all__ = ["RetrievalController", "RetrievalDecision"]
