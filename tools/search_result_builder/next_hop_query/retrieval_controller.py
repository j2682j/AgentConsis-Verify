from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CandidateAnswer, EvidenceItem


@dataclass
class RetrievalDecision:
    """
    儲存 EfficientRAG retrieval controller 對目前 evidence 是否足夠的判斷。

    Args:
        - need_next_hop: 是否需要執行下一跳 search。
        - reason: 判斷原因。
        - confidence: 目前 retrieval 足夠性的信心分數。
        - missing_info: 還缺少的資訊描述。

    Returns:
        - RetrievalDecision: 下一跳搜尋決策。
    """

    need_next_hop: bool
    reason: str
    confidence: float = 0.0
    missing_info: list[str] = field(default_factory=list)


class RetrievalController:
    """
    根據 SEER cleaned evidence 與 candidate answers 判斷是否需要 next-hop retrieval。

    Args:
        - min_evidence_quality: 視為足夠 evidence 的最低品質分數。
        - min_candidate_support: 視為候選答案有支撐的最低 support count。

    Returns:
        - RetrievalController: EfficientRAG retrieval sufficiency controller。
    """

    def __init__(
        self,
        *,
        min_evidence_quality: float = 0.55,
        min_candidate_support: int = 1,
    ) -> None:
        self.min_evidence_quality = min_evidence_quality
        self.min_candidate_support = min_candidate_support

    def assess(
        self,
        *,
        evidence_items: list[EvidenceItem],
        candidates: list[CandidateAnswer],
    ) -> RetrievalDecision:
        """
        評估目前 retrieval 是否足夠，或是否要進行下一跳 search。

        Args:
            - evidence_items: SEER cleaning 後的 evidence。
            - candidates: 從 evidence 抽出的候選答案。

        Returns:
            - RetrievalDecision: 是否需要 next-hop retrieval 的決策。
        """
        if not evidence_items:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_evidence",
                confidence=0.0,
                missing_info=["evidence"],
            )

        best_evidence_quality = max(
            max(item.helpfulness_score, item.evidence_quality)
            for item in evidence_items
        )
        supported_candidates = [
            candidate
            for candidate in candidates
            if candidate.support_count >= self.min_candidate_support
        ]

        if best_evidence_quality < self.min_evidence_quality:
            return RetrievalDecision(
                need_next_hop=True,
                reason="low_evidence_quality",
                confidence=best_evidence_quality,
                missing_info=["high_quality_evidence"],
            )

        if candidates and not supported_candidates:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_supported_candidate",
                confidence=best_evidence_quality,
                missing_info=["candidate_answer"],
            )

        return RetrievalDecision(
            need_next_hop=False,
            reason="sufficient_evidence",
            confidence=best_evidence_quality,
            missing_info=[],
        )
