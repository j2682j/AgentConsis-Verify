from __future__ import annotations

from ..config import AgentEvidencePacket, FactCard, VerifiedCandidate


class EvidenceCompressor:
    """
    Compress verified search evidence into a small packet for SLM prompts.

    Args:
        - None.

    Returns:
        - EvidenceCompressor: Compressor for verified candidates and fact cards.
    """

    def compress(
        self,
        *,
        question: str,
        answer_type: str,
        verified_candidates: list[VerifiedCandidate],
        fact_cards: list[FactCard],
        max_candidates: int = 3,
        max_facts: int = 6,
        max_facts_per_candidate: int = 2,
    ) -> AgentEvidencePacket:
        """
        Select the highest-value candidates and facts for agent consumption.

        Args:
            - question: Original task question.
            - answer_type: Expected answer type.
            - verified_candidates: Candidate answers after verification.
            - fact_cards: Fact cards linked to candidates.
            - max_candidates: Maximum candidates to keep.
            - max_facts: Maximum fact cards to keep.
            - max_facts_per_candidate: Maximum support facts per candidate.

        Returns:
            - AgentEvidencePacket: Compact evidence packet for prompt rendering.
        """
        selected_candidates = verified_candidates[:max_candidates]
        selected_ids = {candidate.candidate_id for candidate in selected_candidates}

        facts_by_candidate: dict[str, list[FactCard]] = {}
        refute_facts: list[FactCard] = []
        for fact in fact_cards:
            if fact.candidate_id not in selected_ids:
                continue
            if fact.relation == "refute":
                refute_facts.append(fact)
            facts_by_candidate.setdefault(fact.candidate_id, []).append(fact)

        selected_facts: list[FactCard] = []
        for fact in sorted(refute_facts, key=lambda item: item.confidence, reverse=True):
            self._append_unique(selected_facts, fact, max_facts)

        for candidate in selected_candidates:
            candidate_facts = facts_by_candidate.get(candidate.candidate_id, [])
            support = [
                fact for fact in candidate_facts
                if fact.relation == "support"
            ]
            neutral = [
                fact for fact in candidate_facts
                if fact.relation == "neutral"
            ]
            for fact in sorted(support, key=lambda item: item.confidence, reverse=True)[:max_facts_per_candidate]:
                self._append_unique(selected_facts, fact, max_facts)
            if not support and neutral:
                self._append_unique(
                    selected_facts,
                    sorted(neutral, key=lambda item: item.confidence, reverse=True)[0],
                    max_facts,
                )

        missing_info: list[str] = []
        risk_flags: list[str] = []
        if not selected_candidates:
            missing_info.append("no_candidate_answer_found")
        elif all(candidate.confidence < 0.35 for candidate in selected_candidates):
            missing_info.append("all_candidates_have_weak_support")

        for candidate in selected_candidates:
            for flag in candidate.risk_flags:
                if flag not in risk_flags:
                    risk_flags.append(flag)

        return AgentEvidencePacket(
            question=question,
            answer_type=answer_type,
            candidates=selected_candidates,
            facts=selected_facts,
            missing_info=missing_info,
            risk_flags=risk_flags,
        )

    def _append_unique(self, items: list[FactCard], fact: FactCard, max_items: int) -> None:
        if len(items) >= max_items:
            return
        if any(item.fact_id == fact.fact_id for item in items):
            return
        items.append(fact)


__all__ = ["EvidenceCompressor"]

