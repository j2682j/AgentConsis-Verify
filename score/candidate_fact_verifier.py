from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable

from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore
from utils.network_utils import normalize_for_exact


@dataclass(frozen=True)
class CandidateFactVerification:
    """保存候選答案相對於任務事實的類別式驗證結果。"""

    status: str
    supporting_fact_ids: list[str] = field(default_factory=list)
    contradicting_fact_ids: list[str] = field(default_factory=list)
    derivation_chain_ids: list[str] = field(default_factory=list)
    support_kind: str = ""
    reason: str = ""
    answer_requirement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateFactVerifier:
    """以直接或推導事實驗證候選答案，不計算加權分數。"""

    def __init__(
        self,
        *,
        equivalence_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.equivalence_fn = equivalence_fn or self._equivalent

    def verify(
        self,
        *,
        candidate_answer: str,
        fact_store: TaskFactStore,
        answer_requirement: str = "",
    ) -> CandidateFactVerification:
        candidate = str(candidate_answer or "").strip()
        if not candidate:
            return CandidateFactVerification(
                status="unknown",
                reason="empty_candidate",
                answer_requirement=answer_requirement,
            )
        answer_facts = fact_store.verifiable_answer_facts()
        negative_subject_support = [
            fact
            for fact in answer_facts
            if (
                fact.polarity == "negative"
                and self.equivalence_fn(candidate, fact.subject)
            )
        ]
        if negative_subject_support:
            return CandidateFactVerification(
                status="supported",
                supporting_fact_ids=[fact.fact_id for fact in negative_subject_support],
                derivation_chain_ids=self._parent_ids(
                    negative_subject_support,
                    fact_store,
                ),
                support_kind=(
                    "derived"
                    if all(fact.parent_fact_ids for fact in negative_subject_support)
                    else "direct"
                ),
                reason="candidate_subject_satisfies_grounded_negative_condition",
                answer_requirement=answer_requirement,
            )
        negative = [
            fact
            for fact in answer_facts
            if fact.polarity == "negative" and self.equivalence_fn(candidate, fact.object)
        ]
        if negative:
            return CandidateFactVerification(
                status="contradicted",
                contradicting_fact_ids=[fact.fact_id for fact in negative],
                derivation_chain_ids=self._parent_ids(negative, fact_store),
                reason="candidate_matches_negative_answer_fact",
                answer_requirement=answer_requirement,
            )
        positive = [
            fact
            for fact in answer_facts
            if fact.polarity == "positive" and self.equivalence_fn(candidate, fact.object)
        ]
        if positive:
            direct = [fact for fact in positive if not fact.parent_fact_ids]
            derived = [fact for fact in positive if fact.parent_fact_ids]
            selected = direct or derived
            return CandidateFactVerification(
                status="supported",
                supporting_fact_ids=[fact.fact_id for fact in selected],
                derivation_chain_ids=self._parent_ids(selected, fact_store),
                support_kind="direct" if direct else "derived",
                reason=(
                    "candidate_matches_grounded_direct_fact"
                    if direct
                    else "candidate_matches_grounded_derived_fact"
                ),
                answer_requirement=answer_requirement,
            )
        return CandidateFactVerification(
            status="unknown",
            reason="no_answer_bound_fact_matches_candidate",
            answer_requirement=answer_requirement,
        )

    def _parent_ids(
        self,
        facts: list[EvidenceFact],
        store: TaskFactStore,
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        def visit(fact: EvidenceFact) -> None:
            for parent_id in fact.parent_fact_ids:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                result.append(parent_id)
                parent = store.get(parent_id)
                if parent is not None:
                    visit(parent)

        for fact in facts:
            visit(fact)
        return result

    @staticmethod
    def _equivalent(first: str, second: str) -> bool:
        first_key = normalize_for_exact(first)
        second_key = normalize_for_exact(second)
        if not first_key or not second_key:
            return False
        if first_key == second_key:
            return True
        first_number = CandidateFactVerifier._number(first_key)
        second_number = CandidateFactVerifier._number(second_key)
        return (
            first_number is not None
            and second_number is not None
            and first_number == second_number
        )

    @staticmethod
    def _number(value: str) -> Decimal | None:
        compact = re.sub(r"[,%$£€\s]", "", value)
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
            return None
        try:
            return Decimal(compact)
        except InvalidOperation:
            return None


__all__ = ["CandidateFactVerification", "CandidateFactVerifier"]
