from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable

from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore
from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)
from utils.network_utils import normalize_for_exact
from utils.canonical_answer_value import CanonicalAnswerValueParser


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
    required_relation: str = ""
    relation_mismatch_fact_ids: list[str] = field(default_factory=list)

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
        self.value_parser = CanonicalAnswerValueParser()
        self.fact_goal_binding_validator = FactGoalBindingValidator()

    def verify(
        self,
        *,
        candidate_answer: str,
        fact_store: TaskFactStore,
        answer_requirement: str = "",
        required_relation: str = "",
        required_relation_goal_id: str = "",
        answer_role: str = "",
    ) -> CandidateFactVerification:
        candidate = str(candidate_answer or "").strip()
        if not candidate:
            return CandidateFactVerification(
                status="unknown",
                reason="empty_candidate",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
            )
        unverified_derived_matches = [
            fact
            for fact in fact_store.by_role("ANSWER_SUPPORT")
            if (
                fact.parent_fact_ids
                and fact.derived_contract
                and str(
                    fact.derived_contract.get("verification_status") or ""
                ).strip() not in {"verified", "legacy_accepted"}
                and (
                    self._values_equivalent(candidate, fact.object, answer_requirement)
                    or (
                        fact.polarity == "negative"
                        and self._values_equivalent(
                            candidate,
                            fact.subject,
                            answer_requirement,
                        )
                    )
                )
            )
        ]
        answer_facts = fact_store.verifiable_answer_facts()
        boolean_negative_support = [
            fact
            for fact in answer_facts
            if (
                fact.polarity == "negative"
                and self._is_boolean_requirement(answer_requirement)
                and self._is_negative_boolean(candidate)
                and self._values_equivalent(candidate, fact.object, answer_requirement)
            )
        ]
        if boolean_negative_support:
            boolean_negative_support, relation_mismatches = self._relation_bound_facts(
                boolean_negative_support,
                required_relation=required_relation,
                required_relation_goal_id=required_relation_goal_id,
                answer_role=answer_role,
            )
        else:
            relation_mismatches = []
        if boolean_negative_support:
            return CandidateFactVerification(
                status="supported",
                supporting_fact_ids=[fact.fact_id for fact in boolean_negative_support],
                derivation_chain_ids=self._parent_ids(boolean_negative_support, fact_store),
                support_kind=(
                    "derived"
                    if all(fact.parent_fact_ids for fact in boolean_negative_support)
                    else "direct"
                ),
                reason="candidate_supported_by_negative_boolean_fact",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
            )
        negative_subject_support = [
            fact
            for fact in answer_facts
            if (
                fact.polarity == "negative"
                and self._values_equivalent(candidate, fact.subject, answer_requirement)
            )
        ]
        if negative_subject_support:
            negative_subject_support, relation_mismatches = self._relation_bound_facts(
                negative_subject_support,
                required_relation=required_relation,
                required_relation_goal_id=required_relation_goal_id,
                answer_role=answer_role,
            )
        else:
            relation_mismatches = list(relation_mismatches)
        if negative_subject_support:
            support_reason = self._negative_support_reason(negative_subject_support)
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
                reason=support_reason,
                answer_requirement=answer_requirement,
                required_relation=required_relation,
            )
        negative = [
            fact
            for fact in answer_facts
            if fact.polarity == "negative" and self._values_equivalent(candidate, fact.object, answer_requirement)
        ]
        if negative:
            negative, negative_relation_mismatches = self._relation_bound_facts(
                negative,
                required_relation=required_relation,
                required_relation_goal_id=required_relation_goal_id,
                answer_role=answer_role,
            )
            relation_mismatches.extend(negative_relation_mismatches)
        if negative:
            return CandidateFactVerification(
                status="contradicted",
                contradicting_fact_ids=[fact.fact_id for fact in negative],
                derivation_chain_ids=self._parent_ids(negative, fact_store),
                reason="candidate_matches_negative_answer_fact",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
            )
        positive = [
            fact
            for fact in answer_facts
            if fact.polarity == "positive" and self._values_equivalent(candidate, fact.object, answer_requirement)
        ]
        if positive:
            positive, positive_relation_mismatches = self._relation_bound_facts(
                positive,
                required_relation=required_relation,
                required_relation_goal_id=required_relation_goal_id,
                answer_role=answer_role,
            )
            relation_mismatches.extend(positive_relation_mismatches)
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
                required_relation=required_relation,
            )
        if relation_mismatches:
            return CandidateFactVerification(
                status="unknown",
                reason="candidate_only_matches_wrong_relation",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
                relation_mismatch_fact_ids=self._dedupe_ids(relation_mismatches),
            )
        if any(check.status == "unknown" for check in fact_store.absence_checks()):
            return CandidateFactVerification(
                status="unknown",
                reason="absence_unverifiable_in_incomplete_scope",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
            )
        if unverified_derived_matches:
            return CandidateFactVerification(
                status="unknown",
                reason="candidate_matches_unverified_derived_fact",
                answer_requirement=answer_requirement,
                required_relation=required_relation,
                relation_mismatch_fact_ids=[
                    fact.fact_id for fact in unverified_derived_matches
                ],
            )
        return CandidateFactVerification(
            status="unknown",
            reason="no_answer_bound_fact_matches_candidate",
            answer_requirement=answer_requirement,
            required_relation=required_relation,
        )

    def _relation_bound_facts(
        self,
        facts: list[EvidenceFact],
        *,
        required_relation: str,
        required_relation_goal_id: str,
        answer_role: str,
    ) -> tuple[list[EvidenceFact], list[str]]:
        if not str(required_relation or "").strip():
            return facts, []
        goal = {
            "goal_id": required_relation_goal_id,
            "subject": "",
            "relation": required_relation,
            "target": answer_role or "answer",
        }
        bound: list[EvidenceFact] = []
        mismatches: list[str] = []
        for fact in facts:
            binding = self.fact_goal_binding_validator.validate(
                fact=fact,
                goal=goal,
                answer_role=answer_role,
            )
            if binding.bound:
                bound.append(fact)
            else:
                mismatches.append(fact.fact_id)
        return bound, mismatches

    @staticmethod
    def _dedupe_ids(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _values_equivalent(self, first: str, second: str, requirement: str) -> bool:
        if self.equivalence_fn(first, second):
            return True
        return self.value_parser.equivalent(
            first,
            second,
            answer_requirement=requirement,
        )

    @staticmethod
    def _negative_support_reason(facts: list[EvidenceFact]) -> str:
        types = {fact.qualifiers.get("negation_type", "") for fact in facts}
        if "explicit_negative" in types:
            return "candidate_supported_by_explicit_negative_fact"
        if types & {"closed_world_absence", "closed_world_set_difference"}:
            return "candidate_supported_by_closed_world_absence"
        return "candidate_subject_satisfies_grounded_negative_condition"

    @staticmethod
    def _is_negative_boolean(value: str) -> bool:
        return normalize_for_exact(value) in {"no", "false"}

    @staticmethod
    def _is_boolean_requirement(requirement: str) -> bool:
        text = str(requirement or "").strip().casefold()
        if re.search(r"\b(?:yes\s*(?:or|/)\s*no|whether)\b", text):
            return True
        return bool(
            re.search(
                r"(?:^|[.!?]\s+)(?:can|could|do|does|did|is|are|was|were|"
                r"has|have|had|will|would|should)\b[^?]*\?",
                text,
            )
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
