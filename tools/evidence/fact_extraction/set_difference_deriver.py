from __future__ import annotations

import hashlib
from typing import Iterable

from utils.network_utils import normalize_for_exact, normalize_text

from .completeness_contract import CompletenessContract, SetDifferenceDerivation
from .models import EvidenceFact


class SetDifferenceFactDeriver:
    """在 universe 與 observed scope 都完整時建立可追溯集合差異事實。"""

    def derive(
        self,
        *,
        universe_values: Iterable[str],
        observed_values: Iterable[str],
        completeness_contracts: Iterable[CompletenessContract],
        universe_fact_ids: Iterable[str] = (),
        observed_fact_ids: Iterable[str] = (),
        negative_relation: str = "is absent from",
        negative_object: str = "observed set",
        answer_subject: str = "missing value",
        answer_relation: str = "is",
        answer_requirement: str = "",
        goal_id: str = "",
        source_id: str = "set-difference",
        source_type: str = "derived",
    ) -> tuple[SetDifferenceDerivation, list[EvidenceFact]] | None:
        contracts = list(completeness_contracts)
        if not contracts or any(not contract.complete for contract in contracts):
            return None
        universe = self._unique(universe_values)
        observed_keys = {
            normalize_for_exact(value)
            for value in observed_values
            if normalize_for_exact(value)
        }
        missing = [
            value for value in universe if normalize_for_exact(value) not in observed_keys
        ]
        universe_ids = self._strings(universe_fact_ids)
        observed_ids = self._strings(observed_fact_ids)
        contract_ids = [contract.contract_id for contract in contracts]
        raw = "\x1f".join([*universe, "--", *sorted(observed_keys), *contract_ids])
        derivation = SetDifferenceDerivation(
            derivation_id="SD-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
            universe_fact_ids=universe_ids,
            observed_fact_ids=observed_ids,
            missing_values=missing,
            completeness_contract_ids=contract_ids,
        )
        parent_ids = list(dict.fromkeys([*universe_ids, *observed_ids]))
        facts: list[EvidenceFact] = []
        for value in missing:
            fact_raw = "\x1f".join([derivation.derivation_id, value, negative_relation])
            qualifiers = {
                "answer_binding": "direct",
                "answer_requirement": normalize_text(answer_requirement),
                "negation_type": "closed_world_set_difference",
                "set_difference_derivation_id": derivation.derivation_id,
                "completeness_contract_ids": ",".join(contract_ids),
            }
            facts.append(
                EvidenceFact(
                    fact_id="NF-" + hashlib.sha1(fact_raw.encode("utf-8")).hexdigest()[:12],
                    subject=value,
                    relation=normalize_text(negative_relation),
                    object=normalize_text(negative_object),
                    qualifiers=qualifiers,
                    polarity="negative",
                    role="ANSWER_SUPPORT",
                    goal_id=normalize_text(goal_id),
                    context=f"{value} is the set difference result.",
                    source_id=normalize_text(source_id),
                    source_type=normalize_text(source_type),
                    grounding_status="grounded",
                    extraction_method="deterministic_set_difference",
                    parent_fact_ids=parent_ids,
                    derivation_type="set_difference",
                )
            )
            if normalize_text(answer_subject):
                answer_raw = "\x1f".join(
                    [derivation.derivation_id, answer_subject, answer_relation, value]
                )
                facts.append(
                    EvidenceFact(
                        fact_id="DF-" + hashlib.sha1(answer_raw.encode("utf-8")).hexdigest()[:12],
                        subject=normalize_text(answer_subject),
                        relation=normalize_text(answer_relation),
                        object=value,
                        qualifiers={
                            **qualifiers,
                            "negation_type": "set_difference_answer",
                        },
                        polarity="positive",
                        role="ANSWER_SUPPORT",
                        goal_id=normalize_text(goal_id),
                        context=f"{answer_subject} {answer_relation} {value}",
                        source_id=normalize_text(source_id),
                        source_type=normalize_text(source_type),
                        grounding_status="grounded",
                        extraction_method="deterministic_set_difference",
                        parent_fact_ids=parent_ids,
                        derivation_type="set_difference",
                    )
                )
        return derivation, facts

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = normalize_for_exact(cleaned)
            if cleaned and key and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    @staticmethod
    def _strings(values: Iterable[str]) -> list[str]:
        return list(
            dict.fromkeys(
                normalize_text(value) for value in values if normalize_text(value)
            )
        )


__all__ = ["SetDifferenceFactDeriver"]
