from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import hashlib
from threading import RLock
from typing import Any

from utils.network_utils import normalize_text

from .models import EvidenceFact
from utils.canonical_answer_value import CanonicalAnswerValueParser
from .completeness_contract import (
    AbsenceCheck,
    CompletenessContract,
    SetDifferenceDerivation,
)


class TaskFactStore:
    """保存單一任務的 grounded facts，並合併完全相同的來源事實。"""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, ...], EvidenceFact] = {}
        self._by_id: dict[str, EvidenceFact] = {}
        self._completeness_contracts: dict[str, CompletenessContract] = {}
        self._absence_checks: dict[str, AbsenceCheck] = {}
        self._set_difference_derivations: dict[str, SetDifferenceDerivation] = {}
        self._lock = RLock()
        self._revision = 0
        self._value_parser = CanonicalAnswerValueParser()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def add(self, fact: EvidenceFact) -> bool:
        if fact.grounding_status != "grounded":
            return False
        if not fact.fact_id:
            fact = replace(fact, fact_id=self._fact_id(fact))
        key = self._key(fact)
        with self._lock:
            existing = self._facts.get(key)
            if existing is not None:
                merged = self._merge(existing, fact)
                self._facts[key] = merged
                self._by_id[existing.fact_id] = merged
                self._by_id[fact.fact_id] = merged
                if merged != existing:
                    self._revision += 1
                return False
            if fact.fact_id in self._by_id:
                return False
            self._facts[key] = fact
            self._by_id[fact.fact_id] = fact
            self._revision += 1
            return True

    def extend(self, facts: Iterable[EvidenceFact]) -> int:
        return sum(1 for fact in facts if self.add(fact))

    def all(self) -> list[EvidenceFact]:
        with self._lock:
            return list(self._facts.values())

    def get(self, fact_id: str) -> EvidenceFact | None:
        with self._lock:
            return self._by_id.get(str(fact_id or "").strip())

    def by_role(self, role: str) -> list[EvidenceFact]:
        normalized = str(role or "").strip().upper()
        return [fact for fact in self.all() if fact.role == normalized]

    def verifiable_answer_facts(self) -> list[EvidenceFact]:
        return [
            fact
            for fact in self.by_role("ANSWER_SUPPORT")
            if (
                fact.grounding_status == "grounded"
                and fact.qualifiers.get("answer_binding") == "direct"
                and self._relation_grounding_is_verifiable(fact)
                and self._negative_scope_is_verifiable(fact)
            )
        ]

    @staticmethod
    def _relation_grounding_is_verifiable(fact: EvidenceFact) -> bool:
        if fact.extraction_method not in {
            "grounded_answer_value_promotion",
            "direct_contract_adapter",
        }:
            return True
        origin_id = normalize_text(fact.qualifiers.get("origin_fact_id", ""))
        return bool(origin_id and (fact.parent_fact_ids or origin_id))

    def add_completeness_contract(self, contract: CompletenessContract) -> bool:
        if not contract.contract_id or not contract.scope_id:
            return False
        with self._lock:
            if contract.contract_id in self._completeness_contracts:
                return False
            self._completeness_contracts[contract.contract_id] = contract
            self._revision += 1
            return True

    def add_absence_check(self, check: AbsenceCheck) -> bool:
        if not check.check_id or not check.scope_id:
            return False
        with self._lock:
            if check.check_id in self._absence_checks:
                return False
            self._absence_checks[check.check_id] = check
            self._revision += 1
            return True

    def add_set_difference_derivation(
        self,
        derivation: SetDifferenceDerivation,
    ) -> bool:
        if not derivation.derivation_id:
            return False
        with self._lock:
            if derivation.derivation_id in self._set_difference_derivations:
                return False
            self._set_difference_derivations[derivation.derivation_id] = derivation
            self._revision += 1
            return True

    def completeness_contracts(self) -> list[CompletenessContract]:
        with self._lock:
            return list(self._completeness_contracts.values())

    def absence_checks(self) -> list[AbsenceCheck]:
        with self._lock:
            return list(self._absence_checks.values())

    def set_difference_derivations(self) -> list[SetDifferenceDerivation]:
        with self._lock:
            return list(self._set_difference_derivations.values())

    def by_entity(self, entity: str) -> list[EvidenceFact]:
        key = normalize_text(entity).casefold()
        if not key:
            return []
        return [
            fact
            for fact in self.all()
            if key in {
                normalize_text(fact.subject).casefold(),
                normalize_text(fact.object).casefold(),
            }
        ]

    def by_relation(self, relation: str) -> list[EvidenceFact]:
        key = normalize_text(relation).casefold()
        if not key:
            return []
        return [
            fact
            for fact in self.all()
            if normalize_text(fact.relation).casefold() == key
        ]

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TaskFactStore":
        store = cls()
        for item in list((value or {}).get("facts") or []):
            if isinstance(item, dict):
                store.add(EvidenceFact.from_dict(item))
        for item in list((value or {}).get("completeness_contracts") or []):
            if isinstance(item, dict):
                store.add_completeness_contract(CompletenessContract.from_dict(item))
        for item in list((value or {}).get("absence_checks") or []):
            if isinstance(item, dict):
                store.add_absence_check(AbsenceCheck.from_dict(item))
        for item in list((value or {}).get("set_difference_derivations") or []):
            if isinstance(item, dict):
                store.add_set_difference_derivation(
                    SetDifferenceDerivation.from_dict(item)
                )
        store._revision = max(
            store._revision,
            int((value or {}).get("revision", 0) or 0),
        )
        return store

    def to_dict(self) -> dict[str, object]:
        facts = self.all()
        verifiable = self.verifiable_answer_facts()
        completeness_contracts = self.completeness_contracts()
        absence_checks = self.absence_checks()
        set_difference_derivations = self.set_difference_derivations()
        return {
            "revision": self.revision,
            "facts": [fact.to_dict() for fact in facts],
            "fact_count": len(facts),
            "role_counts": {
                role: sum(1 for fact in facts if fact.role == role)
                for role in ("ANSWER_SUPPORT", "BRIDGE", "CONTEXT")
            },
            "source_counts": {
                source_type: sum(1 for fact in facts if fact.source_type == source_type)
                for source_type in sorted({fact.source_type for fact in facts if fact.source_type})
            },
            "derived_fact_count": sum(bool(fact.parent_fact_ids) for fact in facts),
            "cross_context_fact_count": sum(
                fact.extraction_method == "cross_context_semantic_model"
                for fact in facts
            ),
            "cross_context_grounded_count": sum(
                fact.extraction_method == "cross_context_semantic_model"
                and fact.grounding_status == "grounded"
                for fact in facts
            ),
            "multi_unit_fact_count": sum(
                len({ref.unit_id for ref in fact.evidence_refs}) >= 2
                for fact in facts
            ),
            "provenance_ref_count": sum(len(fact.evidence_refs) for fact in facts),
            "verification_ready_count": len(verifiable),
            "completeness_contracts": [
                item.to_dict() for item in completeness_contracts
            ],
            "absence_checks": [item.to_dict() for item in absence_checks],
            "set_difference_derivations": [
                item.to_dict() for item in set_difference_derivations
            ],
            "negative_fact_count": sum(fact.polarity == "negative" for fact in facts),
            "explicit_negative_count": sum(
                fact.qualifiers.get("negation_type") == "explicit_negative"
                for fact in facts
            ),
            "closed_world_absence_count": sum(
                fact.qualifiers.get("negation_type") == "closed_world_absence"
                for fact in facts
            ),
            "unknown_absence_count": sum(
                item.status == "unknown" for item in absence_checks
            ),
            "candidate_answer_facts": [
                {
                    "fact_id": fact.fact_id,
                    "value": fact.object,
                    "subject": fact.subject,
                    "relation": fact.relation,
                    "source_id": fact.source_id,
                    "source_type": fact.source_type,
                    "derived": bool(fact.parent_fact_ids),
                }
                for fact in verifiable
            ],
        }

    def _negative_scope_is_verifiable(self, fact: EvidenceFact) -> bool:
        if fact.polarity != "negative":
            return True
        negation_type = fact.qualifiers.get("negation_type", "")
        if negation_type == "explicit_negative":
            return bool(fact.evidence_spans)
        if not negation_type:
            relation = normalize_text(fact.relation).casefold()
            legacy_negative_relation = any(
                marker in relation
                for marker in (
                    "does not",
                    "did not",
                    "is not",
                    "was not",
                    "not contain",
                    "not mention",
                    "without",
                    "lacks",
                    "absent",
                )
            )
            return bool(fact.evidence_spans and legacy_negative_relation)
        if negation_type in {
            "closed_world_absence",
            "closed_world_set_difference",
            "set_difference_answer",
        }:
            contract_ids = [
                item.strip()
                for item in (
                    fact.qualifiers.get("completeness_contract_ids")
                    or fact.qualifiers.get("completeness_contract_id")
                    or ""
                ).split(",")
                if item.strip()
            ]
            return bool(contract_ids) and all(
                (
                    contract := self._completeness_contracts.get(contract_id)
                ) is not None
                and contract.complete
                for contract_id in contract_ids
            )
        return False

    @classmethod
    def _key(cls, fact: EvidenceFact) -> tuple[str, ...]:
        canonical_object = CanonicalAnswerValueParser().parse(fact.object).normalized_text
        return (
            normalize_text(fact.subject).casefold(),
            normalize_text(fact.relation).casefold(),
            canonical_object or normalize_text(fact.object).casefold(),
            fact.polarity,
            fact.role,
            normalize_text(fact.goal_id).casefold(),
        )

    @classmethod
    def _fact_id(cls, fact: EvidenceFact) -> str:
        digest = hashlib.sha1("\x1f".join(cls._key(fact)).encode("utf-8")).hexdigest()[:16]
        return f"fact-{digest}"

    @staticmethod
    def _merge(existing: EvidenceFact, incoming: EvidenceFact) -> EvidenceFact:
        refs = []
        seen_refs: set[tuple[str, ...]] = set()
        for ref in [*existing.evidence_refs, *incoming.evidence_refs]:
            key = (
                normalize_text(ref.source_id).casefold(),
                normalize_text(ref.unit_id).casefold(),
                normalize_text(ref.text).casefold(),
                normalize_text(ref.document_id).casefold(),
            )
            if key in seen_refs:
                continue
            refs.append(ref)
            seen_refs.add(key)
        spans: list[str] = []
        seen_spans: set[str] = set()
        for span in [*existing.evidence_spans, *incoming.evidence_spans]:
            cleaned = normalize_text(span)
            key = cleaned.casefold()
            if cleaned and key not in seen_spans:
                spans.append(cleaned)
                seen_spans.add(key)
        contexts = [
            value
            for value in [normalize_text(existing.context), normalize_text(incoming.context)]
            if value
        ]
        if not contexts:
            context = ""
        elif len(set(contexts)) == 1:
            context = contexts[0]
        else:
            context = "\n\n".join(dict.fromkeys(contexts))
        return replace(
            existing,
            qualifiers={**incoming.qualifiers, **existing.qualifiers},
            evidence_spans=spans,
            evidence_refs=refs,
            context=context,
            parent_fact_ids=list(
                dict.fromkeys([*existing.parent_fact_ids, *incoming.parent_fact_ids])
            ),
        )


__all__ = ["TaskFactStore"]
