from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import hashlib
from threading import RLock
from typing import Any

from utils.network_utils import normalize_text

from .models import EvidenceFact


class TaskFactStore:
    """保存單一任務的 grounded facts，並合併完全相同的來源事實。"""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, ...], EvidenceFact] = {}
        self._by_id: dict[str, EvidenceFact] = {}
        self._lock = RLock()

    def add(self, fact: EvidenceFact) -> bool:
        if fact.grounding_status != "grounded":
            return False
        if not fact.fact_id:
            fact = replace(fact, fact_id=self._fact_id(fact))
        key = self._key(fact)
        with self._lock:
            if key in self._facts or fact.fact_id in self._by_id:
                return False
            self._facts[key] = fact
            self._by_id[fact.fact_id] = fact
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
            )
        ]

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
        return store

    def to_dict(self) -> dict[str, object]:
        facts = self.all()
        verifiable = self.verifiable_answer_facts()
        return {
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
            "verification_ready_count": len(verifiable),
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

    @staticmethod
    def _key(fact: EvidenceFact) -> tuple[str, ...]:
        return (
            normalize_text(fact.subject).casefold(),
            normalize_text(fact.relation).casefold(),
            normalize_text(fact.object).casefold(),
            fact.polarity,
            fact.role,
            normalize_text(fact.goal_id).casefold(),
            normalize_text(fact.source_id).casefold(),
        )

    @classmethod
    def _fact_id(cls, fact: EvidenceFact) -> str:
        digest = hashlib.sha1("\x1f".join(cls._key(fact)).encode("utf-8")).hexdigest()[:16]
        return f"fact-{digest}"


__all__ = ["TaskFactStore"]
