from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config import ToolEvidenceRecord
from tools.evidence.fact_extraction import TaskFactStore


@dataclass
class EvidenceSupportContext:
    """Hold task-level evidence that is cloned for each candidate path."""

    base_fact_store: TaskFactStore
    base_records: list[ToolEvidenceRecord] = field(default_factory=list)
    answer_requirement: str = ""
    answer_role: str = ""
    task_route: str = ""
    evidence_revision: int = 0
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def candidate_evidence(self) -> dict[str, Any]:
        """Create an isolated evidence payload and fact store for one path."""

        payload = dict(self.evidence_payload)
        store = TaskFactStore.from_dict(self.base_fact_store.to_dict())
        payload["_fact_store"] = store
        payload["fact_store"] = store.to_dict()
        payload["answer_requirement"] = self.answer_requirement
        payload["answer_role"] = self.answer_role
        return payload


__all__ = ["EvidenceSupportContext"]
