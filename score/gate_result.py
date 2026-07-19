from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import CandidateEvaluation


@dataclass
class CandidateGateDecision:
    """Record one candidate's outcome at an ordered winner-selection gate."""

    candidate_key: str
    outcome: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    """Carry survivors and eliminated candidates to the next ordered gate."""

    gate_name: str
    survivors: list[CandidateEvaluation] = field(default_factory=list)
    eliminated: list[CandidateGateDecision] = field(default_factory=list)
    decisions: list[CandidateGateDecision] = field(default_factory=list)
    terminal_status: str = ""
    terminal_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "survivors": [item.candidate_key for item in self.survivors],
            "eliminated": [item.to_dict() for item in self.eliminated],
            "decisions": [item.to_dict() for item in self.decisions],
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "metadata": dict(self.metadata),
            "active_candidate_keys": [
                item.candidate_key
                for item in self.survivors
                if item.selection_state == "active"
            ],
            "reserve_candidate_keys": [
                item.candidate_key
                for item in self.survivors
                if item.selection_state == "reserve"
            ],
        }


__all__ = ["CandidateGateDecision", "GateResult"]
