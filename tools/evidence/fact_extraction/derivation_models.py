from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactDerivation:
    """描述一次可由來源事實重現的關係推導。"""

    derivation_id: str
    derivation_type: str
    parent_fact_ids: list[str]
    result_fact_id: str
    explanation: str
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactDerivationResult:
    """保存新推導事實、來源鏈與推導診斷資訊。"""

    derivations: list[FactDerivation] = field(default_factory=list)
    added_fact_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivations": [item.to_dict() for item in self.derivations],
            "added_fact_ids": list(self.added_fact_ids),
            "diagnostics": dict(self.diagnostics),
        }


__all__ = ["FactDerivation", "FactDerivationResult"]
