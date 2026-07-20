from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from utils.network_utils import normalize_text
from .models import EvidenceFact


VALID_AGGREGATION_OPERATORS = frozenset(
    {
        "count",
        "percentage",
        "sum",
        "average",
        "minimum",
        "maximum",
        "set_difference",
        "intersection",
        "union",
    }
)


@dataclass(frozen=True)
class AggregationDerivation:
    derivation_id: str
    operator: str
    input_fact_ids: list[str]
    result: str
    completeness_contract_ids: list[str]
    result_fact_id: str = ""
    grounding_status: str = "ambiguous"
    answer_bound: bool = False
    goal_id: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AggregationDerivation":
        operator = normalize_text(str(value.get("operator") or "")).lower()
        if operator not in VALID_AGGREGATION_OPERATORS:
            operator = ""
        return cls(
            derivation_id=normalize_text(str(value.get("derivation_id") or "")),
            operator=operator,
            input_fact_ids=_strings(value.get("input_fact_ids")),
            result=normalize_text(str(value.get("result") or "")),
            completeness_contract_ids=_strings(
                value.get("completeness_contract_ids")
            ),
            result_fact_id=normalize_text(str(value.get("result_fact_id") or "")),
            grounding_status=normalize_text(
                str(value.get("grounding_status") or "ambiguous")
            ).lower(),
            answer_bound=bool(value.get("answer_bound", False)),
            goal_id=normalize_text(str(value.get("goal_id") or "")),
            reason=normalize_text(str(value.get("reason") or "")),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_fact(self) -> EvidenceFact:
        subject = normalize_text(str(self.metadata.get("subject") or "aggregate result"))
        relation = normalize_text(
            str(self.metadata.get("answer_requirement") or f"aggregate {self.operator}")
        )
        return EvidenceFact(
            fact_id=self.result_fact_id or f"{self.derivation_id}-result",
            subject=subject,
            relation=relation,
            object=self.result,
            qualifiers={
                "answer_binding": "direct",
                "aggregation_operator": self.operator,
                "completeness_contract_ids": ",".join(
                    self.completeness_contract_ids
                ),
            },
            polarity="positive",
            role="ANSWER_SUPPORT",
            goal_id=self.goal_id,
            source_id=f"derived:{self.derivation_id}",
            source_type="derived",
            source_title="Verified aggregate derivation",
            grounding_status=self.grounding_status,
            extraction_method="aggregation_derivation",
            parent_fact_ids=list(self.input_fact_ids),
            derivation_type=self.operator,
        )


def _strings(value: Any) -> list[str]:
    return list(
        dict.fromkeys(
            normalize_text(str(item))
            for item in list(value or [])
            if normalize_text(str(item))
        )
    )


__all__ = ["AggregationDerivation", "VALID_AGGREGATION_OPERATORS"]
