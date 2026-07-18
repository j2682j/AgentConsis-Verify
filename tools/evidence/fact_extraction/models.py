from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_FACT_ROLES = {"ANSWER_SUPPORT", "BRIDGE", "CONTEXT"}
VALID_POLARITIES = {"positive", "negative"}
VALID_GROUNDING_STATUSES = {"grounded", "ambiguous", "invalid"}


@dataclass(frozen=True)
class EvidenceFact:
    """
    保存一筆可回溯至來源文字的語意事實。

    Args:
     - subject: 事實描述的主體。
     - relation: 主體與受詞之間的語意關係。
     - object: 關係指向的值、實體或狀態。
     - evidence_spans: 可在原始內容中定位的支持文字。

    Returns:
     - EvidenceFact: 可交由 evidence contract 與 support checker 使用的事實。
    """

    fact_id: str
    subject: str
    relation: str
    object: str
    qualifiers: dict[str, str] = field(default_factory=dict)
    polarity: str = "positive"
    role: str = "CONTEXT"
    goal_id: str = ""
    evidence_spans: list[str] = field(default_factory=list)
    context: str = ""
    source_id: str = ""
    source_type: str = ""
    source_title: str = ""
    grounding_status: str = "ambiguous"
    extraction_method: str = "semantic_model"
    parent_fact_ids: list[str] = field(default_factory=list)
    derivation_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceFact":
        qualifiers = value.get("qualifiers")
        return cls(
            fact_id=str(value.get("fact_id") or "").strip(),
            subject=str(value.get("subject") or "").strip(),
            relation=str(value.get("relation") or "").strip(),
            object=str(value.get("object") or "").strip(),
            qualifiers={
                str(key): str(item)
                for key, item in dict(qualifiers or {}).items()
                if str(key).strip() and str(item).strip()
            },
            polarity=str(value.get("polarity") or "positive").strip().lower(),
            role=str(value.get("role") or "CONTEXT").strip().upper(),
            goal_id=str(value.get("goal_id") or "").strip(),
            evidence_spans=[
                str(item).strip()
                for item in list(value.get("evidence_spans") or [])[:2]
                if str(item).strip()
            ],
            context=str(value.get("context") or "").strip(),
            source_id=str(value.get("source_id") or "").strip(),
            source_type=str(value.get("source_type") or "").strip(),
            source_title=str(value.get("source_title") or "").strip(),
            grounding_status=str(
                value.get("grounding_status") or "ambiguous"
            ).strip().lower(),
            extraction_method=str(
                value.get("extraction_method") or "semantic_model"
            ).strip(),
            parent_fact_ids=[
                str(item).strip()
                for item in list(value.get("parent_fact_ids") or [])
                if str(item).strip()
            ],
            derivation_type=str(value.get("derivation_type") or "").strip(),
        )


@dataclass(frozen=True)
class SemanticSourceUnit:
    """描述一次語意抽取所使用的短來源單位。"""

    unit_id: str
    text: str
    source_id: str
    source_type: str
    source_title: str = ""
    candidate_span: str = ""
    requested_role: str = ""
    goal_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticExtractionResult:
    """保存語意抽取完成後的 facts、拒絕項目與精簡診斷。"""

    facts: list[EvidenceFact] = field(default_factory=list)
    rejected_items: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [item.to_dict() for item in self.facts],
            "rejected_items": list(self.rejected_items),
            "diagnostics": dict(self.diagnostics),
        }


__all__ = [
    "EvidenceFact",
    "SemanticExtractionResult",
    "SemanticSourceUnit",
    "VALID_FACT_ROLES",
    "VALID_GROUNDING_STATUSES",
    "VALID_POLARITIES",
]
