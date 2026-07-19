from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_FACT_ROLES = {"ANSWER_SUPPORT", "BRIDGE", "CONTEXT"}
VALID_POLARITIES = {"positive", "negative"}
VALID_GROUNDING_STATUSES = {"grounded", "ambiguous", "invalid"}


@dataclass(frozen=True)
class FactEvidenceRef:
    """保存一段事實在原始來源單位中的可追溯位置。"""

    source_id: str
    unit_id: str
    text: str
    document_id: str = ""
    page: int | None = None
    section: str = ""
    start_offset: int = -1
    end_offset: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactEvidenceRef":
        page_value = value.get("page")
        try:
            page = int(page_value) if page_value not in (None, "") else None
        except (TypeError, ValueError):
            page = None
        try:
            start_offset = int(value.get("start_offset", -1))
        except (TypeError, ValueError):
            start_offset = -1
        try:
            end_offset = int(value.get("end_offset", -1))
        except (TypeError, ValueError):
            end_offset = -1
        return cls(
            source_id=str(value.get("source_id") or "").strip(),
            unit_id=str(value.get("unit_id") or "").strip(),
            text=str(value.get("text") or "").strip(),
            document_id=str(value.get("document_id") or "").strip(),
            page=page,
            section=str(value.get("section") or "").strip(),
            start_offset=start_offset,
            end_offset=end_offset,
        )


@dataclass(frozen=True)
class StructuredRelationRecord:
    """完整保存一列結構資料，避免語意選擇提前遺失列關係。"""

    record_id: str
    source_id: str
    source_type: str
    structure_id: str
    row_id: str
    fields: dict[str, str]
    normalized_fields: dict[str, str]
    provenance: list[FactEvidenceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    evidence_refs: list[FactEvidenceRef] = field(default_factory=list)
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
            evidence_refs=[
                FactEvidenceRef.from_dict(item)
                for item in list(value.get("evidence_refs") or [])
                if isinstance(item, dict)
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
    "FactEvidenceRef",
    "SemanticExtractionResult",
    "SemanticSourceUnit",
    "StructuredRelationRecord",
    "VALID_FACT_ROLES",
    "VALID_GROUNDING_STATUSES",
    "VALID_POLARITIES",
]
