from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from .relation_plan import RelationPlan


@dataclass(frozen=True)
class SearchIntentPlan:
    """
    保留搜尋 state 的薄型資料容器，供 retrieval / next-hop 流程相容使用。

    Args:
        - search_needed: 是否需要進行搜尋。
        - intent: 搜尋型態的簡短標記。
        - target: 搜尋目標摘要。
        - must_include: 後續 query 應保留的核心詞。
        - avoid_terms: 後續 query 應避免的低價值詞。
        - preferred_domain: 偏好的搜尋 domain。
        - answer_role: 預期答案角色。
        - state: 目前搜尋狀態。
        - completed_terms: 已被證據覆蓋的詞。
        - missing_terms: 尚未被證據覆蓋的詞。

    Returns:
        - SearchIntentPlan: 可序列化的搜尋狀態資料。
    """

    search_needed: bool = True
    intent: str = "fact"
    target: str = ""
    must_include: list[str] = field(default_factory=list)
    avoid_terms: list[str] = field(default_factory=list)
    preferred_domain: str = ""
    answer_role: str = "unknown"
    state: str = "pending"
    completed_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    source: str = "embedding_span_role_classifier"
    relation_plan: RelationPlan = field(default_factory=RelationPlan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SearchIntentPlan":
        data = dict(value or {})
        return cls(
            search_needed=bool(data.get("search_needed", True)),
            intent=normalize_text(str(data.get("intent") or "fact")).lower() or "fact",
            target=normalize_text(str(data.get("target") or "")),
            must_include=[
                normalize_text(str(item))
                for item in list(data.get("must_include") or [])
                if normalize_text(str(item))
            ],
            avoid_terms=[
                normalize_text(str(item))
                for item in list(data.get("avoid_terms") or [])
                if normalize_text(str(item))
            ],
            preferred_domain=normalize_text(str(data.get("preferred_domain") or "")),
            answer_role=normalize_text(str(data.get("answer_role") or "unknown")).lower()
            or "unknown",
            state=normalize_text(str(data.get("state") or "pending")) or "pending",
            completed_terms=[
                normalize_text(str(item))
                for item in list(data.get("completed_terms") or [])
                if normalize_text(str(item))
            ],
            missing_terms=[
                normalize_text(str(item))
                for item in list(data.get("missing_terms") or [])
                if normalize_text(str(item))
            ],
            source=normalize_text(str(data.get("source") or "embedding_span_role_classifier"))
            or "embedding_span_role_classifier",
            relation_plan=RelationPlan.from_dict(
                data.get("relation_plan")
                if isinstance(data.get("relation_plan"), dict)
                else None
            ),
        )

    def replace(self, **changes: Any) -> "SearchIntentPlan":
        return dataclasses.replace(self, **changes)


__all__ = ["SearchIntentPlan"]
