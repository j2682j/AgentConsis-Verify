from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from utils.network_utils import normalize_text


SOURCE_KINDS = frozenset({"web", "video", "academic", "collection"})
ACCESS_MODES = frozenset({"search", "direct_fetch", "browser"})
URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


@dataclass(frozen=True)
class SourceRequirement:
    """
    描述單一搜尋查詢需要的來源型態與取得方式。

    Args:
     - source_kind: 來源型態，支援 web、video、academic、collection。
     - access_mode: 取得方式，支援 search、direct_fetch、browser。
     - source_hint: 題目指定的平台、網站名稱或 URL 提示。

    Returns:
     - SourceRequirement: 經過正規化的來源取得需求。
    """

    source_kind: str = "web"
    access_mode: str = "search"
    source_hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any] | None,
        *,
        query: str = "",
    ) -> "SourceRequirement":
        data = dict(value or {})
        source_kind = normalize_text(str(data.get("source_kind") or "web")).lower()
        access_mode = normalize_text(str(data.get("access_mode") or "search")).lower()
        source_hint = normalize_text(str(data.get("source_hint") or ""))

        if source_kind not in SOURCE_KINDS:
            source_kind = "web"
        if access_mode not in ACCESS_MODES:
            access_mode = "search"
        if access_mode == "direct_fetch" and not (
            source_hint or URL_RE.search(str(query or ""))
        ):
            access_mode = "search"

        return cls(
            source_kind=source_kind,
            access_mode=access_mode,
            source_hint=source_hint,
        )


@dataclass(frozen=True)
class SearchQueryRequest:
    """
    將搜尋文字與其來源需求綁定，避免後續路由遺失取得條件。

    Args:
     - query: 要送入來源取得流程的查詢文字。
     - source_requirement: 此查詢對應的來源型態與取得方式。

    Returns:
     - SearchQueryRequest: 可交由來源路由器執行的查詢請求。
    """

    query: str
    source_requirement: SourceRequirement = SourceRequirement()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source_requirement": self.source_requirement.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SearchQueryRequest | None":
        data = dict(value or {})
        query = normalize_text(str(data.get("query") or data.get("text") or ""))
        if not query:
            return None
        requirement_value = data.get("source_requirement")
        if not isinstance(requirement_value, dict):
            requirement_value = data
        return cls(
            query=query,
            source_requirement=SourceRequirement.from_dict(
                requirement_value,
                query=query,
            ),
        )

    @classmethod
    def fallback(cls, query: str) -> "SearchQueryRequest":
        return cls(
            query=normalize_text(query),
            source_requirement=SourceRequirement(),
        )


__all__ = [
    "ACCESS_MODES",
    "SOURCE_KINDS",
    "SearchQueryRequest",
    "SourceRequirement",
]
