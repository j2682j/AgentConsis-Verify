from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from utils.network_utils import normalize_text


_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid"})


def canonicalize_page_url(url: str) -> str:
    """
    將來源 URL 正規化為穩定的頁面識別字串。

    Args:
     - url: 原始來源 URL。

    Returns:
     - str: 移除 fragment 與追蹤參數後的 canonical URL。
    """

    text = normalize_text(url)
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.netloc:
        return text.casefold().rstrip("/")
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lowered = key.casefold()
        if lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, value))
    query_items.sort(key=lambda item: (item[0].casefold(), item[1]))
    return urlunparse(
        (
            parsed.scheme.casefold() or "https",
            parsed.netloc.casefold(),
            parsed.path.rstrip("/"),
            "",
            urlencode(query_items),
            "",
        )
    )


def build_page_id(
    *,
    canonical_url: str = "",
    fallback_identity: str = "",
) -> str:
    """
    根據 canonical URL 或本地文件識別建立穩定 page ID。

    Args:
     - canonical_url: 已正規化的來源 URL。
     - fallback_identity: 無 URL 時使用的文件或來源識別。

    Returns:
     - str: 可跨 query 與 hop 比對的 page ID。
    """

    identity = normalize_text(canonical_url or fallback_identity).casefold()
    if not identity:
        return ""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"page-{digest}"


@dataclass
class PassagePageIndex:
    """
    保存 corpus passage 到 page 與 section 的反向索引。

    Args:
     - passage_ids_by_page: Page ID 對應的 passage IDs。
     - passage_ids_by_section: Page/section 對應的 passage IDs。
     - page_id_by_passage: Passage ID 對應的 Page ID。

    Returns:
     - PassagePageIndex: Page-scoped retrieval 使用的記憶體索引。
    """

    passage_ids_by_page: dict[str, list[str]] = field(default_factory=dict)
    passage_ids_by_section: dict[tuple[str, int], list[str]] = field(
        default_factory=dict
    )
    page_id_by_passage: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, passages: Iterable[dict[str, Any]]) -> "PassagePageIndex":
        by_page: dict[str, list[tuple[int, str]]] = defaultdict(list)
        by_section: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(
            list
        )
        page_by_passage: dict[str, str] = {}
        for document in passages:
            passage_id = normalize_text(str(document.get("id") or ""))
            if not passage_id:
                continue
            canonical_url = canonicalize_page_url(
                str(
                    document.get("canonical_url")
                    or document.get("source_url")
                    or document.get("url")
                    or document.get("parent_url")
                    or ""
                )
            )
            page_id = normalize_text(str(document.get("page_id") or ""))
            if not page_id:
                page_id = build_page_id(
                    canonical_url=canonical_url,
                    fallback_identity=passage_id,
                )
            passage_index = _safe_int(document.get("passage_index"), default=0)
            section_index = _safe_int(document.get("section_index"), default=0)
            page_by_passage[passage_id] = page_id
            by_page[page_id].append((passage_index, passage_id))
            by_section[(page_id, section_index)].append(
                (passage_index, passage_id)
            )
        return cls(
            passage_ids_by_page={
                page_id: [
                    passage_id
                    for _, passage_id in sorted(
                        values,
                        key=lambda item: (item[0], item[1]),
                    )
                ]
                for page_id, values in by_page.items()
            },
            passage_ids_by_section={
                key: [
                    passage_id
                    for _, passage_id in sorted(
                        values,
                        key=lambda item: (item[0], item[1]),
                    )
                ]
                for key, values in by_section.items()
            },
            page_id_by_passage=page_by_passage,
        )


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "PassagePageIndex",
    "build_page_id",
    "canonicalize_page_url",
]
