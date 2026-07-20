from __future__ import annotations

from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tools.search_result_builder.config import UnverifiedReference
from utils.network_utils import normalize_text


class BestEffortReferenceSelector:
    """Select compact retrieval references when strict evidence is unavailable."""

    def __init__(
        self,
        *,
        max_items: int = 3,
        max_chars_per_item: int = 800,
        max_total_chars: int = 2400,
        min_chars: int = 80,
        max_items_per_domain: int = 2,
    ) -> None:
        self.max_items = max(1, max_items)
        self.max_chars_per_item = max(120, max_chars_per_item)
        self.max_total_chars = max(self.max_chars_per_item, max_total_chars)
        self.min_chars = max(20, min_chars)
        self.max_items_per_domain = max(1, max_items_per_domain)

    def select(
        self,
        output: dict[str, Any],
        *,
        strict_evidence_items: list[dict[str, Any]] | None = None,
    ) -> list[UnverifiedReference]:
        if strict_evidence_items:
            return []

        retrieval = output.get("retrieval")
        retrieval = retrieval if isinstance(retrieval, dict) else {}
        ranked: list[tuple[float, int, int, dict[str, Any]]] = []
        collection_rows: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
        for round_position, round_info in enumerate(
            list(retrieval.get("rounds") or []),
            start=1,
        ):
            if not isinstance(round_info, dict):
                continue
            round_index = int(round_info.get("round_index", round_position) or round_position)
            for document_position, document in enumerate(
                list(round_info.get("documents") or []),
                start=1,
            ):
                if not isinstance(document, dict) or document.get("duplicate"):
                    continue
                text = normalize_text(document.get("text"))
                group_key = self._collection_group_key(document)
                if group_key:
                    # Collection rows are typically short; keep them even
                    # below min_chars so an aggregate table stays complete.
                    if text:
                        collection_rows.setdefault(group_key, []).append(
                            (
                                float(document.get("retrieval_score", 0.0) or 0.0),
                                round_index,
                                {**document, "text": text},
                            )
                        )
                    continue
                if len(text) < self.min_chars:
                    continue
                ranked.append(
                    (
                        -float(document.get("retrieval_score", 0.0) or 0.0),
                        round_index,
                        document_position,
                        {**document, "text": text},
                    )
                )

        for group_position, (group_key, rows) in enumerate(
            sorted(collection_rows.items()),
            start=1,
        ):
            merged = self._merged_collection_document(group_key, rows)
            if merged is None:
                continue
            ranked.append(
                (
                    -float(merged.get("retrieval_score", 0.0) or 0.0),
                    int(merged.get("_round_index", 1) or 1),
                    group_position,
                    merged,
                )
            )

        selected: list[UnverifiedReference] = []
        seen_urls: set[str] = set()
        seen_content: set[str] = set()
        domain_counts: dict[str, int] = {}
        total_chars = 0
        for _, round_index, _, document in sorted(ranked):
            url = normalize_text(document.get("url"))
            url_key = self._url_key(url)
            text = normalize_text(document.get("text"))
            content_key = self._content_key(text)
            domain = self._domain(url)
            is_merged_collection = bool(document.get("_merged_collection"))
            if url_key and url_key in seen_urls:
                continue
            if content_key in seen_content:
                continue
            if (
                not is_merged_collection
                and domain
                and domain_counts.get(domain, 0) >= self.max_items_per_domain
            ):
                continue

            remaining = self.max_total_chars - total_chars
            if remaining < self.min_chars:
                break
            per_item_limit = (
                self.max_total_chars if is_merged_collection else self.max_chars_per_item
            )
            text = self._truncate(text, min(per_item_limit, remaining))
            if len(text) < self.min_chars:
                continue
            selected.append(
                UnverifiedReference(
                    reference_id=f"R{len(selected) + 1}",
                    source_id=(
                        normalize_text(document.get("document_id"))
                        or normalize_text(document.get("record_id"))
                        or f"round-{round_index}-document-{len(selected) + 1}"
                    ),
                    title=normalize_text(document.get("title")) or "Unknown",
                    text=text,
                    url=url,
                    retrieval_score=float(
                        document.get("retrieval_score", 0.0) or 0.0
                    ),
                    retrieval_round=round_index,
                    source_type=(
                        normalize_text(document.get("record_type")) or "passage"
                    ),
                )
            )
            total_chars += len(text)
            seen_content.add(content_key)
            if url_key:
                seen_urls.add(url_key)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected) >= self.max_items:
                break
        return selected

    def _collection_group_key(self, document: dict[str, Any]) -> str:
        """Group sibling collection rows extracted from one parent page."""

        record_type = normalize_text(document.get("record_type")).casefold()
        if record_type in {"", "passage"}:
            return ""
        fields = document.get("record_fields")
        fields = fields if isinstance(fields, dict) else {}
        parent = (
            normalize_text(fields.get("parent_url"))
            or normalize_text(document.get("parent_url"))
            or normalize_text(fields.get("source"))
            or self._domain(normalize_text(document.get("url")))
        )
        if not parent:
            return ""
        return f"{record_type}::{parent.casefold()}"

    def _merged_collection_document(
        self,
        group_key: str,
        rows: list[tuple[float, int, dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """Merge sibling rows into one reference so aggregates stay complete.

        Per-row selection capped rows via the domain limit, which starved
        count/aggregate questions of most of the table; one merged reference
        keeps every row the char budget allows.
        """

        if not rows:
            return None
        if len(rows) == 1:
            document = dict(rows[0][2])
            document["_round_index"] = rows[0][1]
            return document
        ordered = sorted(rows, key=lambda item: -item[0])
        best_score = ordered[0][0]
        first_round = min(item[1] for item in rows)
        source_title = (
            normalize_text(ordered[0][2].get("title"))
            or group_key.split("::", 1)[-1]
        )
        lines = []
        seen_lines: set[str] = set()
        for _, _, document in ordered:
            line = normalize_text(document.get("text"))
            line = line if len(line) <= 220 else line[:220].rstrip() + " ..."
            key = line.casefold()
            if not line or key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(f"- {line}")
        if not lines:
            return None
        parent = group_key.split("::", 1)[-1]
        return {
            "title": f"Collection rows ({len(lines)}): {source_title}",
            "text": "\n".join(lines),
            "url": normalize_text(ordered[0][2].get("parent_url")) or parent,
            "retrieval_score": best_score,
            "record_type": "collection_rows",
            "document_id": f"merged::{parent}",
            "_merged_collection": True,
            "_round_index": first_round,
        }

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    @staticmethod
    def _content_key(text: str) -> str:
        normalized = normalize_text(text).casefold()
        return sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return urlsplit(url).netloc.casefold().removeprefix("www.")
        except ValueError:
            return ""

    @staticmethod
    def _url_key(url: str) -> str:
        try:
            parsed = urlsplit(url)
            if not parsed.netloc:
                return ""
            return urlunsplit(
                (
                    parsed.scheme.casefold(),
                    parsed.netloc.casefold().removeprefix("www."),
                    parsed.path.rstrip("/"),
                    "",
                    "",
                )
            )
        except ValueError:
            return ""


__all__ = ["BestEffortReferenceSelector"]
