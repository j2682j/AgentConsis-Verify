from __future__ import annotations

from hashlib import sha256
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tools.search_result_builder.config import UnverifiedReference
from utils.network_utils import normalize_text


class BestEffortReferenceSelector:
    """Select compact retrieval references when strict evidence is unavailable."""

    _YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
    _YEAR_RANGE_RE = re.compile(
        r"\b(?:between|from)\s+"
        r"(?P<start>(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\s+"
        r"(?:and|to|through|until|-)\s+"
        r"(?P<end>(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\b",
        flags=re.IGNORECASE,
    )
    _COMPACT_YEAR_RANGE_RE = re.compile(
        r"\b(?P<start>(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\s*[-\u2013]\s*"
        r"(?P<end>(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\b"
    )
    _QUOTED_PHRASE_RE = re.compile(
        r'["\u201c\u201d]([^"\u201c\u201d]{3,160})["\u201c\u201d]'
    )
    _EMPTY_WIKI_ROW_RE = re.compile(
        r"\b(?:wiki(?:tionary|pedia|books|quote|source|news|versity|voyage)|"
        r"commons)\s*\(\s*0\s+entries\s*\)",
        flags=re.IGNORECASE,
    )
    _EDIT_PLACEHOLDER_RE = re.compile(
        r"(?:^|\s)content\s*:\s*edit(?:\s|$)",
        flags=re.IGNORECASE,
    )
    _WIKIDATA_METADATA_RE = re.compile(
        r"(?:no\s+label\s+defined|default\s+for\s+all\s+languages|"
        r"also\s+known\s+as\s*:|property\s*:\s*p360)",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        max_items: int = 5,
        max_chars_per_item: int = 800,
        max_total_chars: int = 4000,
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
        question = normalize_text(output.get("question"))
        question_constraints = self._question_constraints(question)
        merge_parent_record_types = bool(question_constraints[0])
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
                group_key = self._collection_group_key(
                    document,
                    merge_parent_record_types=merge_parent_record_types,
                )
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
            merged = self._merged_collection_document(
                group_key,
                rows,
                question=question,
            )
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
                min(2400, self.max_total_chars)
                if is_merged_collection
                else self.max_chars_per_item
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

    def _collection_group_key(
        self,
        document: dict[str, Any],
        *,
        merge_parent_record_types: bool = False,
    ) -> str:
        """Group sibling collection rows extracted from one parent page."""

        record_type = normalize_text(document.get("record_type")).casefold()
        if record_type in {"", "passage"}:
            return ""
        fields = document.get("record_fields")
        fields = fields if isinstance(fields, dict) else {}
        explicit_parent = (
            normalize_text(fields.get("parent_url"))
            or normalize_text(document.get("parent_url"))
            or normalize_text(fields.get("source"))
        )
        if explicit_parent and merge_parent_record_types:
            # Extraction passes may label sibling rows differently (for
            # example article vs database_row). For range-constrained
            # collection questions, the parent page defines one logical set.
            return f"collection::{explicit_parent.casefold()}"
        if explicit_parent:
            return f"{record_type}::{explicit_parent.casefold()}"
        domain = self._domain(normalize_text(document.get("url")))
        if not domain:
            return ""
        return f"{record_type}::{domain}"

    def _merged_collection_document(
        self,
        group_key: str,
        rows: list[tuple[float, int, dict[str, Any]]],
        *,
        question: str = "",
    ) -> dict[str, Any] | None:
        """Merge sibling rows into one reference so aggregates stay complete.

        Per-row selection capped rows via the domain limit, which starved
        count/aggregate questions of most of the table; one merged reference
        keeps every row the char budget allows.
        """

        effective_rows = [
            item
            for item in rows
            if not self._is_collection_placeholder(
                normalize_text(item[2].get("text"))
            )
        ]
        if len(effective_rows) == 1 and len(rows) == 1:
            # A genuine singleton structured record is still useful as a
            # normal reference. Only mixed groups reduced to one row by junk
            # filtering are abandoned as incomplete collections.
            document = dict(effective_rows[0][2])
            document["_round_index"] = effective_rows[0][1]
            return document
        if len(effective_rows) < 2:
            return None
        original_score_ordered = sorted(rows, key=lambda item: -item[0])
        score_ordered = sorted(effective_rows, key=lambda item: -item[0])
        constraints = self._question_constraints(question)
        if constraints[0] or constraints[1] or constraints[2]:
            ordered = sorted(
                effective_rows,
                key=lambda item: (
                    self._constraint_priority(
                        normalize_text(item[2].get("text")),
                        constraints,
                    ),
                    -item[0],
                    item[1],
                ),
            )
        else:
            ordered = score_ordered
        # Filtering navigation rows must not unexpectedly move an otherwise
        # useful collection behind unrelated references. Preserve the parent
        # collection's original retrieval rank while filtering only its text.
        best_score = original_score_ordered[0][0]
        first_round = min(item[1] for item in effective_rows)
        source_title = (
            normalize_text(score_ordered[0][2].get("title"))
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

    @classmethod
    def _is_collection_placeholder(cls, text: str) -> bool:
        """Reject navigation rows that contain no retrievable record content."""

        value = normalize_text(text)
        if not value:
            return True
        if cls._EMPTY_WIKI_ROW_RE.search(value):
            return True
        if cls._EDIT_PLACEHOLDER_RE.search(value):
            return True
        lowered = value.casefold()
        if "wikidata" in lowered and cls._WIKIDATA_METADATA_RE.search(value):
            return True
        return "special:" in lowered and any(
            marker in lowered
            for marker in ("content link:", "/wiki/", "wikidata", "wikipedia")
        )

    @classmethod
    def _question_constraints(
        cls,
        question: str,
    ) -> tuple[list[tuple[int, int]], set[int], list[str]]:
        text = normalize_text(question)
        ranges: list[tuple[int, int]] = []
        consumed_years: set[int] = set()
        for pattern in (cls._YEAR_RANGE_RE, cls._COMPACT_YEAR_RANGE_RE):
            for match in pattern.finditer(text):
                start = int(match.group("start"))
                end = int(match.group("end"))
                lower, upper = sorted((start, end))
                ranges.append((lower, upper))
                consumed_years.update({start, end})
        years = {
            int(match.group(0))
            for match in cls._YEAR_RE.finditer(text)
            if int(match.group(0)) not in consumed_years
        }
        phrases = [
            normalize_text(match.group(1)).casefold()
            for match in cls._QUOTED_PHRASE_RE.finditer(text)
            if normalize_text(match.group(1))
        ]
        return ranges, years, phrases

    @classmethod
    def _constraint_priority(
        cls,
        text: str,
        constraints: tuple[list[tuple[int, int]], set[int], list[str]],
    ) -> int:
        ranges, years, phrases = constraints
        row_years = {int(match.group(0)) for match in cls._YEAR_RE.finditer(text)}
        if any(lower <= year <= upper for lower, upper in ranges for year in row_years):
            return 0
        lowered = normalize_text(text).casefold()
        if any(phrase in lowered for phrase in phrases) or bool(row_years & years):
            return 1
        return 2

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
