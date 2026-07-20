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
            if url_key and url_key in seen_urls:
                continue
            if content_key in seen_content:
                continue
            if domain and domain_counts.get(domain, 0) >= self.max_items_per_domain:
                continue

            remaining = self.max_total_chars - total_chars
            if remaining < self.min_chars:
                break
            text = self._truncate(text, min(self.max_chars_per_item, remaining))
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
