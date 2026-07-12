from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class QueryInfoTokenRecord:
    """
    Store the word-level input record used by the EfficientRAG filter.

    Args:
        - query_info_tokens: Query and Info tokens consumed by the filter.
        - query_info_labels: Optional gold keep labels for training data.
        - query_tokens: Tokens from the original question/query section.
        - info_tokens: Tokens extracted from useful Labeler spans.

    Returns:
        - QueryInfoTokenRecord: JSONL-compatible filter input record.
    """

    query_info_tokens: list[str]
    query_info_labels: list[bool] | None = None
    query_tokens: list[str] = field(default_factory=list)
    info_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.query_info_labels is None:
            data.pop("query_info_labels", None)
        return data


class FilterInputBuilder:
    """
    Build Query + Info token records between Labeler output and Filter input.

    Args:
        - max_query_tokens: Maximum tokens retained from the original query.
        - max_info_tokens: Maximum useful information tokens retained from Labeler spans.

    Returns:
        - FilterInputBuilder: Stateless Query + Info token formatter.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_.-]*|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        *,
        max_query_tokens: int = 64,
        max_info_tokens: int = 48,
    ) -> None:
        self.max_query_tokens = max(8, max_query_tokens)
        self.max_info_tokens = max(4, max_info_tokens)

    def build(
        self,
        *,
        query: str,
        info_items: list[str],
        query_info_labels: list[bool] | None = None,
    ) -> QueryInfoTokenRecord:
        """
        Build a JSONL-compatible Query + Info token record.

        Args:
            - query: Original user question or current retrieval query.
            - info_items: Useful spans selected by the Labeler from CONTINUE chunks.
            - query_info_labels: Optional word-level keep labels for supervised data.

        Returns:
            - QueryInfoTokenRecord: Query + Info tokens and optional labels.
        """
        query_tokens = self.tokenize(query)[: self.max_query_tokens]
        info_tokens = self._dedupe_tokens(
            token
            for item in info_items
            for token in self.tokenize(item)
            if token not in {",", ";", ":"}
        )[: self.max_info_tokens]
        tokens = ["Query", ":"] + query_tokens + ["Info", ":"] + info_tokens
        labels = None
        if query_info_labels is not None:
            labels = list(query_info_labels)[: len(tokens)]
            labels.extend([False] * (len(tokens) - len(labels)))
        return QueryInfoTokenRecord(
            query_info_tokens=tokens,
            query_info_labels=labels,
            query_tokens=query_tokens,
            info_tokens=info_tokens,
        )

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into word-level units aligned with filter labels.

        Args:
            - text: Raw query or useful information text.

        Returns:
            - list[str]: Word and punctuation tokens.
        """
        normalized = normalize_text(text)
        if not normalized:
            return []
        return [match.group(0) for match in self._TOKEN_RE.finditer(normalized)]

    def _dedupe_tokens(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = normalize_text(str(value or ""))
            key = token.casefold()
            if not token or key in seen:
                continue
            result.append(token)
            seen.add(key)
        return result


__all__ = ["FilterInputBuilder", "QueryInfoTokenRecord"]
