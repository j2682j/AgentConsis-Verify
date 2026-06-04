from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text

from .mask_salience_query import MaskSalienceQueryGenerator


class SearchQueryPlanner:
    """
    使用 token salience 產生搜尋 query，失敗時退回原始問題。

    Args:
        - generator: 可注入的 MaskSalienceQueryGenerator。
        - precision_needed: 是否要求後續 search pipeline 優先抓取較完整內容。

    Returns:
        - SearchQueryPlanner: query planning 入口。
    """

    def __init__(
        self,
        *,
        generator: MaskSalienceQueryGenerator | None = None,
        precision_needed: bool = True,
    ) -> None:
        self.generator = generator or MaskSalienceQueryGenerator()
        self.precision_needed = precision_needed

    def plan(self, question: str, max_queries: int = 5) -> dict[str, Any]:
        """
        依照原始問題產生排序後的 query list。

        Args:
            - question: 原始問題。
            - max_queries: 最多 query 數量。

        Returns:
            - dict[str, Any]: query plan，包含 queries 與 precision_needed。
        """
        text = normalize_text(question)
        if not text:
            return {"queries": [], "precision_needed": False}

        try:
            candidates = self.generator.generate(text, num_candidates=max_queries)
            queries = self._dedupe_queries([candidate.query for candidate in candidates])[: max(1, max_queries)]
            if queries:
                return {
                    "queries": queries,
                    "precision_needed": self.precision_needed,
                    "salient_spans": [span.text for span in self.generator.last_salient_spans],
                }
        except Exception as exc:
            return {
                "queries": [text],
                "precision_needed": self.precision_needed,
                "salient_spans": [],
                "planner_error": f"mask_salience:{type(exc).__name__}: {exc}",
            }

        return {
            "queries": [text],
            "precision_needed": self.precision_needed,
            "salient_spans": [],
        }

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            cleaned = normalize_text(query)
            key = self._normalize_query_key(cleaned)
            if not cleaned or not key or key in seen:
                continue
            deduped.append(cleaned)
            seen.add(key)
        return deduped

    def _normalize_query_key(self, query: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(query).lower()).strip()


QueryGenerator = SearchQueryPlanner

__all__ = ["QueryGenerator", "SearchQueryPlanner"]
