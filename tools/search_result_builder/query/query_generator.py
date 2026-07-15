from __future__ import annotations

import os
import re
from typing import Any

from core.llm_client import LLMClient
from utils.network_utils import normalize_text

from .mask_salience_query import MaskSalienceQueryGenerator
from .query_coverage import QueryCoverageChecker
from .search_intent_planner import SearchIntentPlanner


class QueryGenerator:
    """
    使用 token salience 產生搜尋 query，失敗時退回原始問題。

    Args:
        - generator: 可注入的 MaskSalienceQueryGenerator。
        - query_model_name: 預設 generator 使用的 served model name。
        - llm_client: 預設 generator 使用的 provider-neutral client。
        - precision_needed: 是否要求後續 search pipeline 優先抓取較完整內容。

    Returns:
        - QueryGenerator: query planning 入口。
    """

    def __init__(
        self,
        *,
        generator: MaskSalienceQueryGenerator | None = None,
        coverage_checker: QueryCoverageChecker | None = None,
        intent_planner: SearchIntentPlanner | None = None,
        query_model_name: str | None = None,
        llm_client: LLMClient | None = None,
        precision_needed: bool = True,
    ) -> None:
        resolved_query_model = query_model_name or os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b")
        self.query_model_name = resolved_query_model
        self.generator = generator or MaskSalienceQueryGenerator(
            query_model_name=resolved_query_model,
            llm_client=llm_client,
        )
        self.coverage_checker = coverage_checker or QueryCoverageChecker()
        self.intent_planner = intent_planner or SearchIntentPlanner(
            model_name=resolved_query_model,
            llm_client=llm_client,
        )
        self.precision_needed = precision_needed

    def plan(self, question: str, max_queries: int = 5) -> dict[str, Any]:
        try:
            return self._plan_impl(question, max_queries=max_queries)
        finally:
            self._stop_query_model_after_generation()

    def _plan_impl(self, question: str, max_queries: int = 5) -> dict[str, Any]:
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

        intent_plan = self.intent_planner.plan(text)
        if not intent_plan.search_needed:
            return {
                "queries": [],
                "precision_needed": False,
                "salient_spans": [],
                "query_candidates": [],
                "query_coverage": {
                    "method": "intent_plan",
                    "search_needed": False,
                    "intent_plan": intent_plan.to_dict(),
                },
                "search_intent_plan": intent_plan.to_dict(),
            }

        try:
            try:
                candidates = self.generator.generate(
                    text,
                    num_candidates=max_queries,
                    intent_plan=intent_plan,
                )
            except TypeError:
                candidates = self.generator.generate(
                    text,
                    num_candidates=max_queries,
                )
            raw_queries = self._dedupe_queries([candidate.query for candidate in candidates])
            salient_spans = list(self.generator.last_salient_spans)
            queries, coverage = self.coverage_checker.improve_queries(
                question=text,
                queries=raw_queries,
                salient_spans=salient_spans,
                intent_plan=intent_plan,
                max_queries=max_queries,
            )
            queries = queries[: max(1, max_queries)]
            if queries:
                return {
                    "queries": queries,
                    "precision_needed": self.precision_needed,
                    "salient_spans": [span.text for span in salient_spans],
                    "query_candidates": [
                        {
                            "query": candidate.query,
                            "matched_spans": candidate.matched_spans,
                            "salience_coverage_score": candidate.coverage_score,
                            "semantic_impact_score": candidate.semantic_impact_score,
                            "source": candidate.source,
                        }
                        for candidate in candidates
                    ],
                    "query_coverage": coverage,
                    "search_intent_plan": intent_plan.to_dict(),
                }
        except Exception as exc:
            queries, coverage = self.coverage_checker.improve_queries(
                question=text,
                queries=[text],
                salient_spans=[],
                intent_plan=intent_plan,
                max_queries=max_queries,
            )
            return {
                "queries": queries or [text],
                "precision_needed": self.precision_needed,
                "salient_spans": [],
                "query_coverage": coverage,
                "search_intent_plan": intent_plan.to_dict(),
                "planner_error": f"mask_salience:{type(exc).__name__}: {exc}",
            }

        queries, coverage = self.coverage_checker.improve_queries(
            question=text,
            queries=[text],
            salient_spans=[],
            intent_plan=intent_plan,
            max_queries=max_queries,
        )
        return {
            "queries": queries or [text],
            "precision_needed": self.precision_needed,
            "salient_spans": [],
            "query_coverage": coverage,
            "search_intent_plan": intent_plan.to_dict(),
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

    def _stop_query_model_after_generation(self) -> None:
        return


__all__ = ["QueryGenerator"]
