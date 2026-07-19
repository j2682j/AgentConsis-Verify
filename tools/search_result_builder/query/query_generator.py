from __future__ import annotations

import os
import re
from typing import Any

from core.llm_client import LLMClient
from utils.network_utils import normalize_text

from .mask_salience_query import MaskSalienceQueryGenerator
from .query_coverage import QueryCoverageChecker
from .relation_plan import RelationPlan
from .search_intent_plan import SearchIntentPlan
from .source_requirement import SearchQueryRequest
from .span_classifier import ClassifiedSpan


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
            return {
                "queries": [],
                "query_requests": [],
                "precision_needed": False,
                "salient_spans": [],
                "query_candidates": [],
                "query_coverage": {
                    "method": "empty_question",
                    "search_needed": False,
                },
                "search_intent_plan": SearchIntentPlan(search_needed=False).to_dict(),
                "relation_plan": RelationPlan().to_dict(),
            }

        try:
            try:
                candidates = self.generator.generate(
                    text,
                    num_candidates=max_queries,
                )
            except TypeError:
                candidates = self.generator.generate(
                    text,
                    num_candidates=max_queries,
                )
            raw_queries = self._dedupe_queries([candidate.query for candidate in candidates])
            salient_spans = list(self.generator.last_salient_spans)
            classified_spans = list(getattr(self.generator, "last_classified_spans", []))
            relation_plan = getattr(self.generator, "last_relation_plan", RelationPlan())
            relation_validation = getattr(
                self.generator,
                "last_relation_plan_validation",
                None,
            )
            query_state = self._query_state_from_classified_spans(
                classified_spans,
                relation_plan=relation_plan,
            )
            _, coverage = self.coverage_checker.improve_queries(
                question=text,
                queries=raw_queries,
                salient_spans=salient_spans,
                intent_plan=query_state,
                max_queries=max_queries,
            )
            queries = raw_queries[: max(1, max_queries)]
            if queries:
                selected_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.query in queries
                ][: max(1, max_queries)]
                return {
                    "queries": queries,
                    "query_requests": [
                        SearchQueryRequest(
                            query=candidate.query,
                            source_requirement=candidate.source_requirement,
                        ).to_dict()
                        for candidate in selected_candidates
                    ],
                    "precision_needed": self.precision_needed,
                    "salient_spans": [span.text for span in salient_spans],
                    "query_candidates": [
                        {
                            "query": candidate.query,
                            "matched_spans": candidate.matched_spans,
                            "salience_coverage_score": candidate.coverage_score,
                            "semantic_impact_score": candidate.semantic_impact_score,
                            "source": candidate.source,
                            "source_requirement": candidate.source_requirement.to_dict(),
                        }
                        for candidate in candidates
                    ],
                    "classified_spans": [
                        span.to_dict() if hasattr(span, "to_dict") else dict(span)
                        for span in classified_spans
                    ],
                    "query_coverage": coverage,
                    "query_state": query_state.to_dict(),
                    "search_intent_plan": query_state.to_dict(),
                    "relation_plan": relation_plan.to_dict(),
                    "relation_plan_validation": (
                        relation_validation.to_dict()
                        if relation_validation is not None
                        else {"valid": True, "errors": [], "repairs": []}
                    ),
                    "intent_planning": "disabled",
                }
        except Exception as exc:
            query_state = self._fallback_query_state(text)
            _, coverage = self.coverage_checker.improve_queries(
                question=text,
                queries=[text],
                salient_spans=[],
                intent_plan=query_state,
                max_queries=max_queries,
            )
            return {
                "queries": [text],
                "query_requests": [SearchQueryRequest.fallback(text).to_dict()],
                "precision_needed": self.precision_needed,
                "salient_spans": [],
                "query_coverage": coverage,
                "query_state": query_state.to_dict(),
                "search_intent_plan": query_state.to_dict(),
                "relation_plan": query_state.relation_plan.to_dict(),
                "intent_planning": "disabled",
                "planner_error": f"mask_salience:{type(exc).__name__}: {exc}",
            }

        query_state = self._fallback_query_state(text)
        _, coverage = self.coverage_checker.improve_queries(
            question=text,
            queries=[text],
            salient_spans=[],
            intent_plan=query_state,
            max_queries=max_queries,
        )
        return {
            "queries": [text],
            "query_requests": [SearchQueryRequest.fallback(text).to_dict()],
            "precision_needed": self.precision_needed,
            "salient_spans": [],
            "query_coverage": coverage,
            "query_state": query_state.to_dict(),
            "search_intent_plan": query_state.to_dict(),
            "relation_plan": query_state.relation_plan.to_dict(),
            "intent_planning": "disabled",
        }

    def _query_state_from_classified_spans(
        self,
        classified_spans: list[ClassifiedSpan],
        *,
        relation_plan: RelationPlan | None = None,
    ) -> SearchIntentPlan:
        grouped: dict[str, list[ClassifiedSpan]] = {}
        for span in classified_spans:
            grouped.setdefault(span.role, []).append(span)
        for spans in grouped.values():
            spans.sort(key=lambda item: (item.score, item.confidence), reverse=True)

        source_clues = self._top_role_terms(grouped.get("source_clue", []), limit=5)
        constraints = self._top_role_terms(grouped.get("constraint", []), limit=4)
        answer_targets = self._top_role_terms(grouped.get("answer_target", []), limit=2)
        avoid_terms = self._top_role_terms(
            [
                *grouped.get("format_instruction", []),
                *grouped.get("weak_generic", []),
            ],
            limit=5,
        )
        must_include = self._unique_terms([*source_clues, *constraints], limit=6)
        target = " ".join(self._unique_terms([*answer_targets, *source_clues], limit=4))
        return SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target=target,
            must_include=must_include,
            avoid_terms=avoid_terms,
            preferred_domain="",
            answer_role=answer_targets[0].lower() if answer_targets else "unknown",
            state="pending",
            completed_terms=[],
            missing_terms=must_include,
            source="embedding_span_role_classifier",
            relation_plan=relation_plan or RelationPlan(),
        )

    def _top_role_terms(self, spans: list[ClassifiedSpan], *, limit: int) -> list[str]:
        return self._unique_terms([span.text for span in spans], limit=limit)

    def _unique_terms(self, terms: list[str], *, limit: int) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = normalize_text(str(term or ""))
            key = self._normalize_query_key(cleaned)
            if not cleaned or not key or key in seen:
                continue
            output.append(cleaned)
            seen.add(key)
            if len(output) >= limit:
                break
        return output

    def _fallback_query_state(self, question: str) -> SearchIntentPlan:
        return SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target=normalize_text(question)[:160],
            must_include=[],
            avoid_terms=[],
            preferred_domain="",
            answer_role="unknown",
            state="pending",
            completed_terms=[],
            missing_terms=[],
            source="fallback_question",
        )

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
