from __future__ import annotations

from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text

from .config import (
    EvidenceOutput,
    SearchQueryPlan,
    SearchSourceCandidate,
)
from .evidence_renderer import EvidenceRenderer
from .extractor import CandidateExtractor, EvidenceExtractor
from .search_query_planner import SearchQueryPlanner
from .search_source_filter import SourceFilter


class EvidenceSearcher:
    """
    Build evidence-oriented search output from planned queries and structured search.

    Args:
        - tool_manager: Object exposing execute_tool(tool_name, parameters, agent_id, stage).
        - query_planner: Optional SearchQueryPlanner instance.
        - source_filter: Optional SourceFilter instance.
        - evidence_extractor: Optional EvidenceExtractor instance.
        - candidate_extractor: Optional CandidateExtractor instance.
        - renderer: Optional EvidenceRenderer instance.

    Returns:
        - EvidenceSearcher: Search pipeline service that returns EvidenceOutput.
    """

    def __init__(
        self,
        *,
        tool_manager: Any,
        query_planner: SearchQueryPlanner | None = None,
        source_filter: SourceFilter | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        candidate_extractor: CandidateExtractor | None = None,
        renderer: EvidenceRenderer | None = None,
    ) -> None:
        self.tool_manager = tool_manager
        self.query_planner = query_planner or SearchQueryPlanner()
        self.source_filter = source_filter or SourceFilter()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor()
        self.candidate_extractor = candidate_extractor or CandidateExtractor()
        self.renderer = renderer or EvidenceRenderer()

    def search(
        self,
        question: str,
        *,
        max_queries: int = 3,
        max_results_per_query: int = 5,
        max_full_page_results: int = 2,
        agent_id: str = "evidence_searcher",
        stage: str = "search_evidence",
    ) -> EvidenceOutput:
        """
        Execute planned search queries and build an EvidenceOutput bundle.

        Args:
            - question: Original task question.
            - max_queries: Maximum planned queries to execute.
            - max_results_per_query: Maximum search results per query.
            - max_full_page_results: Maximum full pages fetched per query.
            - agent_id: Trace agent id used by ToolManager.
            - stage: Trace stage used by ToolManager.

        Returns:
            - EvidenceOutput: Structured evidence bundle for rendering or scoring.
        """
        plan_dict = self.query_planner.plan(question=question, max_queries=max_queries)
        queries = self._build_query_plans(plan_dict)
        if not queries:
            queries = [
                SearchQueryPlan(
                    query_id="Q1",
                    query=question,
                    purpose="original_question",
                    priority=100,
                    requires_full_page=True,
                )
            ]

        search_runs: list[dict[str, Any]] = []
        tool_usage: list[dict[str, Any]] = []
        for query_plan in queries:
            result = self._execute_search(
                query_plan=query_plan,
                precision_needed=bool(plan_dict.get("precision_needed")),
                max_results=max_results_per_query,
                max_full_page_results=max_full_page_results,
                agent_id=agent_id,
                stage=stage,
            )
            tool_usage.append(result)
            search_runs.append(
                {
                    "query_id": query_plan.query_id,
                    "query": query_plan.query,
                    "result": result,
                }
            )

        sources = self._sources_from_runs(search_runs)
        filtered_sources = self.source_filter.filter_sources(sources)
        blocked_sources = [source for source in sources if source.blocked]
        evidence_items = self.evidence_extractor.extract(
            question=question,
            sources=filtered_sources,
        )
        candidates = self.candidate_extractor.extract_candidates(
            question=question,
            evidence_items=evidence_items,
            sources=filtered_sources,
            max_candidates=5,
        )

        output = EvidenceOutput(
            question=question,
            queries=queries,
            sources=filtered_sources,
            evidence_items=evidence_items,
            candidates=candidates,
            summary="",
            tool_usage=tool_usage,
            blocked_sources=blocked_sources,
        )
        output.summary = self.renderer.render(output)
        return output

    def render(self, output: EvidenceOutput) -> str:
        """
        Render an existing EvidenceOutput bundle.

        Args:
            - output: Evidence-oriented search result bundle.

        Returns:
            - str: Prompt-ready evidence context.
        """
        return self.renderer.render(output)

    def to_dict(self, output: EvidenceOutput) -> dict[str, Any]:
        """
        Convert EvidenceOutput to a JSON-serializable dict.

        Args:
            - output: Evidence-oriented search result bundle.

        Returns:
            - dict[str, Any]: JSON-serializable evidence payload.
        """
        return asdict(output)

    def _build_query_plans(self, plan_dict: dict[str, Any]) -> list[SearchQueryPlan]:
        source_hints = list(plan_dict.get("source_hints") or [])
        precision_needed = bool(plan_dict.get("precision_needed"))
        query_values = list(plan_dict.get("queries") or [])
        plans: list[SearchQueryPlan] = []
        for index, query in enumerate(query_values, start=1):
            query = normalize_text(query)
            if not query:
                continue
            plans.append(
                SearchQueryPlan(
                    query_id=f"Q{index}",
                    query=query,
                    purpose=self._query_purpose(index),
                    priority=100 - index,
                    source_hints=source_hints,
                    expected_answer_type=self._expected_answer_type(query),
                    requires_full_page=precision_needed,
                )
            )
        return plans

    def _execute_search(
        self,
        *,
        query_plan: SearchQueryPlan,
        precision_needed: bool,
        max_results: int,
        max_full_page_results: int,
        agent_id: str,
        stage: str,
    ) -> dict[str, Any]:
        try:
            return self.tool_manager.execute_tool(
                "search",
                {
                    "input": query_plan.query,
                    "mode": "structured",
                    "conditional_fetch": precision_needed or query_plan.requires_full_page,
                    "max_results": max_results,
                    "max_full_page_results": max_full_page_results,
                },
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            return {
                "ok": False,
                "tool_name": "search",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }

    def _sources_from_runs(self, search_runs: list[dict[str, Any]]) -> list[SearchSourceCandidate]:
        sources: list[SearchSourceCandidate] = []
        for run in search_runs:
            query_id = str(run.get("query_id", "") or "")
            result = run.get("result") or {}
            payload = result.get("raw_result")
            if not isinstance(payload, dict):
                continue
            raw_items = payload.get("results") or []
            if not isinstance(raw_items, list):
                continue

            for rank, item in enumerate(raw_items, start=1):
                if not isinstance(item, dict):
                    continue
                url = normalize_text(item.get("url", ""))
                title = normalize_text(item.get("title", "")) or url
                snippet = normalize_text(item.get("content", ""))
                raw_content = normalize_text(item.get("raw_content", ""))
                source_id = f"S{len(sources) + 1}"
                sources.append(
                    SearchSourceCandidate(
                        source_id=source_id,
                        query_id=query_id,
                        title=title,
                        url=url,
                        domain=self._domain(url),
                        snippet=snippet,
                        raw_content=raw_content,
                        rank=rank,
                        rerank_score=float(item.get("rerank_score", 0.0) or 0.0),
                        fetched=bool(raw_content),
                    )
                )
        return sources

    def _query_purpose(self, index: int) -> str:
        if index == 1:
            return "core_question"
        if index == 2:
            return "quoted_or_entity_focus"
        return "source_or_keyword_focus"

    def _expected_answer_type(self, query: str) -> str:
        lowered = query.lower()
        if any(marker in lowered for marker in ("when", "date", "year")):
            return "date"
        if "where" in lowered:
            return "place"
        if "who" in lowered:
            return "person"
        if any(marker in lowered for marker in ("title", "book", "paper", "video")):
            return "title"
        if any(marker in lowered for marker in ("url", "website")):
            return "website"
        return "entity"

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""


__all__ = ["EvidenceSearcher"]
