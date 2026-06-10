from __future__ import annotations

from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text

from .config import CandidateAnswer, EvidenceOutput, SearchQueryPlan, SearchSignals, SearchSourceCandidate
from .next_hop_query import EfficientRAGFilterAdapter, NextHopQueryGenerator, RetrievalController
from .query import QueryGenerator
from .evidence_renderer import EvidenceRenderer
from .source_analyze import SourceAnalysis, SourceUsefulnessResult


class EvidenceSearcher:
    """
    以精簡 search pipeline 建立可傳給 Agent 的 evidence context。

    Args:
        - tool_manager: 提供 execute_tool(tool_name, parameters, agent_id, stage) 的工具管理器。
        - query_planner: 第一跳 query generator。
        - source_analysis: 負責 source filter、full-page fetch、label、dedup 與 helpfulness scoring。
        - retrieval_controller: 判斷目前 evidence 是否需要 next-hop retrieval。
        - next_hop_query_generator: 根據目前 evidence 建立下一跳 query。
        - renderer: 將 EvidenceOutput 轉成 prompt context 的 renderer。

    Returns:
        - EvidenceSearcher: 精簡後的 search evidence 主控器。
    """

    def __init__(
        self,
        *,
        tool_manager: Any,
        query_planner: QueryGenerator | None = None,
        source_analysis: SourceAnalysis | None = None,
        retrieval_controller: RetrievalController | None = None,
        rag_filter: EfficientRAGFilterAdapter | None = None,
        next_hop_query_generator: NextHopQueryGenerator | None = None,
        renderer: EvidenceRenderer | None = None,
        enable_evidence_driven_search: bool = True,
        max_evidence_driven_queries: int = 2,
        **_: Any,
    ) -> None:
        self.tool_manager = tool_manager
        self.query_planner = query_planner or QueryGenerator()
        self.source_analysis = source_analysis or SourceAnalysis()
        self.retrieval_controller = retrieval_controller or RetrievalController()
        self.rag_filter = rag_filter or EfficientRAGFilterAdapter()
        self.next_hop_query_generator = next_hop_query_generator or NextHopQueryGenerator()
        self.renderer = renderer or EvidenceRenderer()
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.max_evidence_driven_queries = max(0, max_evidence_driven_queries)

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
        執行 query generation、SEER cleaning 與可選 next-hop retrieval。

        Args:
            - question: 原始任務問題。
            - max_queries: 第一跳最多執行的 query 數量。
            - max_results_per_query: 每個 query 最多保留的搜尋結果數。
            - max_full_page_results: 每個 query 對應的全文抓取上限。
            - agent_id: ToolManager trace 用 agent id。
            - stage: ToolManager trace 用 stage。

        Returns:
            - EvidenceOutput: 結構化 search evidence 結果。
        """
        plan_dict = self.query_planner.plan(question=question, max_queries=max_queries)
        search_signals = self._build_search_signals(plan_dict)
        initial_queries = self._build_query_plans(plan_dict, fallback_question=question)

        tool_usage: list[dict[str, Any]] = []
        initial_sources = self._search_sources(
            queries=initial_queries,
            max_results=max_results_per_query,
            agent_id=agent_id,
            stage=stage,
            source_prefix="S",
            tool_usage=tool_usage,
        )
        initial_seer = self._build_seer_result(
            question=question,
            sources=initial_sources,
            queries=initial_queries,
            fetch_limit=max(0, max_full_page_results) * max(1, len(initial_queries)),
            max_pages=max(0, max_full_page_results) * max(1, len(initial_queries)),
            max_candidates=10,
        )
        initial_source_diagnostics = dict(self.source_analysis.last_diagnostics)
        initial_source_diagnostics["fetched_pages"] = initial_seer.fetched_pages
        all_queries = list(initial_queries)
        all_sources = list(initial_seer.sources)
        blocked_sources = list(self.source_analysis.last_blocked_sources)
        evidence_items = list(self.source_analysis.last_evidence_items)
        candidates = list(self.source_analysis.last_candidates)

        initial_decision = self.retrieval_controller.assess(
            evidence_items=evidence_items,
            candidates=candidates,
        )
        follow_up_queries: list[SearchQueryPlan] = []
        follow_up_source_diagnostics: dict[str, Any] = {}

        if self.enable_evidence_driven_search and initial_decision.need_next_hop:
            follow_up_queries = self._build_follow_up_queries(
                question=question,
                search_signals=search_signals,
                initial_queries=initial_queries,
                evidence_items=evidence_items,
                candidates=candidates,
                max_queries=self.max_evidence_driven_queries,
            )
            if follow_up_queries:
                follow_up_sources = self._search_sources(
                    queries=follow_up_queries,
                    max_results=max(2, min(max_results_per_query, 3)),
                    agent_id=agent_id,
                    stage=f"{stage}_efficient_rag_followup",
                    source_prefix="HS",
                    tool_usage=tool_usage,
                )
                follow_up_fetch_limit = (
                    max(1, min(max_full_page_results, 1)) * max(1, len(follow_up_queries))
                    if max_full_page_results > 0
                    else 0
                )
                follow_up_seer = self._build_seer_result(
                    question=question,
                    sources=follow_up_sources,
                    queries=follow_up_queries,
                    fetch_limit=follow_up_fetch_limit,
                    max_pages=follow_up_fetch_limit,
                    max_evidence_items=12,
                    max_candidates=10,
                )
                follow_up_source_diagnostics = dict(self.source_analysis.last_diagnostics)
                follow_up_source_diagnostics["fetched_pages"] = follow_up_seer.fetched_pages
                all_queries.extend(follow_up_queries)
                all_sources.extend(follow_up_seer.sources)
                blocked_sources.extend(self.source_analysis.last_blocked_sources)
                evidence_items = self._merge_evidence(
                    evidence_items,
                    list(self.source_analysis.last_evidence_items),
                )
                candidates = self._merge_candidates(
                    candidates,
                    list(self.source_analysis.last_candidates),
                )

        diagnostics = {
            "initial_source_analysis": initial_source_diagnostics,
            "initial_retrieval_decision": asdict(initial_decision),
            "evidence_driven_search": {
                "enabled": self.enable_evidence_driven_search,
                "triggered": bool(follow_up_queries),
                "queries": [query.query for query in follow_up_queries],
                "source_analysis": follow_up_source_diagnostics,
            },
            "final_counts": {
                "query_count": len(all_queries),
                "source_count": len(all_sources),
                "evidence_count": len(evidence_items),
                "candidate_count": len(candidates),
                "blocked_source_count": len(blocked_sources),
            },
        }
        output = EvidenceOutput(
            question=question,
            queries=all_queries,
            sources=all_sources,
            evidence_items=evidence_items,
            summary="",
            candidates=candidates,
            search_signals=search_signals,
            candidate_diagnostics=diagnostics,
            tool_usage=tool_usage,
            blocked_sources=blocked_sources,
        )
        output.summary = self.renderer.render(output)
        return output

    def render(self, output: EvidenceOutput) -> str:
        """
        將既有 EvidenceOutput 轉成 prompt-ready context。

        Args:
            - output: Search evidence 結構化輸出。

        Returns:
            - str: 可傳給 Agent 的 evidence context。
        """
        return self.renderer.render(output)

    def to_dict(self, output: EvidenceOutput) -> dict[str, Any]:
        """
        將 EvidenceOutput 轉成 JSON-serializable dict。

        Args:
            - output: Search evidence 結構化輸出。

        Returns:
            - dict[str, Any]: 可寫入 log 的 dict。
        """
        return asdict(output)

    def _build_query_plans(
        self,
        plan_dict: dict[str, Any],
        *,
        fallback_question: str,
    ) -> list[SearchQueryPlan]:
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
                    expected_answer_type="unknown",
                    requires_full_page=precision_needed,
                )
            )
        if not plans:
            plans.append(
                SearchQueryPlan(
                    query_id="Q1",
                    query=fallback_question,
                    purpose="original_question",
                    priority=100,
                    expected_answer_type="unknown",
                    requires_full_page=True,
                )
            )
        return plans

    def _build_follow_up_queries(
        self,
        *,
        question: str,
        search_signals: SearchSignals,
        initial_queries: list[SearchQueryPlan],
        evidence_items: list[Any],
        candidates: list[CandidateAnswer],
        max_queries: int,
    ) -> list[SearchQueryPlan]:
        if max_queries <= 0:
            return []

        initial_keys = {self._query_key(plan.query) for plan in initial_queries}
        plans: list[SearchQueryPlan] = []

        filter_result = self.rag_filter.build_query(
            question=question,
            evidence_items=evidence_items,
        )
        filter_query = normalize_text(filter_result.query)
        if filter_query and self._query_key(filter_query) not in initial_keys:
            plans.append(
                SearchQueryPlan(
                    query_id="H1",
                    query=filter_query,
                    purpose="efficientrag_filtered_followup",
                    priority=90,
                    source_hints=[],
                    expected_answer_type=search_signals.answer_type,
                    requires_full_page=True,
                )
            )

        remaining = max_queries - len(plans)
        if remaining <= 0:
            return plans

        generated = self.next_hop_query_generator.build(
            question=question,
            search_signals=search_signals,
            initial_queries=[*initial_queries, *plans],
            evidence_items=evidence_items,
            candidates=candidates,
            max_queries=remaining,
        )
        for generated_plan in generated:
            key = self._query_key(generated_plan.query)
            if not key or key in {self._query_key(plan.query) for plan in [*initial_queries, *plans]}:
                continue
            generated_plan.query_id = f"H{len(plans) + 1}"
            generated_plan.priority = min(generated_plan.priority, 80 - len(plans))
            plans.append(generated_plan)
            if len(plans) >= max_queries:
                break
        return plans

    def _build_search_signals(self, plan_dict: dict[str, Any]) -> SearchSignals:
        salient_spans = [
            normalize_text(str(span))
            for span in list(plan_dict.get("salient_spans") or [])
            if normalize_text(str(span))
        ]
        return SearchSignals(
            answer_type="unknown",
            target_terms=self._dedupe_texts(salient_spans)[:8],
            constraints=[],
            source_hints=[],
            needs_multi_hop=False,
        )

    def _dedupe_texts(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value).strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                deduped.append(cleaned)
                seen.add(key)
        return deduped

    def _search_sources(
        self,
        *,
        queries: list[SearchQueryPlan],
        max_results: int,
        agent_id: str,
        stage: str,
        source_prefix: str,
        tool_usage: list[dict[str, Any]],
    ) -> list[SearchSourceCandidate]:
        search_runs: list[dict[str, Any]] = []
        for query_plan in queries:
            result = self._execute_search(
                query_plan=query_plan,
                max_results=max_results,
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
        return self._sources_from_runs(search_runs, source_prefix=source_prefix)

    def _execute_search(
        self,
        *,
        query_plan: SearchQueryPlan,
        max_results: int,
        agent_id: str,
        stage: str,
    ) -> dict[str, Any]:
        try:
            return self.tool_manager.execute_tool(
                "search",
                {
                    "input": query_plan.query,
                    "mode": "structured",
                    "max_results": max_results,
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

    def _build_seer_result(
        self,
        *,
        question: str,
        sources: list[SearchSourceCandidate],
        queries: list[SearchQueryPlan],
        fetch_limit: int,
        max_pages: int,
        max_evidence_items: int = 8,
        max_candidates: int = 10,
    ) -> SourceUsefulnessResult:
        return self.source_analysis.build(
            question=question,
            sources=sources,
            query_text_by_id={query.query_id: query.query for query in queries},
            fetch_limit=fetch_limit,
            max_pages=max_pages,
            max_evidence_items=max_evidence_items,
            max_chars_per_item=420,
            max_candidates=max_candidates,
        )

    def _sources_from_runs(
        self,
        search_runs: list[dict[str, Any]],
        *,
        source_prefix: str = "S",
    ) -> list[SearchSourceCandidate]:
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
                fetched = bool(
                    raw_content
                    and raw_content != snippet
                    and len(raw_content) > len(snippet) + 120
                )
                sources.append(
                    SearchSourceCandidate(
                        source_id=f"{source_prefix}{len(sources) + 1}",
                        query_id=query_id,
                        title=title,
                        url=url,
                        domain=self._domain(url),
                        snippet=snippet,
                        raw_content=raw_content,
                        rank=rank,
                        fetched=fetched,
                    )
                )
        return sources

    def _merge_evidence(
        self,
        initial: list[Any],
        follow_up: list[Any],
    ) -> list[Any]:
        merged = [*initial, *follow_up]
        for index, item in enumerate(merged, start=1):
            item.evidence_id = f"E{index}"
        return merged

    def _merge_candidates(
        self,
        initial: list[CandidateAnswer],
        follow_up: list[CandidateAnswer],
    ) -> list[CandidateAnswer]:
        grouped: dict[str, CandidateAnswer] = {}
        for candidate in [*initial, *follow_up]:
            key = normalize_text(candidate.answer).lower()
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = candidate
                continue
            existing.support_count += candidate.support_count
            existing.confidence = max(existing.confidence, candidate.confidence)
            existing.evidence_ids = sorted(set(existing.evidence_ids + candidate.evidence_ids))
            existing.source_ids = sorted(set(existing.source_ids + candidate.source_ids))
        candidates = list(grouped.values())
        candidates.sort(
            key=lambda item: (
                item.support_count,
                item.confidence,
            ),
            reverse=True,
        )
        return candidates

    def _query_purpose(self, index: int) -> str:
        if index == 1:
            return "core_question"
        if index == 2:
            return "quoted_or_entity_focus"
        return "source_or_keyword_focus"

    def _query_key(self, value: str) -> str:
        return normalize_text(value).lower().strip(" \"'`.,;:-")

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""


__all__ = ["EvidenceSearcher"]
