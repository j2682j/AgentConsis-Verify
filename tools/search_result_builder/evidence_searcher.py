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
from .analysis import QuestionAnalyzer
from .compression import EvidenceCompressor
from .extraction import EvidenceExtractor, QuestionTermFilter, TypedCandidateExtractor
from .filtering import SourceFilter
from .planning import CandidateVerificationSearcher, SearchQueryPlanner
from .reranking import ProbabilityCandidateReranker
from .rendering import AgentEvidenceRenderer, EvidenceRenderer
from .verification import CandidateVerifier


class EvidenceSearcher:
    """
    Build evidence-oriented search output from planned queries and structured search.

    Args:
        - tool_manager: Object exposing execute_tool(tool_name, parameters, agent_id, stage).
        - query_planner: Optional SearchQueryPlanner instance.
        - source_filter: Optional SourceFilter instance.
        - evidence_extractor: Optional EvidenceExtractor instance.
        - candidate_extractor: Optional TypedCandidateExtractor instance.
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
        candidate_extractor: TypedCandidateExtractor | None = None,
        candidate_verifier: CandidateVerifier | None = None,
        evidence_compressor: EvidenceCompressor | None = None,
        renderer: EvidenceRenderer | None = None,
        agent_renderer: AgentEvidenceRenderer | None = None,
        question_analyzer: QuestionAnalyzer | None = None,
        question_term_filter: QuestionTermFilter | None = None,
        candidate_verification_searcher: CandidateVerificationSearcher | None = None,
        probability_candidate_reranker: ProbabilityCandidateReranker | None = None,
        compact_evidence: bool = False,
        enable_signal_query_planner: bool = False,
        enable_probability_candidate_rerank: bool = False,
    ) -> None:
        self.tool_manager = tool_manager
        self.query_planner = query_planner or SearchQueryPlanner(
            mode="signal" if enable_signal_query_planner else "legacy"
        )
        self.source_filter = source_filter or SourceFilter()
        self.evidence_extractor = evidence_extractor or EvidenceExtractor()
        self.candidate_extractor = candidate_extractor or TypedCandidateExtractor()
        self.candidate_verifier = candidate_verifier or CandidateVerifier()
        self.evidence_compressor = evidence_compressor or EvidenceCompressor()
        self.renderer = renderer or EvidenceRenderer()
        self.agent_renderer = agent_renderer or AgentEvidenceRenderer()
        self.question_analyzer = question_analyzer or QuestionAnalyzer()
        self.question_term_filter = question_term_filter or QuestionTermFilter()
        self.candidate_verification_searcher = candidate_verification_searcher or CandidateVerificationSearcher(
            tool_manager=tool_manager,
            domain_parser=self._domain,
        )
        self.probability_candidate_reranker = probability_candidate_reranker or ProbabilityCandidateReranker()
        self.compact_evidence = compact_evidence
        self.enable_probability_candidate_rerank = enable_probability_candidate_rerank

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
        analysis = self.question_analyzer.analyze(question)
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
        raw_candidates = self.candidate_extractor.extract_candidates(
            question=question,
            analysis=analysis,
            evidence_items=evidence_items,
            sources=filtered_sources,
            max_candidates=5 if self.enable_probability_candidate_rerank else 10,
        )
        candidates, rejected_candidates = self.question_term_filter.filter(
            question=question,
            analysis=analysis,
            candidates=raw_candidates,
        )
        rerank_diagnostics: dict[str, Any] = {"enabled": False}
        if self.enable_probability_candidate_rerank:
            candidates, rerank_diagnostics = self.probability_candidate_reranker.rerank(
                question=question,
                candidates=candidates,
                evidence_items=evidence_items,
            )

        verification_results, verification_tool_usage = self.candidate_verification_searcher.verify_candidates(
            question=question,
            analysis=analysis,
            candidates=candidates[:5],
            max_candidates=5,
            max_results_per_query=3,
            max_full_page_results=1,
            agent_id=agent_id,
            stage=f"{stage}_candidate_verification",
        )
        tool_usage.extend(verification_tool_usage)

        verification_sources: list[SearchSourceCandidate] = []
        verification_queries: list[SearchQueryPlan] = []
        for verification in verification_results:
            verification_queries.extend(verification.queries)
            verification_sources.extend(verification.sources)

        filtered_verification_sources = self.source_filter.filter_sources(verification_sources)
        blocked_sources.extend(source for source in verification_sources if source.blocked)
        verification_evidence = self.evidence_extractor.extract(
            question=question,
            sources=filtered_verification_sources,
            max_items=12,
            max_chars_per_item=360,
        )
        for index, item in enumerate(verification_evidence, start=1):
            item.evidence_id = f"V{index}"

        all_sources = [*filtered_sources, *filtered_verification_sources]
        all_evidence = [*evidence_items, *verification_evidence]
        verified_candidates, fact_cards = self.candidate_verifier.verify(
            question=question,
            candidates=candidates,
            evidence_items=all_evidence,
            sources=all_sources,
            analysis=analysis,
        )
        answer_type = analysis.answer_type
        agent_packet = self.evidence_compressor.compress(
            question=question,
            answer_type=answer_type,
            verified_candidates=verified_candidates,
            fact_cards=fact_cards,
        )

        output = EvidenceOutput(
            question=question,
            queries=[*queries, *verification_queries],
            sources=all_sources,
            evidence_items=all_evidence,
            candidates=candidates,
            summary="",
            verified_candidates=verified_candidates,
            fact_cards=fact_cards,
            agent_packet=agent_packet,
            question_analysis=analysis,
            candidate_diagnostics=self._candidate_diagnostics(
                analysis=analysis,
                raw_candidates=raw_candidates,
                filtered_candidates=candidates,
                verified_candidates=verified_candidates,
                rejected_candidates=rejected_candidates,
                probability_rerank=rerank_diagnostics,
            ),
            tool_usage=tool_usage,
            blocked_sources=blocked_sources,
        )
        output.summary = (
            self.agent_renderer.render(agent_packet)
            if self.compact_evidence
            else self.renderer.render(output)
        )
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

    def _answer_type(self, question: str, candidates: list[Any]) -> str:
        if candidates:
            return str(candidates[0].answer_type or "entity")
        lowered = question.lower()
        if "who" in lowered:
            return "person"
        if "where" in lowered:
            return "place"
        if "when" in lowered or "date" in lowered or "year" in lowered:
            return "date"
        if "title" in lowered or "book" in lowered or "paper" in lowered:
            return "title"
        if any(marker in lowered for marker in ("number", "how many", "amount", "score")):
            return "number"
        return "entity"

    def _candidate_diagnostics(
        self,
        *,
        analysis: Any,
        raw_candidates: list[Any],
        filtered_candidates: list[Any],
        verified_candidates: list[Any],
        rejected_candidates: list[dict[str, str]],
        probability_rerank: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        top_candidate = verified_candidates[0] if verified_candidates else None
        return {
            "answer_type": getattr(analysis, "answer_type", "unknown"),
            "raw_candidate_count": len(raw_candidates),
            "filtered_candidate_count": len(filtered_candidates),
            "verified_candidate_count": len(verified_candidates),
            "top_candidate": getattr(top_candidate, "answer", "") if top_candidate else "",
            "top_candidate_confidence": getattr(top_candidate, "confidence", 0.0) if top_candidate else 0.0,
            "top_candidate_support": getattr(top_candidate, "support_count", 0) if top_candidate else 0,
            "top_candidate_refute": getattr(top_candidate, "refute_count", 0) if top_candidate else 0,
            "rejected_candidates": rejected_candidates[:20],
            "probability_rerank": probability_rerank or {"enabled": False},
        }

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""


__all__ = ["EvidenceSearcher"]
