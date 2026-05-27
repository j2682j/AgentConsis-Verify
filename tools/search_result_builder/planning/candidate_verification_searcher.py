from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, QuestionAnalysis, SearchQueryPlan, SearchSourceCandidate


@dataclass
class CandidateVerificationResult:
    """
    Store verification search output for one candidate answer.
    """
    candidate: CandidateAnswer
    queries: list[SearchQueryPlan] = field(default_factory=list)
    sources: list[SearchSourceCandidate] = field(default_factory=list)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)


class CandidateVerificationSearcher:
    """
    Run candidate-specific verification searches and convert results into sources.

    Args:
        - tool_manager: Object exposing execute_tool(tool_name, parameters, agent_id, stage).

    Returns:
        - CandidateVerificationSearcher: Candidate verification search service.
    """

    def __init__(self, *, tool_manager: Any, domain_parser: Any) -> None:
        self.tool_manager = tool_manager
        self.domain_parser = domain_parser
        self._cache: dict[str, dict[str, Any]] = {}

    def verify_candidates(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis,
        candidates: list[CandidateAnswer],
        max_candidates: int = 5,
        max_results_per_query: int = 3,
        max_full_page_results: int = 1,
        agent_id: str = "evidence_searcher",
        stage: str = "candidate_verification",
    ) -> tuple[list[CandidateVerificationResult], list[dict[str, Any]]]:
        """
        Run one verification search per candidate.

        Args:
            - question: Original task question.
            - analysis: QuestionAnalysis for target terms and constraints.
            - candidates: Candidate answers to verify.
            - max_candidates: Maximum candidates to verify.
            - max_results_per_query: Maximum search results per verification query.
            - max_full_page_results: Maximum full pages fetched per query.
            - agent_id: Trace agent id used by ToolManager.
            - stage: Trace stage used by ToolManager.

        Returns:
            - tuple[list[CandidateVerificationResult], list[dict[str, Any]]]: Verification bundles and tool usage.
        """
        results: list[CandidateVerificationResult] = []
        tool_usage: list[dict[str, Any]] = []

        for index, candidate in enumerate(candidates[:max_candidates], start=1):
            query_text = self.build_query(
                question=question,
                analysis=analysis,
                candidate=candidate,
            )
            if not query_text:
                continue
            query_plan = SearchQueryPlan(
                query_id=f"V{index}",
                query=query_text,
                purpose="candidate_verification",
                priority=70 - index,
                source_hints=list(analysis.source_hints),
                expected_answer_type=analysis.answer_type,
                requires_full_page=True,
            )
            result = self._execute_search(
                query=query_text,
                max_results=max_results_per_query,
                max_full_page_results=max_full_page_results,
                agent_id=agent_id,
                stage=stage,
            )
            tool_usage.append(result)
            sources = self._sources_from_result(
                result,
                query_id=query_plan.query_id,
                source_prefix=f"VS{index}",
            )
            results.append(
                CandidateVerificationResult(
                    candidate=candidate,
                    queries=[query_plan],
                    sources=sources,
                    tool_usage=[result],
                )
            )

        return results, tool_usage

    def build_query(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis,
        candidate: CandidateAnswer,
    ) -> str:
        terms = [f'"{candidate.answer}"']
        terms.extend(f'"{term}"' for term in analysis.target_terms[:3] if term)
        terms.extend(analysis.constraints[:3])
        terms.extend(analysis.source_hints[:2])
        if len(terms) <= 1:
            terms.append(normalize_text(question)[:160])
        return normalize_text(" ".join(terms))[:300]

    def _execute_search(
        self,
        *,
        query: str,
        max_results: int,
        max_full_page_results: int,
        agent_id: str,
        stage: str,
    ) -> dict[str, Any]:
        cache_key = f"{query.lower()}|{max_results}|{max_full_page_results}"
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cache_hit"] = True
            return cached
        try:
            result = self.tool_manager.execute_tool(
                "search",
                {
                    "input": query,
                    "mode": "structured",
                    "conditional_fetch": True,
                    "max_results": max_results,
                    "max_full_page_results": max_full_page_results,
                },
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": "search",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }
        self._cache[cache_key] = dict(result)
        return result

    def _sources_from_result(
        self,
        result: dict[str, Any],
        *,
        query_id: str,
        source_prefix: str,
    ) -> list[SearchSourceCandidate]:
        payload = result.get("raw_result")
        if not isinstance(payload, dict):
            return []
        raw_items = payload.get("results") or []
        if not isinstance(raw_items, list):
            return []

        sources: list[SearchSourceCandidate] = []
        for rank, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                continue
            url = normalize_text(item.get("url", ""))
            title = normalize_text(item.get("title", "")) or url
            snippet = normalize_text(item.get("content", ""))
            raw_content = normalize_text(item.get("raw_content", ""))
            sources.append(
                SearchSourceCandidate(
                    source_id=f"{source_prefix}_{len(sources) + 1}",
                    query_id=query_id,
                    title=title,
                    url=url,
                    domain=self.domain_parser(url),
                    snippet=snippet,
                    raw_content=raw_content,
                    rank=rank,
                    rerank_score=float(item.get("rerank_score", 0.0) or 0.0),
                    fetched=bool(raw_content),
                )
            )
        return sources


__all__ = ["CandidateVerificationResult", "CandidateVerificationSearcher"]

