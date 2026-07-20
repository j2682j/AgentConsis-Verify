from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from score.candidate_fact_verifier import CandidateFactVerifier
from tools.evidence.fact_extraction import (
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactStore,
)
from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.source_analyze.seer.source_filter import SourceFilter
from utils.network_utils import normalize_for_exact, normalize_text


@dataclass
class CandidateVerificationTrace:
    """Record one bounded evidence-recovery attempt for an answer candidate."""

    candidate_key: str
    candidate_answer: str
    query: str
    status: str = "unresolved"
    source_ids: list[str] = field(default_factory=list)
    supporting_fact_ids: list[str] = field(default_factory=list)
    contradicting_fact_ids: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateVerificationResult:
    """Return recovery traces and the number of newly grounded facts."""

    attempted: bool = False
    reason: str = ""
    added_fact_count: int = 0
    traces: list[CandidateVerificationTrace] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_evidence(self) -> bool:
        return self.added_fact_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "reason": self.reason,
            "added_fact_count": self.added_fact_count,
            "traces": [trace.to_dict() for trace in self.traces],
            "diagnostics": dict(self.diagnostics),
        }


class CandidateVerificationSearcher:
    """Recover candidate-bound facts only when the normal evidence path is empty."""

    def __init__(
        self,
        *,
        search_executor: Callable[[str, int], dict[str, Any]] | None = None,
        semantic_fact_extractor: SemanticFactExtractor | None = None,
        candidate_fact_verifier: CandidateFactVerifier | None = None,
        source_filter: SourceFilter | None = None,
        max_candidates: int = 5,
        max_results_per_candidate: int = 3,
        max_workers: int = 2,
    ) -> None:
        self.search_executor = search_executor
        self.semantic_fact_extractor = semantic_fact_extractor or SemanticFactExtractor()
        self.candidate_fact_verifier = candidate_fact_verifier or CandidateFactVerifier()
        self.source_filter = source_filter or SourceFilter(min_sources=0)
        self.max_candidates = max(1, int(max_candidates))
        self.max_results_per_candidate = max(1, int(max_results_per_candidate))
        self.max_workers = max(1, int(max_workers))

    def verify(
        self,
        *,
        question: str,
        candidate_answers: list[str],
        fact_store: TaskFactStore,
        answer_requirement: str = "",
        answer_role: str = "",
        required_relation: str = "",
        required_relation_goal_id: str = "",
    ) -> CandidateVerificationResult:
        candidates = self._dedupe_candidates(candidate_answers)[: self.max_candidates]
        if self.search_executor is None:
            return CandidateVerificationResult(reason="search_executor_unavailable")
        if not candidates:
            return CandidateVerificationResult(reason="no_candidate_answers")

        traces = [
            CandidateVerificationTrace(
                candidate_key=normalize_for_exact(answer),
                candidate_answer=answer,
                query=self._verification_query(question, answer),
            )
            for answer in candidates
        ]
        sources_by_key: dict[str, list[SearchSourceCandidate]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as executor:
            pending = {
                executor.submit(
                    self.search_executor,
                    trace.query,
                    self.max_results_per_candidate,
                ): trace
                for trace in traces
            }
            for future in as_completed(pending):
                trace = pending[future]
                try:
                    payload = future.result()
                    sources = self._filtered_sources(
                        payload,
                        question=question,
                        candidate_key=trace.candidate_key,
                    )
                    sources_by_key[trace.candidate_key] = sources
                    trace.source_ids = [source.source_id for source in sources]
                    if not sources:
                        trace.rejection_reasons.append("no_safe_search_sources")
                except Exception as exc:
                    trace.rejection_reasons.append(
                        f"search_error:{type(exc).__name__}"
                    )

        units: list[SemanticSourceUnit] = []
        for trace in traces:
            for source in sources_by_key.get(trace.candidate_key, []):
                units.append(
                    SemanticSourceUnit(
                        unit_id=source.source_id,
                        source_id=source.source_id,
                        source_type="candidate_verification_search",
                        source_title=source.title,
                        text=source.raw_content or source.snippet,
                        requested_role="ANSWER_SUPPORT",
                        goal_id=required_relation_goal_id,
                        metadata={
                            "candidate_key": trace.candidate_key,
                            "candidate_answer": trace.candidate_answer,
                            "url": source.url,
                        },
                    )
                )

        added = 0
        extraction_diagnostics: list[dict[str, Any]] = []
        batch_size = max(1, int(self.semantic_fact_extractor.max_units_per_call))
        for start in range(0, len(units), batch_size):
            extraction = self.semantic_fact_extractor.extract_batch(
                question=question,
                answer_requirement=answer_requirement,
                current_goal=required_relation,
                units=units[start : start + batch_size],
                keep_alive=0,
            )
            added += fact_store.extend(extraction.facts)
            extraction_diagnostics.append(dict(extraction.diagnostics))

        for trace in traces:
            verification = self.candidate_fact_verifier.verify(
                candidate_answer=trace.candidate_answer,
                fact_store=fact_store,
                answer_requirement=answer_requirement,
                required_relation=required_relation,
                required_relation_goal_id=required_relation_goal_id,
                answer_role=answer_role,
            )
            trace.supporting_fact_ids = list(verification.supporting_fact_ids)
            trace.contradicting_fact_ids = list(verification.contradicting_fact_ids)
            trace.status = (
                "supported"
                if verification.status == "supported"
                else "contradicted"
                if verification.status == "contradicted"
                else "unresolved"
            )
            if verification.reason:
                trace.rejection_reasons.append(verification.reason)

        return CandidateVerificationResult(
            attempted=True,
            reason="all_candidates_unsupported_recovery",
            added_fact_count=added,
            traces=traces,
            diagnostics={
                "searched_candidate_count": len(traces),
                "safe_source_count": sum(len(items) for items in sources_by_key.values()),
                "semantic_extraction": extraction_diagnostics,
            },
        )

    def _filtered_sources(
        self,
        payload: dict[str, Any],
        *,
        question: str,
        candidate_key: str,
    ) -> list[SearchSourceCandidate]:
        raw = payload.get("raw_result") if isinstance(payload, dict) else None
        if isinstance(raw, dict):
            results = list(raw.get("results") or [])
        else:
            results = list(payload.get("results") or []) if isinstance(payload, dict) else []
        sources: list[SearchSourceCandidate] = []
        for index, item in enumerate(results[: self.max_results_per_candidate], start=1):
            if not isinstance(item, dict):
                continue
            url = normalize_text(str(item.get("url") or ""))
            content = normalize_text(
                str(item.get("content") or item.get("snippet") or "")
            )
            if not url or len(content) < 24:
                continue
            sources.append(
                SearchSourceCandidate(
                    source_id=f"candidate-{candidate_key}-{index}",
                    query_id=f"candidate-{candidate_key}",
                    title=normalize_text(str(item.get("title") or url)),
                    url=url,
                    domain=urlparse(url).netloc.casefold(),
                    snippet=content,
                    raw_content=content,
                    rank=index,
                    fetched=True,
                    content_complete=False,
                    transport_ok=True,
                    content_extracted=True,
                )
            )
        return self.source_filter.filter_sources(
            sources,
            question=question,
            fetch_limit=0,
        )

    @staticmethod
    def _verification_query(question: str, answer: str) -> str:
        return f'{normalize_text(question)} "{normalize_text(answer)}"'

    @staticmethod
    def _dedupe_candidates(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            answer = normalize_text(value)
            key = normalize_for_exact(answer)
            if answer and key and key not in seen:
                seen.add(key)
                result.append(answer)
        return result


__all__ = [
    "CandidateVerificationResult",
    "CandidateVerificationSearcher",
    "CandidateVerificationTrace",
]
