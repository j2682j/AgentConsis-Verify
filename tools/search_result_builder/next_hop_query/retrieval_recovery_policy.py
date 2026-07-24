from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from utils.network_utils import normalize_text

from ..query.relation_plan import RelationPlan
from ..query.source_requirement import SearchQueryRequest, SourceRequirement


@dataclass(frozen=True)
class RetrievalRecoveryDecision:
    """Describe one bounded recovery action for an incomplete retrieval state."""

    action: str
    reason: str
    fingerprint: str
    requests: list[SearchQueryRequest] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)
    top_k: int = 0
    candidate_pool_size: int = 0

    @property
    def viable(self) -> bool:
        return self.action != "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requests": [item.to_dict() for item in self.requests],
        }


class RetrievalRecoveryPolicy:
    """Choose direct fetch, browser fetch, broader retrieval, or a goal query."""

    def __init__(
        self,
        *,
        max_top_k: int = 64,
        max_candidate_pool_size: int = 120,
        max_source_requests: int = 3,
        max_expand_attempts: int = 1,
    ) -> None:
        self.max_top_k = max(1, max_top_k)
        self.max_candidate_pool_size = max(1, max_candidate_pool_size)
        self.max_source_requests = max(1, max_source_requests)
        self.max_expand_attempts = max(0, int(max_expand_attempts))

    def decide(
        self,
        *,
        relation_plan: RelationPlan | None,
        corpus_documents: Iterable[Mapping[str, Any]],
        attempted: set[str],
        top_k: int,
        candidate_pool_size: int,
        original_question: str,
        bridge_terms: list[str] | None = None,
        missing_constraints: list[str] | None = None,
    ) -> RetrievalRecoveryDecision:
        plan = relation_plan or RelationPlan()
        corpus = list(corpus_documents)
        unresolved = [goal for goal in plan.goals if goal.state != "resolved"]
        full_document_goal = next(
            (
                goal
                for goal in unresolved
                if goal.verification_scope in {"full_document", "collection"}
            ),
            None,
        )
        if full_document_goal is not None:
            urls = self._unfetched_content_urls(corpus)
            for access_mode in ("direct_fetch", "browser"):
                requests = [
                    SearchQueryRequest(
                        query=url,
                        source_requirement=SourceRequirement(
                            source_kind=full_document_goal.source_kind,
                            access_mode=access_mode,
                            source_hint=url,
                            required_content=full_document_goal.required_content,
                        ),
                    )
                    for url in urls[: self.max_source_requests]
                ]
                fingerprint = self._request_fingerprint(access_mode, requests)
                if requests and fingerprint not in attempted:
                    return RetrievalRecoveryDecision(
                        action=access_mode,
                        reason="full_document_verification_required",
                        fingerprint=fingerprint,
                        requests=requests,
                        next_queries=[item.query for item in requests],
                        top_k=top_k,
                        candidate_pool_size=candidate_pool_size,
                    )

        expanded_top_k = min(self.max_top_k, max(top_k + 1, top_k * 2))
        expanded_pool = min(
            self.max_candidate_pool_size,
            max(candidate_pool_size + 1, candidate_pool_size * 2),
        )
        expand_fingerprint = f"expand:{expanded_top_k}:{expanded_pool}"
        expand_attempts = sum(
            str(fingerprint).startswith("expand:") for fingerprint in attempted
        )
        if (
            len(corpus) > top_k
            and
            (expanded_top_k > top_k or expanded_pool > candidate_pool_size)
            and expand_attempts < self.max_expand_attempts
            and expand_fingerprint not in attempted
        ):
            return RetrievalRecoveryDecision(
                action="expand_retrieval",
                reason="goal_incomplete_without_bridge",
                fingerprint=expand_fingerprint,
                next_queries=[normalize_text(original_question)],
                top_k=expanded_top_k,
                candidate_pool_size=expanded_pool,
            )

        goal_query = self._goal_query(plan)
        goal_fingerprint = f"goal_query:{goal_query.casefold()}"
        if goal_query and goal_fingerprint not in attempted:
            active = next((goal for goal in unresolved), None)
            request = SearchQueryRequest(
                query=goal_query,
                source_requirement=SourceRequirement(
                    source_kind=active.source_kind if active is not None else "web",
                    access_mode="search",
                    required_content=(active.required_content if active is not None else "html_text"),
                ),
            )
            return RetrievalRecoveryDecision(
                action="goal_query",
                reason="unresolved_goal_requires_new_source",
                fingerprint=goal_fingerprint,
                requests=[request],
                next_queries=[goal_query],
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
            )

        bridge_query = self._bridge_query(
            original_question,
            bridge_terms=list(bridge_terms or []),
            missing_constraints=list(missing_constraints or []),
        )
        bridge_fingerprint = f"bridge_query:{bridge_query.casefold()}"
        if bridge_query and bridge_fingerprint not in attempted:
            # Coverage found intermediate entities (bridge terms) that the
            # original question never names; a second hop built from them can
            # reach pages the question-shaped query cannot.
            request = SearchQueryRequest(
                query=bridge_query,
                source_requirement=SourceRequirement(
                    source_kind="web",
                    access_mode="search",
                ),
            )
            return RetrievalRecoveryDecision(
                action="bridge_query",
                reason="coverage_bridge_terms_second_hop",
                fingerprint=bridge_fingerprint,
                requests=[request],
                next_queries=[bridge_query],
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
            )

        return RetrievalRecoveryDecision(
            action="stop",
            reason="no_viable_recovery_action",
            fingerprint="stop",
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )

    def _unfetched_content_urls(
        self,
        corpus_documents: Iterable[Mapping[str, Any]],
    ) -> list[str]:
        complete_urls: set[str] = set()
        candidate_urls: list[str] = []
        seen: set[str] = set()
        for document in corpus_documents:
            url = normalize_text(
                str(document.get("content_url") or document.get("url") or "")
            )
            if not url:
                continue
            key = url.casefold().rstrip("/")
            if (
                document.get("content_scope") == "full_document"
                and bool(document.get("content_complete"))
                and not bool(document.get("content_truncated"))
            ):
                complete_urls.add(key)
            elif document.get("content_url") and key not in seen:
                candidate_urls.append(url)
                seen.add(key)
        return [url for url in candidate_urls if url.casefold().rstrip("/") not in complete_urls]

    def _bridge_query(
        self,
        original_question: str,
        *,
        bridge_terms: list[str],
        missing_constraints: list[str],
        max_bridge_terms: int = 4,
        max_missing_terms: int = 2,
    ) -> str:
        """Compose a second-hop query from coverage bridge terms.

        Requires at least one bridge term so the query differs from a plain
        re-search; missing constraints keep the hop anchored to what the
        sufficiency gate said is still unresolved.
        """
        seen: set[str] = set()
        parts: list[str] = []
        for term in bridge_terms:
            cleaned = normalize_text(str(term))
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            parts.append(cleaned)
            if len(parts) >= max_bridge_terms:
                break
        if not parts:
            return ""
        missing_added = 0
        for term in missing_constraints:
            cleaned = normalize_text(str(term))
            key = cleaned.casefold()
            if not cleaned or key in seen or ":" in cleaned:
                continue
            seen.add(key)
            parts.append(cleaned)
            missing_added += 1
            if missing_added >= max_missing_terms:
                break
        query = normalize_text(" ".join(parts))
        if not query or query.casefold() == normalize_text(original_question).casefold():
            return ""
        return query

    def _goal_query(self, plan: RelationPlan) -> str:
        unresolved = next((goal for goal in plan.goals if goal.state != "resolved"), None)
        if unresolved is None:
            return ""
        prior_values = [
            value
            for goal in plan.goals
            if goal.state == "resolved"
            for value in goal.resolved_values
        ]
        subject = " ".join(prior_values[-2:]) or unresolved.subject
        return normalize_text(
            " ".join(
                part
                for part in (subject, unresolved.relation, unresolved.target)
                if normalize_text(part)
            )
        )

    def _request_fingerprint(
        self,
        access_mode: str,
        requests: list[SearchQueryRequest],
    ) -> str:
        urls = "|".join(item.query.casefold().rstrip("/") for item in requests)
        return f"{access_mode}:{urls}"


__all__ = ["RetrievalRecoveryDecision", "RetrievalRecoveryPolicy"]
