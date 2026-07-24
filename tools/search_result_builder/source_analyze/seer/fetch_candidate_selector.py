"""Decide which filtered sources are fetched in full, and in what order.

``SourceFilter`` answers "is this source usable"; this module answers "is this
source worth a fetch slot". They are separate because the second question is a
ranking problem and the first is a safety problem.

Ranking is lexicographic over tiers rather than a weighted score, so a
promotion is always explainable by one concrete reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...config import SearchSourceCandidate
from .source_selection_signals import FULL_MATCH, PARTIAL_MATCH


MODE_ADDITIVE = "additive"
MODE_PRIORITY = "priority"

TIER_NAMED_SOURCE = 0
TIER_CROSS_QUERY = 1
TIER_REQUIREMENT = 2
TIER_CONSTRAINT = 3
TIER_GENERAL = 4
TIER_ECHO = 5
TIER_DEMOTED = 6


@dataclass
class FetchSelectionDecision:
    source_id: str
    url: str
    priority_tier: int
    fetch_batch: int
    legacy_position: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "priority_tier": self.priority_tier,
            "fetch_batch": self.fetch_batch,
            "legacy_position": self.legacy_position,
            "selection_reasons": list(self.reasons),
        }


@dataclass
class FetchSelectionResult:
    initial_sources: list[SearchSourceCandidate] = field(default_factory=list)
    deferred_sources: list[SearchSourceCandidate] = field(default_factory=list)
    decisions: list[FetchSelectionDecision] = field(default_factory=list)
    mode: str = MODE_ADDITIVE

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_mode": self.mode,
            "initial_count": len(self.initial_sources),
            "deferred_count": len(self.deferred_sources),
            "candidates": [item.to_dict() for item in self.decisions],
        }


class FetchCandidateSelector:
    """Order sources into an initial fetch batch and a deferred queue."""

    def __init__(
        self,
        *,
        mode: str = MODE_ADDITIVE,
        legacy_fetch_limit: int = 6,
        initial_fetch_limit: int = 8,
        promoted_slots: int = 2,
        demoted_domain_markers: tuple[str, ...] = (),
        product_page_markers: tuple[str, ...] = (),
    ) -> None:
        self.mode = mode if mode in {MODE_ADDITIVE, MODE_PRIORITY} else MODE_ADDITIVE
        self.legacy_fetch_limit = max(0, legacy_fetch_limit)
        self.initial_fetch_limit = max(self.legacy_fetch_limit, initial_fetch_limit)
        self.promoted_slots = max(0, promoted_slots)
        self.demoted_domain_markers = tuple(demoted_domain_markers)
        self.product_page_markers = tuple(product_page_markers)

    def select(
        self,
        sources: list[SearchSourceCandidate],
        *,
        fetch_limit: int | None = None,
    ) -> FetchSelectionResult:
        """Split eligible sources into an initial batch and a deferred queue."""
        eligible = [
            source
            for source in sources
            if not source.blocked and source.url and not source.fetched
        ]
        for position, source in enumerate(eligible):
            source.legacy_fetch_position = position
            source.fetch_priority_tier = self._tier(source)

        limit = self.initial_fetch_limit if fetch_limit is None else max(0, fetch_limit)
        if self.mode == MODE_PRIORITY:
            initial = self._by_priority(eligible)[:limit]
        else:
            initial = self._additive(eligible, limit=limit)

        initial_ids = {id(source) for source in initial}
        deferred = [
            source
            for source in self._by_priority(eligible)
            if id(source) not in initial_ids
        ]

        decisions: list[FetchSelectionDecision] = []
        for source in eligible:
            batch = 1 if id(source) in initial_ids else 2
            source.fetch_batch = batch
            source.should_fetch_full_page = batch == 1
            if batch == 1 and "fetch_candidate" not in source.filter_reasons:
                source.filter_reasons.append("fetch_candidate")
            decisions.append(
                FetchSelectionDecision(
                    source_id=source.source_id,
                    url=source.url,
                    priority_tier=source.fetch_priority_tier,
                    fetch_batch=batch,
                    legacy_position=source.legacy_fetch_position,
                    reasons=list(source.fetch_priority_reasons),
                )
            )
        for source in sources:
            if source.blocked or source.fetched:
                source.should_fetch_full_page = False
        return FetchSelectionResult(
            initial_sources=initial,
            deferred_sources=deferred,
            decisions=decisions,
            mode=self.mode,
        )

    def _additive(
        self,
        eligible: list[SearchSourceCandidate],
        *,
        limit: int,
    ) -> list[SearchSourceCandidate]:
        """Keep the legacy head, then add the best sources it missed.

        This is the safe deployment shape: every page the previous policy
        fetched is still fetched, so a task that was already answered cannot
        lose the source that answered it. Promotions only use the spare slots.
        """
        legacy = eligible[: min(self.legacy_fetch_limit, limit)]
        legacy_ids = {id(source) for source in legacy}
        spare = max(0, limit - len(legacy))
        promoted: list[SearchSourceCandidate] = []
        for source in self._by_priority(eligible):
            if len(promoted) >= min(self.promoted_slots, spare):
                break
            if id(source) in legacy_ids:
                continue
            # Only promote on a positive reason; never pad the batch just to
            # fill slots, which would add cost without adding evidence.
            if source.fetch_priority_tier > TIER_CONSTRAINT:
                continue
            promoted.append(source)
        return legacy + promoted

    def _by_priority(
        self,
        eligible: list[SearchSourceCandidate],
    ) -> list[SearchSourceCandidate]:
        return sorted(eligible, key=self._sort_key)

    def _sort_key(self, source: SearchSourceCandidate) -> tuple:
        constraint_rank = {
            FULL_MATCH: 0,
            PARTIAL_MATCH: 1,
        }.get(source.constraint_match_level, 2)
        kind_rank = 0 if source.source_kind in {"academic", "collection"} else 1
        return (
            source.fetch_priority_tier,
            -source.query_hit_count,
            constraint_rank,
            kind_rank,
            source.legacy_fetch_position,
            source.source_id,
        )

    def _tier(self, source: SearchSourceCandidate) -> int:
        haystack = f"{source.domain} {source.url}".casefold()
        if any(marker in haystack for marker in self.demoted_domain_markers) or any(
            marker in haystack for marker in self.product_page_markers
        ):
            return TIER_DEMOTED
        if source.named_source_match:
            # A question that names its source outranks everything: the answer
            # is by construction on that site.
            if source.constraint_match_level != "no_match" or not source.url_echo:
                return TIER_NAMED_SOURCE
        if source.url_echo or "question_echo_only" in source.filter_reasons:
            return TIER_ECHO
        if source.query_hit_count > 1:
            return TIER_CROSS_QUERY
        if source.source_kind in {"academic", "collection"} or source.required_content in {
            "pdf_text",
            "collection_records",
        }:
            return TIER_REQUIREMENT
        if source.constraint_match_level == FULL_MATCH:
            return TIER_CONSTRAINT
        return TIER_GENERAL


__all__ = [
    "FetchCandidateSelector",
    "FetchSelectionDecision",
    "FetchSelectionResult",
    "MODE_ADDITIVE",
    "MODE_PRIORITY",
]
