"""Plan-12 fetch selection: authoritative sources earn fetch slots, SEO pages don't."""

from __future__ import annotations

import unittest

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.source_analyze.seer.fetch_candidate_selector import (
    FetchCandidateSelector,
    MODE_ADDITIVE,
    MODE_PRIORITY,
    TIER_DEMOTED,
    TIER_ECHO,
    TIER_NAMED_SOURCE,
)
from tools.search_result_builder.source_analyze.seer.source_filter import SourceFilter
from tools.search_result_builder.source_analyze.seer.source_selection_signals import (
    FULL_MATCH,
    SourceSelectionSignalBuilder,
    extract_constraints,
)


MW_QUESTION = (
    "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"
)


def source(source_id: str, query_id: str, title: str, url: str, **kwargs) -> SearchSourceCandidate:
    return SearchSourceCandidate(
        source_id=source_id,
        query_id=query_id,
        title=title,
        url=url,
        snippet=kwargs.pop("snippet", "A snippet with enough distinct wording to survive dedup."),
        **kwargs,
    )


class CrossQueryAggregationTests(unittest.TestCase):
    def test_same_url_from_two_queries_counts_once_per_query(self) -> None:
        sources = [
            source("S1", "Q1", "Jingoism", "https://www.merriam-webster.com/word-of-the-day/jingoism"),
            source("S2", "Q2", "Jingoism", "https://www.merriam-webster.com/word-of-the-day/jingoism"),
        ]
        SourceFilter().filter_sources(
            sources,
            question=MW_QUESTION,
            query_text_by_id={"Q1": "merriam webster word of the day", "Q2": "jingoism quote writer"},
            fetch_limit=6,
        )
        self.assertEqual(sources[0].query_hit_count, 2)
        self.assertEqual(sources[1].block_reason, "duplicate_url")

    def test_repeated_identical_query_does_not_inflate_consensus(self) -> None:
        sources = [
            source("S1", "Q1", "Jingoism", "https://www.merriam-webster.com/word-of-the-day/jingoism"),
            source("S2", "Q2", "Jingoism", "https://www.merriam-webster.com/word-of-the-day/jingoism"),
        ]
        SourceFilter().filter_sources(
            sources,
            question=MW_QUESTION,
            # Same query text under two ids must not count as consensus.
            query_text_by_id={"Q1": "merriam webster jingoism", "Q2": "merriam webster jingoism"},
            fetch_limit=6,
        )
        self.assertEqual(sources[0].query_hit_count, 1)


class SelectionSignalTests(unittest.TestCase):
    def test_named_source_matches_hyphenated_brand(self) -> None:
        sources = [
            source("S1", "Q1", "Word of the Day", "https://www.merriam-webster.com/word-of-the-day/jingoism"),
            source("S2", "Q1", "Random blog", "https://example.com/post"),
        ]
        SourceSelectionSignalBuilder().build(sources, question=MW_QUESTION)
        self.assertTrue(sources[0].named_source_match)
        self.assertFalse(sources[1].named_source_match)

    def test_url_echo_detected_on_path_not_domain(self) -> None:
        echo = source(
            "S1",
            "Q1",
            "Crossword",
            "https://www.wordplays.com/crossword-solver/"
            "what-writer-is-quoted-by-merriam-webster-for-the-word-of-the-day-from-june-27-2022",
        )
        clean = source("S2", "Q1", "Archive", "https://www.merriam-webster.com/word-of-the-day/jingoism")
        SourceSelectionSignalBuilder().build([echo, clean], question=MW_QUESTION)
        self.assertTrue(echo.url_echo)
        self.assertFalse(clean.url_echo)

    def test_constraints_extracted_and_matched(self) -> None:
        constraints = extract_constraints(
            'In the Scikit-Learn July 2017 changelog for version 0.19, what changed?'
        )
        self.assertIn("2017", constraints.years)
        self.assertIn("0.19", " ".join(constraints.versions))
        candidate = source(
            "S1", "Q1", "Version 0.19 changelog 2017", "https://scikit-learn.org/whats_new/v0.19.html"
        )
        SourceSelectionSignalBuilder().build(
            [candidate],
            question="In the Scikit-Learn July 2017 changelog for version 0.19, what changed?",
        )
        self.assertEqual(candidate.constraint_match_level, FULL_MATCH)


class FetchSelectorTests(unittest.TestCase):
    def _selector(self, **kwargs) -> FetchCandidateSelector:
        return FetchCandidateSelector(
            demoted_domain_markers=SourceFilter.DEMOTED_DOMAIN_MARKERS,
            product_page_markers=SourceFilter.PRODUCT_PAGE_MARKERS,
            **kwargs,
        )

    def test_seo_and_product_pages_are_demoted(self) -> None:
        seo = source("S1", "Q1", "Crossword", "https://www.wordplays.com/crossword-solver/foo-bar")
        shop = source("S2", "Q1", "Listing", "https://www.etsy.com/market/libretext_chemistry")
        normal = source("S3", "Q1", "Docs", "https://example.org/docs/page")
        selector = self._selector()
        selector.select([seo, shop, normal])
        self.assertEqual(seo.fetch_priority_tier, TIER_DEMOTED)
        self.assertEqual(shop.fetch_priority_tier, TIER_DEMOTED)
        self.assertLess(normal.fetch_priority_tier, TIER_DEMOTED)

    def test_named_source_outranks_echo_page(self) -> None:
        echo = source("S1", "Q1", "Crossword", "https://www.wordplays.com/crossword-solver/x")
        echo.url_echo = True
        named = source("S2", "Q1", "MW", "https://www.merriam-webster.com/word-of-the-day/jingoism")
        named.named_source_match = True
        selector = self._selector()
        selector.select([echo, named])
        self.assertEqual(named.fetch_priority_tier, TIER_NAMED_SOURCE)
        self.assertEqual(echo.fetch_priority_tier, TIER_DEMOTED)

    def test_additive_mode_preserves_every_legacy_page(self) -> None:
        """The safety property: additive mode can add, never remove."""
        sources = [source(f"S{i}", "Q1", f"T{i}", f"https://example{i}.com/a") for i in range(10)]
        promoted = source("S99", "Q1", "Named", "https://www.merriam-webster.com/word-of-the-day/x")
        promoted.named_source_match = True
        pool = sources + [promoted]
        selector = self._selector(legacy_fetch_limit=6, initial_fetch_limit=8, promoted_slots=2)

        result = selector.select(pool)

        legacy_urls = [item.url for item in pool[:6]]
        initial_urls = [item.url for item in result.initial_sources]
        for url in legacy_urls:
            self.assertIn(url, initial_urls)
        self.assertIn(promoted.url, initial_urls)
        self.assertLessEqual(len(result.initial_sources), 8)

    def test_no_padding_when_nothing_deserves_promotion(self) -> None:
        sources = [source(f"S{i}", "Q1", f"T{i}", f"https://example{i}.com/a") for i in range(10)]
        selector = self._selector(legacy_fetch_limit=6, initial_fetch_limit=8, promoted_slots=2)

        result = selector.select(sources)

        # Nothing has a positive reason, so the batch stays at the legacy size
        # instead of spending two extra fetches for nothing.
        self.assertEqual(len(result.initial_sources), 6)
        self.assertTrue(result.deferred_sources)

    def test_deferred_queue_holds_unfetched_sources(self) -> None:
        sources = [source(f"S{i}", "Q1", f"T{i}", f"https://example{i}.com/a") for i in range(10)]
        selector = self._selector(legacy_fetch_limit=3, initial_fetch_limit=3, promoted_slots=0)

        result = selector.select(sources)

        self.assertEqual(len(result.initial_sources), 3)
        self.assertEqual(len(result.deferred_sources), 7)
        for item in result.initial_sources:
            self.assertTrue(item.should_fetch_full_page)
            self.assertEqual(item.fetch_batch, 1)
        for item in result.deferred_sources:
            self.assertFalse(item.should_fetch_full_page)
            self.assertEqual(item.fetch_batch, 2)

    def test_priority_mode_puts_named_source_first(self) -> None:
        filler = [source(f"S{i}", "Q1", f"T{i}", f"https://example{i}.com/a") for i in range(6)]
        named = source("S99", "Q1", "MW", "https://www.merriam-webster.com/word-of-the-day/x")
        named.named_source_match = True
        selector = self._selector(mode=MODE_PRIORITY, initial_fetch_limit=3)

        result = selector.select(filler + [named])

        self.assertEqual(result.initial_sources[0].url, named.url)


class SourceFilterIntegrationTests(unittest.TestCase):
    def test_named_source_reaches_initial_batch_over_seo_pages(self) -> None:
        """The regression this plan exists for: the named site must be fetched."""
        seo = [
            source(
                f"S{i}",
                "Q1",
                f"What writer is quoted by Merriam-Webster {i}",
                f"https://www.wordplays.com/crossword-solver/"
                f"what-writer-is-quoted-by-merriam-webster-word-of-the-day-june-27-2022-{i}",
                snippet=f"Crossword answers listing number {i} with assorted filler wording.",
            )
            for i in range(6)
        ]
        authoritative = source(
            "S99",
            "Q2",
            "Word of the Day: Jingoism | Merriam-Webster",
            "https://www.merriam-webster.com/word-of-the-day/jingoism-2022-06-27",
            snippet="The Word of the Day for June 27 2022 with an example sentence and attribution.",
        )
        filtered = SourceFilter().filter_sources(
            seo + [authoritative],
            question=MW_QUESTION,
            query_text_by_id={"Q1": "writer quoted merriam webster", "Q2": "merriam webster jingoism"},
            fetch_limit=8,
        )
        selection = SourceFilter().last_fetch_selection
        del selection  # instance-local; assert on the filtered candidates instead

        chosen = [item for item in filtered if item.should_fetch_full_page]
        self.assertIn(
            authoritative.url,
            [item.url for item in chosen],
            "the question names Merriam-Webster, so its page must be fetched",
        )
        self.assertTrue(authoritative.named_source_match)


if __name__ == "__main__":
    unittest.main()
