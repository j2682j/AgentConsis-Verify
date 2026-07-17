from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.search_result_builder.query import (
    ClassifiedSpan,
    QueryCoverageChecker,
    QueryGenerator,
    SalienceQueryCandidate,
    SourceRequirement,
)


class FakeCandidateGenerator:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.last_salient_spans = [
            SimpleNamespace(text="Merriam-Webster Word of the Day", score=1.0),
            SimpleNamespace(text="June 27 2022", score=0.8),
        ]
        self.last_classified_spans = [
            ClassifiedSpan(
                text="Merriam-Webster Word of the Day",
                role="source_clue",
                confidence=0.2,
                score=1.0,
            ),
            ClassifiedSpan(
                text="June 27 2022",
                role="constraint",
                confidence=0.1,
                score=0.8,
            ),
            ClassifiedSpan(
                text="answer with the writer",
                role="answer_target",
                confidence=0.08,
                score=0.4,
            ),
        ]

    def generate(self, question, *, num_candidates):
        del question, num_candidates
        return [
            SalienceQueryCandidate(
                query=query,
                matched_spans=[],
                coverage_score=0.0,
                score=0.0,
                source_requirement=SourceRequirement(
                    source_kind="academic",
                    access_mode="search",
                    source_hint="merriam-webster.com",
                ),
            )
            for query in self.queries
        ]


class NoIntentQueryGeneratorTests(unittest.TestCase):
    def test_query_state_is_built_from_classified_spans(self):
        generator = QueryGenerator(
            generator=FakeCandidateGenerator(
                [
                    "Merriam-Webster Word of the Day June 27 2022",
                    "June 27 2022 Word of the Day",
                ]
            ),
            coverage_checker=QueryCoverageChecker(),
        )

        plan = generator.plan(
            "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?",
            max_queries=2,
        )

        self.assertEqual(plan["intent_planning"], "disabled")
        self.assertEqual(plan["search_intent_plan"]["source"], "embedding_span_role_classifier")
        self.assertIn("Merriam-Webster Word of the Day", plan["search_intent_plan"]["must_include"])
        self.assertIn("June 27 2022", plan["search_intent_plan"]["must_include"])
        self.assertEqual(plan["search_intent_plan"]["answer_role"], "answer with the writer")
        self.assertEqual(len(plan["classified_spans"]), 3)
        self.assertEqual(len(plan["query_requests"]), 2)
        self.assertEqual(
            plan["query_requests"][0]["source_requirement"]["source_kind"],
            "academic",
        )


if __name__ == "__main__":
    unittest.main()
