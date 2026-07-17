from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.search_result_builder.query import (
    QueryCoverageChecker,
    SalienceQueryCandidate,
)


class FakeGenerator:
    def __init__(self, candidates):
        self.candidates = candidates
        self.last_salient_spans = [
            SimpleNamespace(text="Nedoshivina 2010"),
            SimpleNamespace(text="Kuznetzov Vietnamese specimens"),
        ]

    def generate(self, question, *, num_candidates):
        del question, num_candidates
        return self.candidates

class QueryCoverageTests(unittest.TestCase):
    def checker(self) -> QueryCoverageChecker:
        checker = QueryCoverageChecker(min_repair_coverage=0.6)
        checker._spacy_entities = lambda text: ["Nedoshivina", "Kuznetzov"]  # type: ignore[method-assign]
        return checker

    def test_coverage_score_is_direct_constraint_ratio(self):
        checker = self.checker()
        constraints = checker.extract_constraints(
            question="In Nedoshivina's 2010 paper, where were Kuznetzov specimens deposited? Give the city.",
            salient_spans=["Nedoshivina 2010", "Kuznetzov specimens"],
        )

        result = checker.score_query(
            "Nedoshivina 2010 paper",
            constraints,
            original_index=0,
        )

        self.assertEqual(
            result.coverage_score,
            round(len(result.covered) / len(constraints), 6),
        )
        self.assertGreater(len(result.missing), 0)

    def test_low_coverage_adds_repair_query(self):
        checker = self.checker()

        queries, diagnostics = checker.improve_queries(
            question="In Nedoshivina's 2010 paper, where were Kuznetzov Vietnamese specimens deposited? Give the city.",
            queries=["Vietnamese specimens deposited"],
            salient_spans=["Nedoshivina 2010", "Kuznetzov Vietnamese specimens"],
            max_queries=3,
        )

        self.assertTrue(diagnostics["repair_added"])
        self.assertEqual(queries[0], diagnostics["repair_query"])
        self.assertIn("Nedoshivina", queries[0])
        self.assertIn("2010", queries[0])
        self.assertIn("Kuznetzov", queries[0])

    def test_reranks_higher_coverage_query_first_without_repair(self):
        checker = self.checker()

        queries, diagnostics = checker.improve_queries(
            question="In Nedoshivina's 2010 paper, where were Kuznetzov Vietnamese specimens deposited? Give the city.",
            queries=[
                "Vietnamese specimens deposited",
                "Nedoshivina 2010 Kuznetzov Vietnamese specimens deposited city",
            ],
            salient_spans=["Nedoshivina 2010", "Kuznetzov Vietnamese specimens"],
            max_queries=2,
        )

        self.assertFalse(diagnostics["repair_added"])
        self.assertEqual(
            queries[0],
            "Nedoshivina 2010 Kuznetzov Vietnamese specimens deposited city",
        )

    def test_query_generator_keeps_model_order_and_records_coverage_diagnostics(self):
        from tools.search_result_builder.query.query_generator import QueryGenerator

        candidates = [
            SalienceQueryCandidate(
                query="Vietnamese specimens deposited",
                matched_spans=[],
                coverage_score=0.0,
                score=0.0,
            ),
            SalienceQueryCandidate(
                query="Nedoshivina 2010 Kuznetzov Vietnamese specimens deposited city",
                matched_spans=[],
                coverage_score=0.0,
                score=0.0,
            ),
        ]
        checker = self.checker()
        generator = QueryGenerator(
            generator=FakeGenerator(candidates),
            coverage_checker=checker,
        )

        plan = generator.plan(
            "In Nedoshivina's 2010 paper, where were Kuznetzov Vietnamese specimens deposited? Give the city.",
            max_queries=2,
        )

        self.assertEqual(
            plan["queries"][0],
            "Vietnamese specimens deposited",
        )
        self.assertEqual(
            plan["query_requests"][0]["query"],
            "Vietnamese specimens deposited",
        )
        self.assertEqual(
            plan["query_coverage"]["score_formula"],
            "covered_constraints / total_constraints",
        )


if __name__ == "__main__":
    unittest.main()
