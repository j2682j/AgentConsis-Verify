from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.llm_client import LLMChatResult
from tools.search_result_builder.query import (
    QueryCoverageChecker,
    QueryGenerator,
    SalienceQueryCandidate,
    SearchIntentPlan,
    SearchIntentPlanner,
)


class FakeLLMClient:
    def __init__(self, content: str, provider: str = "ollama") -> None:
        self.content = content
        self.provider = provider
        self.native_calls = []
        self.calls = []

    def ollama_native_chat(self, **kwargs):
        self.native_calls.append(kwargs)
        return LLMChatResult(content=self.content)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMChatResult(content=self.content)


class FakeCandidateGenerator:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.last_salient_spans = [SimpleNamespace(text="Word of the Day")]

    def generate(self, question, *, num_candidates, intent_plan=None):
        del question, num_candidates, intent_plan
        return [
            SalienceQueryCandidate(
                query=query,
                matched_spans=[],
                coverage_score=0.0,
                score=0.0,
            )
            for query in self.queries
        ]


class FakeIntentPlanner:
    def __init__(self, plan: SearchIntentPlan) -> None:
        self.plan_value = plan

    def plan(self, question):
        del question
        return self.plan_value


class SearchIntentPlannerTests(unittest.TestCase):
    def test_planner_parses_minimal_json_and_uses_ollama_no_think(self):
        client = FakeLLMClient(
            '{"search_needed": true, "intent": "official_page", '
            '"target": "Find the official Merriam-Webster page.", '
            '"must_include": ["Merriam-Webster", "June 27 2022"], '
            '"avoid_terms": ["author"], '
            '"preferred_domain": "merriam-webster.com"}'
        )
        planner = SearchIntentPlanner(model_name="qwen3:4b", llm_client=client)

        plan = planner.plan("What writer is quoted by Merriam-Webster?")

        self.assertTrue(plan.search_needed)
        self.assertEqual(plan.intent, "official_page")
        self.assertEqual(plan.preferred_domain, "merriam-webster.com")
        self.assertIn("Merriam-Webster", plan.must_include)
        self.assertEqual(len(client.native_calls), 1)
        self.assertFalse(client.native_calls[0]["think"])

    def test_query_generator_prefers_intent_seed_and_drops_avoid_terms(self):
        intent = SearchIntentPlan(
            search_needed=True,
            intent="official_page",
            target="Find the official Merriam-Webster Word of the Day page for June 27 2022.",
            must_include=["Merriam-Webster", "Word of the Day", "June 27 2022"],
            avoid_terms=["author"],
            preferred_domain="merriam-webster.com",
        )
        generator = QueryGenerator(
            generator=FakeCandidateGenerator(
                [
                    "Merriam-Webster June 27 2022 Word of the Day author",
                    "Who quoted Merriam-Webster Word of the Day June 27 2022",
                ]
            ),
            coverage_checker=QueryCoverageChecker(),
            intent_planner=FakeIntentPlanner(intent),
        )

        plan = generator.plan(
            "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?",
            max_queries=3,
        )

        self.assertEqual(
            plan["queries"][0],
            "site:merriam-webster.com Merriam-Webster Word of the Day June 27 2022",
        )
        self.assertNotIn("author", " ".join(plan["queries"]).lower())
        self.assertEqual(plan["search_intent_plan"]["intent"], "official_page")
        self.assertTrue(plan["query_coverage"]["query_results"][0]["preferred_domain_used"])

    def test_query_generator_returns_empty_queries_when_no_search(self):
        intent = SearchIntentPlan(
            search_needed=False,
            intent="no_search",
            target="Solve locally.",
            must_include=[],
            avoid_terms=[],
            preferred_domain="",
        )
        generator = QueryGenerator(
            generator=FakeCandidateGenerator(["should not be used"]),
            coverage_checker=QueryCoverageChecker(),
            intent_planner=FakeIntentPlanner(intent),
        )

        plan = generator.plan("A local puzzle that needs no web search.", max_queries=3)

        self.assertEqual(plan["queries"], [])
        self.assertFalse(plan["precision_needed"])


if __name__ == "__main__":
    unittest.main()
