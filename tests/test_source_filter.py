from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.source_analyze.seer.source_filter import SourceFilter


class SourceSafetyFilterTests(unittest.TestCase):
    def test_blocks_github_gaia_json_trace(self):
        question = "Who wrote the quoted line in this GAIA task?"
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="GAIA validation answers",
            url="https://raw.githubusercontent.com/example/repo/main/gaia/tasks.json",
            snippet='{"task_id": "abc", "final_answer": "Alice", "messages": []}',
        )

        filtered = SourceFilter().filter_sources([source], question=question)

        self.assertEqual(filtered, [])
        self.assertTrue(source.blocked)
        self.assertEqual(source.block_reason, "benchmark_task_trace_leak")

    def test_blocks_task_id_like_uuid(self):
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="Search result",
            url="https://example.com/4b6bb5f7-f634-410e-815d-e673ab7f8632",
            snippet="Cached answer page",
        )

        filtered = SourceFilter().filter_sources([source], question="What is the answer?")

        self.assertEqual(filtered, [])
        self.assertEqual(source.block_reason, "benchmark_task_id_leak")
        self.assertIn("pre_fetch:task_id_like_uuid", source.filter_reasons)

    def test_blocks_dialogue_trace_after_fetch(self):
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="Normal page title",
            url="https://example.com/page",
            snippet="A normal search snippet.",
            raw_content=(
                "role: user\nWhat is the answer?\nrole: assistant\n"
                "Initial plan: inspect the task. We need answer. Final answer: Alice"
            ),
            fetched=True,
        )
        source_filter = SourceFilter()

        kept = source_filter.apply_post_fetch_safety([source], question="What is the answer?")

        self.assertEqual(kept, [])
        self.assertEqual(source.block_reason, "benchmark_dialogue_trace_leak")
        self.assertIn("post_fetch:dialogue_trace_marker", source.filter_reasons)

    def test_keeps_normal_source(self):
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="Taiwan - Wikipedia",
            url="https://en.wikipedia.org/wiki/Taiwan",
            snippet="Taiwan is an island country in East Asia with a documented history.",
        )

        filtered = SourceFilter().filter_sources([source], question="Where is Taiwan?")

        self.assertEqual([item.source_id for item in filtered], ["S1"])
        self.assertFalse(source.blocked)

    def test_question_echo_is_soft_signal_without_leak_marker(self):
        question = "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="Question mirror",
            url="https://example.com/mirror",
            snippet=question,
        )

        with patch(
            "tools.search_result_builder.source_analyze.seer.source_filter.semantic_similarity_score",
            return_value=0.97,
        ):
            filtered = SourceFilter().filter_sources([source], question=question)

        self.assertEqual([item.source_id for item in filtered], ["S1"])
        self.assertFalse(source.blocked)
        self.assertIn("question_semantic_echo", source.filter_reasons)


if __name__ == "__main__":
    unittest.main()
