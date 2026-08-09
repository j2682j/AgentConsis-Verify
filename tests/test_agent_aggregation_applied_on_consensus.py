"""Pin that an Agent's own vote decides its compressed answer.

`_summarize_with_aggregation` runs two summarisers over the same runs: the
Stage1 summary, and the answer aggregator that actually counts the votes. The
aggregator's result used to be applied only when its status was needs_review --
the case where every run disagrees and the count means least -- and dropped on
consensus, where it means most.

That inversion is visible in both saved runs. On task 1f975693 qwen voted 2 of 3
for 'Saint Petersburg' while the summary kept the 'St. Petersburg' of run 1,
which the exact-match scorer counts as wrong; and a 2-of-3 split reported
confidence 1.00 into winner selection rather than 0.67.

These tests hold the aggregator as the source of truth whenever it produced an
answer, across all three vote shapes.
"""

from __future__ import annotations

import unittest

from core.config import AgentConfig, EachAgentReply
from core.stage1_runner import Stage1Runner
from score.agent_answer_aggregator import AgentAnswerAggregator
from score.stage1_aggregator import Stage1Aggregator


def _run(index: int, answer: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id="qwen",
        model_name="qwen3:4b",
        run_index=index,
        raw_reply="",
        reasoning=f"step 1. Conclude {answer}.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        structured_output={},
        schema_valid=True,
        eligible_for_winner=True,
    )


def _summarize(answers: list[str]):
    """Drive only the aggregation path, without Stage1Runner's dependencies."""

    runner = Stage1Runner.__new__(Stage1Runner)
    runner.aggregator = Stage1Aggregator()
    runner.answer_aggregator = AgentAnswerAggregator()
    config = AgentConfig(agent_id="qwen", model_name="qwen3:4b")
    runs = [_run(index, answer) for index, answer in enumerate(answers, start=1)]
    summary = runner._summarize_with_aggregation(config, runs)
    return summary, config


class AggregationAppliedOnConsensusTest(unittest.TestCase):
    def test_two_of_three_consensus_decides_the_compressed_answer(self) -> None:
        summary, _ = _summarize(
            ["St. Petersburg", "Saint Petersburg", "Saint Petersburg"]
        )

        self.assertEqual(summary.aggregation_metadata["status"], "consensus_2_of_3")
        self.assertEqual(summary.compressed_answer, "Saint Petersburg")

    def test_two_of_three_reports_its_real_confidence(self) -> None:
        summary, config = _summarize(
            ["St. Petersburg", "Saint Petersburg", "Saint Petersburg"]
        )

        self.assertAlmostEqual(summary.confidence_score, 0.67, places=2)
        self.assertAlmostEqual(config.confidence_score, 0.67, places=2)

    def test_unanimous_runs_keep_their_answer_and_full_confidence(self) -> None:
        summary, _ = _summarize(["Saint Petersburg"] * 3)

        self.assertEqual(summary.aggregation_metadata["status"], "consensus_3_of_3")
        self.assertEqual(summary.compressed_answer, "Saint Petersburg")
        self.assertAlmostEqual(summary.confidence_score, 1.0, places=2)

    def test_all_runs_differing_still_takes_the_aggregator_answer(self) -> None:
        """The needs_review path that already worked must keep working."""

        summary, _ = _summarize(["GBR", "CUB", "NLD"])

        self.assertTrue(summary.aggregation_metadata["needs_review"])
        self.assertEqual(summary.compressed_answer, summary.aggregation_metadata["answer"])


if __name__ == "__main__":
    unittest.main()
