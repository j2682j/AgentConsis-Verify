"""Presentation-only differences must not split an agent's own votes."""

from __future__ import annotations

import unittest

from core.config import EachAgentReply
from score.agent_answer_aggregator import AgentAnswerAggregator


def run(index: int, answer: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id="a1",
        model_name="test-model",
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


class SurfaceFormAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = AgentAnswerAggregator()

    def test_boxed_wrapper_is_not_reported_as_the_answer(self) -> None:
        """A wrapper on the first run used to become the agent's answer.

        It then failed to cluster with another agent's plain form and split
        the cross-agent vote.
        """
        result = self.aggregator.aggregate(
            [run(1, r"$\boxed{2}$"), run(2, "2"), run(3, "3")]
        )
        self.assertEqual(result.answer, "2")
        self.assertEqual(result.confidence_score, 0.67)

    def test_separator_spacing_does_not_split_agreeing_runs(self) -> None:
        """"a, b" and "a,b" are the same list."""
        result = self.aggregator.aggregate(
            [
                run(1, "broccoli, celery, fresh basil"),
                run(2, "broccoli,celery,fresh basil"),
                run(3, "broccoli, celery, corn"),
            ]
        )
        self.assertEqual(result.answer, "broccoli, celery, fresh basil")
        self.assertEqual(result.confidence_score, 0.67)
        self.assertEqual(result.status, "consensus_2_of_3")

    def test_spacing_variants_reach_full_consensus(self) -> None:
        result = self.aggregator.aggregate(
            [run(1, "b, e"), run(2, "b,e"), run(3, "b, e")]
        )
        self.assertEqual(result.answer, "b, e")
        self.assertEqual(result.confidence_score, 1.0)

    def test_answer_text_is_never_shortened_to_a_variant(self) -> None:
        """Guards a fix that was nearly made worse.

        Ranking group members by length rewrote answers that only looked
        equivalent: "FF0099FF" became "0099FF" and "b, e" became "b,e".
        The representative must stay a form an agent actually produced.
        """
        for answers in (["FF0099FF", "FF0099FF"], ["b, e", "b, e"]):
            with self.subTest(answers=answers):
                result = self.aggregator.aggregate(
                    [run(index, value) for index, value in enumerate(answers, start=1)]
                )
                self.assertEqual(result.answer, answers[0])

    def test_currency_is_not_treated_as_a_math_wrapper(self) -> None:
        result = self.aggregator.aggregate([run(1, "$5.00"), run(2, "$5.00")])
        self.assertEqual(result.answer, "$5.00")

    def test_plain_answers_are_untouched(self) -> None:
        result = self.aggregator.aggregate(
            [run(1, "Saint Petersburg"), run(2, "Saint Petersburg"), run(3, "Lyon")]
        )
        self.assertEqual(result.answer, "Saint Petersburg")
        self.assertEqual(result.confidence_score, 0.67)


if __name__ == "__main__":
    unittest.main()
