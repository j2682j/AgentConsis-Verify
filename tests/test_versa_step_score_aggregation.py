"""The verifier ranks on the median over every step, not the critical floor.

`critical_step_floor` held this role until it was measured: over final_13/15/16,
on the 58 tasks that produced both a correct and an incorrect candidate, it
ranked the correct one higher on 0.399 of within-task pairs -- worse than a coin
flip, and last of the six statistics compared. The median over every step scored
0.577, and beat the floor 31W-12L by task-level sign test (p = 0.0054).

Two defects, not one, and the tests here pin both. The floor is a worst-case
statistic over a chain averaging 14.7 steps, so one weak step decided a
candidate even though correct chains carry weak steps as readily as wrong ones.
And the critical subset has a median size of 1, so for half the paths every
statistic collapses to the same value -- which is why the median confined to
that subset recovers almost nothing (0.491) while the median over all steps
does (0.577, p = 0.041 against it).

The suite passed unchanged when the ranking key was swapped, because its
verifier helper sets floor, geometric mean and average to one probability and
cannot tell them apart. These cases give them different values on purpose.
"""

from __future__ import annotations

import unittest

from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply, VerifierScoreByReasoning
from core.network import Network
from core.stage2_runner import Stage2Runner


class FakeVersaScorer:
    """Stage2Runner builds a real scorer otherwise, which loads a model."""

    def score_steps(self, *, question: str, reasoning_steps: list[tuple[int, str]]):
        raise AssertionError("the summariser under test does not score")


def summariser() -> Stage2Runner:
    return Stage2Runner(
        question="q",
        agents=[AgentConfig(agent_id="a1", model_name="test-model")],
        versa_scorer=FakeVersaScorer(),
    )


def step(index: int, text: str, probability: float, support: str = "unknown") -> dict:
    return {
        "step_index": index,
        "step_text": text,
        "reward_probability": probability,
        "support_status": support,
    }


def make_run(agent_id: str, answer: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=1,
        raw_reply="",
        reasoning=f"step 1. Conclude {answer}.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        structured_output={},
        schema_valid=True,
        eligible_for_winner=True,
    )


def make_summary(agent_id: str, answer: str) -> AgentReasoningSummary:
    runs = [make_run(agent_id, answer)]
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=runs,
        compressed_answer=answer,
        compressed_reasoning=runs[0].reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=len(runs),
        eligible_run_count=len(runs),
    )


def verifier(
    agent_id: str,
    answer: str,
    *,
    median: float,
    floor: float,
    geometric_mean: float,
) -> VerifierScoreByReasoning:
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent_id,
        verifier_score=median,
        metadata={
            "candidate_key": answer.lower(),
            "target_run_index": 1,
            "evidence_support": {"status": "no_support"},
            "process_verification": {
                "step_score_median": median,
                "critical_step_floor": floor,
                "critical_step_geometric_mean": geometric_mean,
                "average_probability": median,
            },
        },
    )


class StepScoreMedianTests(unittest.TestCase):
    def test_median_covers_every_step_not_the_critical_subset(self) -> None:
        """The subset restriction cost more than the choice of statistic."""

        runner = summariser()
        steps = [
            step(1, "Read the table.", 0.90),
            step(2, "Take the second row.", 0.95),
            step(3, "Compare the two totals.", 0.85),
            # Supported, so critical -- and the only low value in the chain.
            step(4, "Evidence gives the total as 12.", 0.10, support="supported"),
            step(5, "The answer is 12.", 0.92),
        ]

        summary = runner._process_verification_summary(step_scores=steps, final_answer="12")

        # Critical steps are 4 (supported) and 5 (last), so the floor sees 0.10
        # while the chain is strong everywhere else.
        self.assertEqual(summary["critical_step_indices"], [4, 5])
        self.assertAlmostEqual(summary["critical_step_floor"], 0.10)
        self.assertAlmostEqual(summary["step_score_median"], 0.90)

    def test_median_is_reported_when_no_step_scores_arrive(self) -> None:
        summary = summariser()._process_verification_summary(step_scores=[], final_answer="12")

        self.assertEqual(summary["step_score_median"], 0.0)
        self.assertEqual(summary["critical_step_floor"], 0.0)

    def test_gate_prefers_the_higher_median_over_the_higher_floor(self) -> None:
        """The case the measurement is about: one weak step in a strong chain."""

        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        network = Network("Choose A or B.", configs)

        winner = network._select_winner(
            [make_summary("a1", "A"), make_summary("a2", "B")],
            verifier_results=[
                # A strong chain that dips once.
                verifier("a1", "A", median=0.91, floor=0.10, geometric_mean=0.55),
                # Mediocre throughout, but its worst step is higher.
                verifier("a2", "B", median=0.60, floor=0.50, geometric_mean=0.58),
            ],
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "A")
        self.assertEqual(
            network._last_winner_selection_trace["gate_trace"][-1]["gate_name"],
            "versa_verification",
        )

    def test_geometric_mean_still_breaks_a_tied_median(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        network = Network("Choose A or B.", configs)

        winner = network._select_winner(
            [make_summary("a1", "A"), make_summary("a2", "B")],
            verifier_results=[
                verifier("a1", "A", median=0.80, floor=0.90, geometric_mean=0.40),
                verifier("a2", "B", median=0.80, floor=0.20, geometric_mean=0.70),
            ],
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "B")


if __name__ == "__main__":
    unittest.main()
