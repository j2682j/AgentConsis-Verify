"""Pin that runs an agent discarded do not count as independent agreement.

The consensus rank breaks ties on distinct supporting agents, which is right in
principle -- two agents agreeing beats one agent repeating itself. But a run
counts toward a candidate even when its own agent discarded it, so two agents'
leftover runs colliding on the same value look identical to two agents agreeing.

level1_final_08 task dc28cf18 is the failure: candidate '3' was one rejected run
from nemotron (which had voted 2 of 3 for '2') plus one from gemma (which settled
on '9'), and on equal run counts that collision outranked nemotron's own answer.

So agents that settled on a value rank ahead of agents that merely produced it
somewhere, and only then does raw agent breadth apply.
"""

from __future__ import annotations

import unittest

from core.config import AgentConfig
from core.network import Network
from test_ordered_winner_gates import make_run, make_summary

QUESTION = "How many albums were released?"


def _select():
    """Rebuild task dc28cf18: leftovers colliding against a settled 2-of-3."""

    configs = [
        AgentConfig(agent_id="nemotron", model_name="test-model"),
        AgentConfig(agent_id="gemma", model_name="test-model"),
    ]
    results = [
        # Voted 2 of 3 for '2'; its run 1 said '3' and was discarded.
        make_summary(
            "nemotron",
            [
                make_run("nemotron", 1, "3"),
                make_run("nemotron", 2, "2"),
                make_run("nemotron", 3, "2"),
            ],
            answer="2",
            confidence=0.67,
        ),
        # Settled on '9'; its run 3 said '3' and was discarded.
        make_summary(
            "gemma",
            [
                make_run("gemma", 1, "9"),
                make_run("gemma", 2, "9"),
                make_run("gemma", 3, "3"),
            ],
            answer="9",
            confidence=0.67,
        ),
    ]
    network = Network(QUESTION, configs)
    winner = network._select_winner(results, evidence={})
    return winner, network._last_winner_selection_trace


class ConsensusRankCountsSettledAgentsTest(unittest.TestCase):
    def test_leftover_collision_does_not_outrank_a_settled_answer(self) -> None:
        winner, _ = _select()

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "2")

    def test_the_collision_candidate_exists_with_equal_runs(self) -> None:
        """Guard the setup: without equal run counts the tie-break never runs."""

        _, trace = _select()
        by_key = {
            item["candidate_key"]: item for item in trace["candidates"]
        }

        self.assertIn("3", by_key)
        self.assertEqual(by_key["3"]["supporting_run_count"], 2)
        self.assertEqual(by_key["2"]["supporting_run_count"], 2)
        self.assertEqual(
            sorted(by_key["3"]["supporting_agent_ids"]), ["gemma", "nemotron"]
        )

    def test_two_agents_genuinely_settling_still_wins_on_breadth(self) -> None:
        """The original principle has to survive: real agreement still ranks up."""

        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "7"), make_run("a1", 2, "7")],
                answer="7",
                confidence=1.0,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "7"), make_run("a2", 2, "5")],
                answer="7",
                confidence=0.5,
            ),
        ]
        network = Network(QUESTION, configs)

        winner = network._select_winner(results, evidence={})

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "7")


if __name__ == "__main__":
    unittest.main()
