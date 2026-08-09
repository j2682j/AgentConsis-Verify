"""Pin that an inferred answer type cannot hard-reject the leading answer.

The answer requirement's expected type is inferred from the question, so it can
be wrong. When it is, this gate rejects the candidates that declared a type
honestly and lets candidates declaring nothing through as merely undecided --
which selects against well-formed answers.

Task 388a80fd loses this way in level1_final_06, _07 and _08 alike. It asks for
a sentence hidden in a 5x7 block of letters; the requirement infers `list`, so
the sentence held by three runs is rejected as incompatible while a one-run
garbled variant survives and wins.

Replaying the three runs' recorded candidates, the guard spares four rejections
and all four are the correct answer, while the 22 other rejections stand --
including every rejection on a task that is currently answered correctly.
"""

from __future__ import annotations

import unittest

from core.config import AgentConfig
from core.network import Network
from test_ordered_winner_gates import make_run, make_summary

SENTENCE = "The seagull glided peacefully to my chair"
GARBLED = "THESE AGULL GLIDE DPEAC"
QUESTION = "Pull out the sentence in the following 5x7 block of text."


def _select(sentence_runs: int, garbled_runs: int, *, guard: int | None = None):
    """Two candidates: a declared-text sentence and an undeclared garble."""

    configs = [
        AgentConfig(agent_id="sentence", model_name="test-model"),
        AgentConfig(agent_id="garbled", model_name="test-model"),
    ]
    results = [
        make_summary(
            "sentence",
            [
                make_run("sentence", index, SENTENCE, answer_type="text")
                for index in range(1, sentence_runs + 1)
            ],
            answer=SENTENCE,
            confidence=1.0,
        ),
        make_summary(
            "garbled",
            [
                make_run("garbled", index, GARBLED)
                for index in range(1, garbled_runs + 1)
            ],
            answer=GARBLED,
            confidence=1.0,
        ),
    ]
    network = Network(QUESTION, configs)
    if guard is not None:
        network.final_winner_selector.requirement_reject_support_guard = guard
    winner = network._select_winner(
        results,
        evidence={"answer_requirement": QUESTION, "answer_role": "list"},
    )
    return winner, network._last_winner_selection_trace


class RequirementRejectSupportGuardTest(unittest.TestCase):
    def test_multi_run_sentence_survives_a_wrong_type_inference(self) -> None:
        winner, _ = _select(sentence_runs=3, garbled_runs=1)

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, SENTENCE)

    def test_a_tie_on_runs_still_spares_the_rejection(self) -> None:
        """level1_final_08 shape: 3 runs against a 3-run survivor."""

        _, trace = _select(sentence_runs=3, garbled_runs=3)
        gate = next(
            item for item in trace["gate_trace"]
            if item["gate_name"] == "answer_requirement"
        )

        self.assertIn(
            "answer_requirement_incompatible_but_best_supported",
            [decision["reason"] for decision in gate["decisions"]],
        )

    def test_a_single_run_rejection_still_stands(self) -> None:
        """Guard the other direction: one run cannot outweigh the requirement."""

        _, trace = _select(sentence_runs=1, garbled_runs=3)
        gate = next(
            item for item in trace["gate_trace"]
            if item["gate_name"] == "answer_requirement"
        )
        rejected = [
            decision["candidate_key"]
            for decision in gate["decisions"]
            if decision["outcome"] == "reject"
        ]

        self.assertTrue(rejected)

    def test_a_rejection_below_the_best_survivor_still_stands(self) -> None:
        """level1_final_08 task 024 shape: 2 runs against a 4-run survivor."""

        _, trace = _select(sentence_runs=2, garbled_runs=4)
        gate = next(
            item for item in trace["gate_trace"]
            if item["gate_name"] == "answer_requirement"
        )
        rejected = [
            decision["candidate_key"]
            for decision in gate["decisions"]
            if decision["outcome"] == "reject"
        ]

        self.assertTrue(rejected)

    def test_guard_can_be_disabled(self) -> None:
        _, trace = _select(sentence_runs=3, garbled_runs=1, guard=0)
        gate = next(
            item for item in trace["gate_trace"]
            if item["gate_name"] == "answer_requirement"
        )
        rejected = [
            decision["candidate_key"]
            for decision in gate["decisions"]
            if decision["outcome"] == "reject"
        ]

        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
