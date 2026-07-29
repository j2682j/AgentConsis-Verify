"""Pin the cross-agent-majority safety net over corpus attestation.

`corpus_attestation` drops candidates the fetched pages never state — a strong
signal on text-search tasks. On level1_final_06 task 9d191bce (a YouTube video
question) it silenced the 4-run cross-agent answer 'extremely' in favour of a
chatty 2-run rival that happened to occur in unrelated fetched pages: 'Got it,
let's see'. The correct answer lived in the video, not the corpus, so absence
carried no information.

The guard keeps a rescue lane for candidates supported by enough runs across
enough distinct agents. Both thresholds have to fire — one agent repeating
itself is not a majority, so the run-count bound alone would rescue single-
agent guesses too.
"""

from __future__ import annotations

import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network


def _run(agent: str, idx: int, answer: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent,
        model_name="test-model",
        run_index=idx,
        raw_reply="",
        reasoning=f"step 1. Conclude {answer}.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )


def _summary(agent: str, runs: list[EachAgentReply]) -> AgentReasoningSummary:
    return AgentReasoningSummary(
        agent_id=agent,
        model_name="test-model",
        runs=runs,
        compressed_answer=runs[0].final_answer,
        compressed_reasoning=runs[0].reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=len(runs),
        eligible_run_count=len(runs),
    )


def _verifier(agent: str, answer: str, run_index: int = 1) -> VerifierScoreByReasoning:
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent,
        verifier_score=0.5,
        metadata={
            "candidate_key": answer.lower(),
            "target_run_index": run_index,
            "evidence_support": {"status": "no_support"},
            "process_verification": {
                "critical_step_floor": 0.5,
                "critical_step_geometric_mean": 0.5,
                "average_probability": 0.5,
            },
        },
    )


def _network() -> Network:
    """Three agents so a cross-agent majority is measurable."""
    return Network(
        "What did Teal'c say?",
        [
            AgentConfig(agent_id="gemma", model_name="test-model"),
            AgentConfig(agent_id="qwen", model_name="test-model"),
            AgentConfig(agent_id="nemotron", model_name="test-model"),
        ],
    )


def _evidence_with_only_the_chatty_answer_mentioned() -> dict[str, object]:
    """Fetched pages contain 'Got it' many times but never 'extremely'.

    Matches the 9d191bce failure shape: a common English phrase attests, while
    the correct answer lives in a video and is absent from the corpus.
    """
    filler_lines = "\n".join(f"Got it, let's see line {i}." for i in range(1, 8))
    return {
        "tool_usage": [
            {
                "raw_result": {
                    "retrieval": {
                        "rounds": [
                            {
                                "documents": [
                                    {"text": filler_lines},
                                ]
                            }
                        ]
                    }
                }
            }
        ]
    }


class CorpusAttestationMajorityOverrideTest(unittest.TestCase):
    def test_four_run_two_agent_majority_survives_corpus_silence(self) -> None:
        network = _network()

        results = [
            _summary("gemma", [_run("gemma", i, "extremely") for i in (1, 2)]),
            _summary("qwen", [_run("qwen", i, "extremely") for i in (1, 2)]),
            _summary(
                "nemotron",
                [_run("nemotron", 1, "Got it, let's see"), _run("nemotron", 2, "Got it, let's see")],
            ),
        ]
        verifiers = [
            _verifier("gemma", "extremely"),
            _verifier("qwen", "extremely"),
            _verifier("nemotron", "Got it, let's see"),
        ]

        winner = network._select_winner(
            results,
            verifier_results=verifiers,
            evidence=_evidence_with_only_the_chatty_answer_mentioned(),
        )

        self.assertEqual(winner.compressed_answer, "extremely")

    def test_disabling_the_guard_restores_the_broken_behaviour(self) -> None:
        """Threshold 0 lets corpus silence dominate again."""
        network = _network()
        network.final_winner_selector.attestation_majority_override_min_runs = 0

        results = [
            _summary("gemma", [_run("gemma", i, "extremely") for i in (1, 2)]),
            _summary("qwen", [_run("qwen", i, "extremely") for i in (1, 2)]),
            _summary(
                "nemotron",
                [_run("nemotron", 1, "Got it, let's see"), _run("nemotron", 2, "Got it, let's see")],
            ),
        ]
        verifiers = [
            _verifier("gemma", "extremely"),
            _verifier("qwen", "extremely"),
            _verifier("nemotron", "Got it, let's see"),
        ]

        winner = network._select_winner(
            results,
            verifier_results=verifiers,
            evidence=_evidence_with_only_the_chatty_answer_mentioned(),
        )

        self.assertEqual(winner.compressed_answer, "Got it, let's see")

    def test_one_agent_repeating_itself_is_not_a_rescue(self) -> None:
        """Three runs from a single agent should still lose to a corpus-attested rival."""
        network = _network()

        results = [
            _summary("gemma", [_run("gemma", i, "jan wagner") for i in (1, 2, 3)]),
            _summary("nemotron", [_run("nemotron", 1, "Got it, let's see")]),
        ]
        verifiers = [
            _verifier("gemma", "jan wagner"),
            _verifier("nemotron", "Got it, let's see"),
        ]

        winner = network._select_winner(
            results,
            verifier_results=verifiers,
            evidence=_evidence_with_only_the_chatty_answer_mentioned(),
        )

        # The corpus-attested rival still wins because a single agent repeating
        # itself does not clear the 2-agent bound.
        self.assertEqual(winner.compressed_answer, "Got it, let's see")

    def test_two_run_majority_is_not_enough_to_rescue(self) -> None:
        """The run-count floor is 3 by default; 2-run majorities keep losing."""
        network = _network()

        results = [
            _summary("gemma", [_run("gemma", 1, "16000")]),
            _summary("qwen", [_run("qwen", 1, "16000")]),
            _summary("nemotron", [_run("nemotron", 1, "Got it, let's see")]),
        ]
        verifiers = [
            _verifier("gemma", "16000"),
            _verifier("qwen", "16000"),
            _verifier("nemotron", "Got it, let's see"),
        ]

        winner = network._select_winner(
            results,
            verifier_results=verifiers,
            evidence=_evidence_with_only_the_chatty_answer_mentioned(),
        )

        self.assertEqual(winner.compressed_answer, "Got it, let's see")

    def test_rescue_thresholds_can_be_disabled_per_bound(self) -> None:
        """Each threshold guards independently; either being 0 disables the rescue."""

        for run_bound, agent_bound in ((0, 2), (3, 0)):
            with self.subTest(run_bound=run_bound, agent_bound=agent_bound):
                network = _network()
                network.final_winner_selector.attestation_majority_override_min_runs = run_bound
                network.final_winner_selector.attestation_majority_override_min_agents = agent_bound

                results = [
                    _summary("gemma", [_run("gemma", i, "extremely") for i in (1, 2)]),
                    _summary("qwen", [_run("qwen", i, "extremely") for i in (1, 2)]),
                    _summary(
                        "nemotron",
                        [_run("nemotron", 1, "Got it, let's see"), _run("nemotron", 2, "Got it, let's see")],
                    ),
                ]
                verifiers = [
                    _verifier("gemma", "extremely"),
                    _verifier("qwen", "extremely"),
                    _verifier("nemotron", "Got it, let's see"),
                ]

                winner = network._select_winner(
                    results,
                    verifier_results=verifiers,
                    evidence=_evidence_with_only_the_chatty_answer_mentioned(),
                )

                self.assertEqual(winner.compressed_answer, "Got it, let's see")


if __name__ == "__main__":
    unittest.main()
