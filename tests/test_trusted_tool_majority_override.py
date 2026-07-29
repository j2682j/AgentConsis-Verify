"""Pin the trusted-tool vs cross-agent-majority carve-out.

On level1_final_06 task 7bd855d8 the tool answer '18028' was labelled
tool_final_supported by two runs, while '89706.00' came from five runs across
two agents with no evidence label. The evidence gate then filtered '89706.00'
into the reserve pool because its bucket ranked below trusted_tool_final,
leaving the consensus gate with only the trusted answer to pick. The tool label
overrode a 5-vs-2 majority.

`trusted_tool_majority_override_ratio` (default 2.0) keeps both answers in the
survivor set whenever a rival's supporting_run_count strictly exceeds the
trusted answer's count times the ratio, so the consensus gate downstream can
prefer the majority. The guard applies only to TRUSTED_TOOL_FINAL because the
other buckets already share a soft-signal path.
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


def _verifier(
    agent: str, answer: str, *, support_status: str, run_index: int = 1
) -> VerifierScoreByReasoning:
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent,
        verifier_score=0.5,
        metadata={
            "candidate_key": answer.lower(),
            "target_run_index": run_index,
            "evidence_support": {"status": support_status},
            "process_verification": {
                "critical_step_floor": 0.5,
                "critical_step_geometric_mean": 0.5,
                "average_probability": 0.5,
            },
        },
    )


def _network() -> Network:
    """Three agents so a 5-run majority is achievable across agents."""
    return Network(
        "What was the total in USD?",
        [
            AgentConfig(agent_id="qwen", model_name="test-model"),
            AgentConfig(agent_id="gemma", model_name="test-model"),
            AgentConfig(agent_id="nemotron", model_name="test-model"),
        ],
    )


def _seven_bd_scenario_results() -> list[AgentReasoningSummary]:
    """Reproduces the task-7bd855d8 vote distribution: 5 correct vs 2 trusted-wrong."""
    return [
        _summary("qwen", [_run("qwen", i, "89706.00") for i in (1, 2, 3)]),
        _summary(
            "gemma",
            [_run("gemma", 1, "18028"), _run("gemma", 2, "89706.00"), _run("gemma", 3, "89706.00")],
        ),
        _summary(
            "nemotron",
            [_run("nemotron", 1, "18028"), _run("nemotron", 2, "19867.00"), _run("nemotron", 3, "18079")],
        ),
    ]


def _seven_bd_verifiers() -> list[VerifierScoreByReasoning]:
    """Only the wrong '18028' carries the tool_final_supported label."""
    return [
        _verifier("qwen", "89706.00", support_status="no_support", run_index=1),
        _verifier("gemma", "89706.00", support_status="no_support", run_index=2),
        _verifier("nemotron", "18028", support_status="tool_final_supported", run_index=1),
    ]


class TrustedToolMajorityOverrideTest(unittest.TestCase):
    def test_five_run_majority_wins_over_two_run_trusted_tool_answer(self) -> None:
        network = _network()

        winner = network._select_winner(
            _seven_bd_scenario_results(),
            verifier_results=_seven_bd_verifiers(),
            evidence={},
        )

        self.assertEqual(winner.compressed_answer, "89706.00")

    def test_disabling_the_guard_restores_the_broken_behaviour(self) -> None:
        """Ratio 0.0 lets the trusted label dominate again — the regression path."""
        network = _network()
        network.final_winner_selector.trusted_tool_majority_override_ratio = 0.0

        winner = network._select_winner(
            _seven_bd_scenario_results(),
            verifier_results=_seven_bd_verifiers(),
            evidence={},
        )

        self.assertEqual(winner.compressed_answer, "18028")

    def test_trusted_wins_when_it_has_the_majority(self) -> None:
        """Guard only fires against clear majorities, not co-supported trusted answers."""
        network = _network()
        results = [
            _summary("qwen", [_run("qwen", i, "TOOL-ANS") for i in (1, 2, 3)]),
            _summary("gemma", [_run("gemma", i, "TOOL-ANS") for i in (1, 2, 3)]),
            _summary("nemotron", [_run("nemotron", 1, "guessed")]),
        ]
        verifiers = [
            _verifier("qwen", "TOOL-ANS", support_status="tool_final_supported", run_index=1),
            _verifier("nemotron", "guessed", support_status="no_support"),
        ]

        winner = network._select_winner(results, verifier_results=verifiers, evidence={})

        self.assertEqual(winner.compressed_answer, "TOOL-ANS")

    def test_trusted_wins_when_rival_is_only_slightly_larger(self) -> None:
        """3 vs 2 does not exceed 2x threshold; trusted label still dominates."""
        network = _network()
        results = [
            _summary("qwen", [_run("qwen", i, "TOOL-ANS") for i in (1, 2)]),
            _summary("gemma", [_run("gemma", i, "rival") for i in (1, 2, 3)]),
        ]
        verifiers = [
            _verifier("qwen", "TOOL-ANS", support_status="tool_final_supported"),
            _verifier("gemma", "rival", support_status="no_support"),
        ]

        winner = network._select_winner(results, verifier_results=verifiers, evidence={})

        self.assertEqual(winner.compressed_answer, "TOOL-ANS")

    def test_metadata_records_the_rescued_candidate(self) -> None:
        """Trace should show which key survived via the majority carve-out."""
        network = _network()
        network._select_winner(
            _seven_bd_scenario_results(),
            verifier_results=_seven_bd_verifiers(),
            evidence={},
        )

        gate = next(
            gate
            for gate in network._last_winner_selection_trace.get("gate_trace") or []
            if gate.get("gate_name") == "evidence_support"
        )
        self.assertIn(
            "89706.00",
            gate.get("metadata", {}).get("trusted_tool_majority_rescued", []),
        )


if __name__ == "__main__":
    unittest.main()
