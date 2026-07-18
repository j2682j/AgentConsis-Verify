from __future__ import annotations

import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network


def make_run(
    agent_id: str,
    run_index: int,
    answer: str,
    *,
    answer_type: str = "",
) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=run_index,
        raw_reply="",
        reasoning=f"step 1. Conclude {answer}.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        structured_output={"answer_type": answer_type} if answer_type else {},
        schema_valid=True,
        eligible_for_winner=True,
    )


def make_summary(
    agent_id: str,
    runs: list[EachAgentReply],
    *,
    answer: str,
    confidence: float,
) -> AgentReasoningSummary:
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=runs,
        compressed_answer=answer,
        compressed_reasoning=runs[0].reasoning,
        confidence_score=confidence,
        active=True,
        valid_run_count=len(runs),
        eligible_run_count=len(runs),
    )


def verifier(
    agent_id: str,
    answer: str,
    *,
    support_status: str,
    probability: float,
    run_index: int = 1,
) -> VerifierScoreByReasoning:
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent_id,
        verifier_score=probability,
        metadata={
            "candidate_key": answer.lower(),
            "target_run_index": run_index,
            "evidence_support": {"status": support_status},
            "process_verification": {
                "critical_step_floor": probability,
                "critical_step_geometric_mean": probability,
                "average_probability": probability,
            },
        },
    )


class OrderedWinnerGateTests(unittest.TestCase):
    def test_requirement_gate_rejects_clear_answer_shape_mismatch(self) -> None:
        configs = [
            AgentConfig(agent_id="text", model_name="test-model"),
            AgentConfig(agent_id="number", model_name="test-model"),
        ]
        results = [
            make_summary(
                "text",
                [make_run("text", 1, "Fred", answer_type="person")],
                answer="Fred",
                confidence=1.0,
            ),
            make_summary(
                "number",
                [make_run("number", 1, "3", answer_type="number")],
                answer="3",
                confidence=1.0,
            ),
        ]
        network = Network("How many studio albums were released?", configs)

        winner = network._select_winner(
            results,
            evidence={
                "answer_requirement": "how many studio albums",
                "answer_role": "number",
            },
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "3")
        requirement_gate = network._last_winner_selection_trace["gate_trace"][1]
        self.assertEqual(requirement_gate["survivors"], ["3"])

    def test_contradicted_candidate_cannot_be_rescued_by_high_versa(self) -> None:
        configs = [
            AgentConfig(agent_id="wrong", model_name="test-model"),
            AgentConfig(agent_id="right", model_name="test-model"),
        ]
        results = [
            make_summary(
                "wrong",
                [make_run("wrong", 1, "Lyon")],
                answer="Lyon",
                confidence=1.0,
            ),
            make_summary(
                "right",
                [make_run("right", 1, "Paris")],
                answer="Paris",
                confidence=1.0,
            ),
        ]
        network = Network("What is the capital of France?", configs)

        winner = network._select_winner(
            results,
            verifier_results=[
                verifier("wrong", "Lyon", support_status="contradicted", probability=0.99),
                verifier(
                    "right",
                    "Paris",
                    support_status="search_evidence_supported",
                    probability=0.60,
                ),
            ],
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "Paris")

    def test_direct_evidence_minority_beats_unsupported_majority(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
            AgentConfig(agent_id="a3", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "2")],
                answer="2",
                confidence=1.0,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "2")],
                answer="2",
                confidence=1.0,
            ),
            make_summary(
                "a3",
                [make_run("a3", 1, "3")],
                answer="3",
                confidence=1.0,
            ),
        ]
        network = Network("How many?", configs)

        winner = network._select_winner(
            results,
            verifier_results=[
                verifier("a1", "2", support_status="no_support", probability=0.99),
                verifier("a2", "2", support_status="no_support", probability=0.99),
                verifier(
                    "a3",
                    "3",
                    support_status="search_evidence_supported",
                    probability=0.70,
                ),
            ],
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "3")

    def test_single_unsupported_factual_candidate_is_unresolved(self) -> None:
        configs = [AgentConfig(agent_id="a1", model_name="test-model")]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "Paris")],
                answer="Paris",
                confidence=1.0,
            )
        ]
        network = Network("What is the capital?", configs)

        winner = network._select_winner(
            results,
            evidence={"routing": {"primary_route": "factual_search"}},
        )

        self.assertIsNone(winner)
        self.assertEqual(
            network._last_winner_selection_trace["status"],
            "unresolved_factual_without_support",
        )

    def test_closed_world_consensus_uses_distinct_agents_before_run_count(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
            AgentConfig(agent_id="a3", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "A")],
                answer="A",
                confidence=0.33,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "A")],
                answer="A",
                confidence=0.33,
            ),
            make_summary(
                "a3",
                [
                    make_run("a3", 1, "B"),
                    make_run("a3", 2, "B"),
                    make_run("a3", 3, "B"),
                ],
                answer="B",
                confidence=1.0,
            ),
        ]
        network = Network("Choose A or B from the puzzle.", configs)

        winner = network._select_winner(results)

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "A")

    def test_versa_only_compares_candidates_after_equal_earlier_gates(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "A")],
                answer="A",
                confidence=1.0,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "B")],
                answer="B",
                confidence=1.0,
            ),
        ]
        network = Network("Choose A or B.", configs)

        winner = network._select_winner(
            results,
            verifier_results=[
                verifier("a1", "A", support_status="no_support", probability=0.80),
                verifier("a2", "B", support_status="no_support", probability=0.90),
            ],
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "B")
        self.assertEqual(
            network._last_winner_selection_trace["gate_trace"][-1]["gate_name"],
            "versa_verification",
        )

    def test_equal_versa_results_remain_unresolved(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "A")],
                answer="A",
                confidence=1.0,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "B")],
                answer="B",
                confidence=1.0,
            ),
        ]
        network = Network("Choose A or B.", configs)

        winner = network._select_winner(
            results,
            verifier_results=[
                verifier("a1", "A", support_status="no_support", probability=0.90),
                verifier("a2", "B", support_status="no_support", probability=0.90),
            ],
        )

        self.assertIsNone(winner)
        self.assertEqual(
            network._last_winner_selection_trace["status"],
            "unresolved_exact_tie",
        )


if __name__ == "__main__":
    unittest.main()
