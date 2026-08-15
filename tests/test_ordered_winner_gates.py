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
    def test_corpus_abbreviation_does_not_override_full_form_directive(self) -> None:
        configs = [
            AgentConfig(agent_id="short", model_name="test-model"),
            AgentConfig(agent_id="full", model_name="test-model"),
        ]
        results = [
            make_summary(
                "short",
                [make_run("short", 1, "St. Petersburg", answer_type="place")],
                answer="St. Petersburg",
                confidence=1.0,
            ),
            make_summary(
                "full",
                [make_run("full", 1, "Saint Petersburg", answer_type="place")],
                answer="Saint Petersburg",
                confidence=1.0,
            ),
        ]
        network = Network(
            "Give the city name without abbreviations.",
            configs,
        )

        winner = network._select_winner(
            results,
            verifier_results=[
                verifier(
                    "short",
                    "St. Petersburg",
                    support_status="no_support",
                    probability=0.80,
                ),
                verifier(
                    "full",
                    "Saint Petersburg",
                    support_status="no_support",
                    probability=0.95,
                ),
            ],
            evidence={
                "answer_requirement": "Give the city name without abbreviations.",
                "tool_usage": [
                    {
                        "raw_result": {
                            "retrieval": {
                                "rounds": [
                                    {
                                        "documents": [
                                            {"text": "The collection is in St. Petersburg."}
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                ],
            },
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "Saint Petersburg")

    def test_selected_output_uses_canonical_explicit_list_order(self) -> None:
        configs = [AgentConfig(agent_id="a1", model_name="test-model")]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "sugar, apples, flour", answer_type="list")],
                answer="sugar, apples, flour",
                confidence=1.0,
            )
        ]
        network = Network(
            "Return a comma-separated list in alphabetical order.",
            configs,
        )

        winner = network._select_winner(
            results,
            evidence={
                "answer_requirement": (
                    "Return a comma-separated list in alphabetical order."
                ),
                "answer_role": "list",
            },
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "apples, flour, sugar")
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
        # By name, not by position: this read `gate_trace[1]` and broke when a
        # gate was inserted ahead of the requirement gate, reporting the new
        # gate's survivors as though the requirement gate had passed both.
        requirement_gate = next(
            gate
            for gate in network._last_winner_selection_trace["gate_trace"]
            if gate["gate_name"] == "answer_requirement"
        )
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

    def test_single_unsupported_factual_candidate_falls_back(self) -> None:
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

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "Paris")
        trace = network._last_winner_selection_trace
        self.assertEqual(trace["selection_origin"], "fallback_best_candidate")
        self.assertEqual(
            trace["evidence_only_resolution"].get("fallback_from_status"),
            "unresolved_factual_without_support",
        )

    def test_closed_world_consensus_prefers_self_consistent_agent(self) -> None:
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

        # Two internally inconsistent agents (0.33) agreeing may not outvote a
        # fully self-consistent agent (1.0): correlated errors dominate GAIA
        # closed-book runs, so conviction outranks headcount.
        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "B")

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

    def test_equal_versa_results_fall_back_deterministically(self) -> None:
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

        # A fully tied pool must still produce a non-empty answer: the
        # fallback keeps the first-seen candidate instead of abstaining.
        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "A")
        trace = network._last_winner_selection_trace
        self.assertEqual(trace["selection_origin"], "fallback_best_candidate")
        self.assertEqual(
            trace["evidence_only_resolution"].get("fallback_from_status"),
            "unresolved_exact_tie",
        )

    def test_markdown_tail_does_not_split_same_candidate(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "egalitarian")],
                answer="egalitarian",
                confidence=1.0,
            ),
            make_summary(
                "a2",
                [make_run("a2", 1, "egalitarian**.")],
                answer="egalitarian**.",
                confidence=1.0,
            ),
        ]
        network = Network("Which word?", configs)

        candidates = network.answer_candidate_clusterer.cluster(results)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_key, "egalitarian")

    def test_year_requirement_canonicalizes_full_date_before_clustering(self) -> None:
        configs = [AgentConfig(agent_id="a1", model_name="test-model")]
        results = [
            make_summary(
                "a1",
                [make_run("a1", 1, "August 16, 2018", answer_type="date")],
                answer="August 16, 2018",
                confidence=1.0,
            )
        ]
        network = Network("What year was it released?", configs)

        candidates = network.answer_candidate_clusterer.cluster(
            results,
            answer_requirement="what year was it released",
            answer_role="year",
        )

        self.assertEqual(candidates[0].representative_answer, "2018")


if __name__ == "__main__":
    unittest.main()
