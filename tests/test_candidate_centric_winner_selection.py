from __future__ import annotations

import math
import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network
from core.stage1_runner import Stage1Runner
from core.stage2_runner import Stage2Runner
from score.evidence_support_checker import EvidenceSupportChecker
from score.versa_prm_scorer import VersaPRMScoreResult, VersaPRMStepScore


def run(agent_id: str, run_index: int, answer: str, reasoning: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=run_index,
        raw_reply="",
        reasoning=reasoning,
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )


def summary(
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


class SequenceVersaScorer:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def score_steps(self, *, question: str, reasoning_steps: list[tuple[int, str]]):
        return VersaPRMScoreResult(
            scorer_name="fake",
            model_id="fake",
            base_model_id="fake",
            step_scores=[
                VersaPRMStepScore(
                    step_index=step_index,
                    step_text=step_text,
                    reward_probability=self.probabilities[position],
                )
                for position, (step_index, step_text) in enumerate(reasoning_steps)
            ],
        )


class ReviewAgent:
    def invoke_with_usage(self, messages: list[dict[str, str]]):
        return '{"type":"final_answer","answer":"Paris"}', 10, 5


class CandidateCentricWinnerSelectionTests(unittest.TestCase):
    def test_cross_agent_minority_answer_is_not_lost(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            summary(
                "a1",
                [
                    run("a1", 1, "A", "step 1. choose A"),
                    run("a1", 2, "A", "step 1. choose A again"),
                    run("a1", 3, "B", "step 1. evidence points to B"),
                ],
                answer="A",
                confidence=0.67,
            ),
            summary(
                "a2",
                [run("a2", 1, "B", "step 1. independently choose B")],
                answer="B",
                confidence=1.0,
            ),
        ]
        network = Network("Which answer is correct?", configs)

        winner = network._select_winner(results, evidence={})

        self.assertIsNotNone(winner)
        self.assertEqual(winner.compressed_answer, "B")
        trace = network._last_winner_selection_trace
        selected = next(
            candidate
            for candidate in trace["candidates"]
            if candidate["candidate_key"] == "b"
        )
        self.assertEqual(selected["supporting_agent_ids"], ["a1", "a2"])
        self.assertEqual(selected["supporting_run_count"], 2)

    def test_selected_search_evidence_supports_candidate(self) -> None:
        target = summary(
            "a1",
            [run("a1", 1, "Paris", "step 1. The evidence identifies Paris.")],
            answer="Paris",
            confidence=1.0,
        )
        evidence = {
            "tool_usage": [
                {
                    "tool_name": "search",
                    "ok": True,
                    "raw_result": {
                        "evidence_items": [
                            {
                                "evidence_id": "E1",
                                "title": "France",
                                "text": "Paris is the capital and largest city of France.",
                                "compatible_spans": ["Paris"],
                            }
                        ]
                    },
                }
            ]
        }

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, "The evidence identifies Paris.")],
            evidence=evidence,
        )

        self.assertEqual(support.status, "search_evidence_supported")
        self.assertEqual(support.priority, 3)
        self.assertEqual(support.step_results[0].status, "supported")

    def test_critical_step_floor_exposes_low_answer_step(self) -> None:
        target = summary(
            "a1",
            [
                run(
                    "a1",
                    1,
                    "Paris",
                    "step 1. France has a capital.\n"
                    "step 2. The result is Paris.\n"
                    "step 3. Return the requested short answer.",
                )
            ],
            answer="Paris",
            confidence=1.0,
        )
        runner = Stage2Runner(
            question="What is the capital of France?",
            agents=[AgentConfig(agent_id="a1", model_name="test-model")],
            versa_scorer=SequenceVersaScorer([0.99, 0.20, 0.99]),
        )

        result = runner.score_candidate(target)
        process = result.metadata["process_verification"]

        self.assertEqual(process["critical_step_indices"], [2, 3])
        self.assertAlmostEqual(process["critical_step_floor"], 0.20)
        self.assertAlmostEqual(
            process["critical_step_geometric_mean"],
            math.sqrt(0.20 * 0.99),
        )
        self.assertGreater(result.verifier_score, process["critical_step_floor"])

    def test_candidate_does_not_mix_support_and_versa_from_different_runs(self) -> None:
        configs = [
            AgentConfig(agent_id="supported", model_name="test-model"),
            AgentConfig(agent_id="high_prm", model_name="test-model"),
        ]
        results = [
            summary(
                "supported",
                [run("supported", 1, "Paris", "step 1. Evidence says Paris.")],
                answer="Paris",
                confidence=1.0,
            ),
            summary(
                "high_prm",
                [run("high_prm", 1, "Paris", "step 1. Guess Paris.")],
                answer="Paris",
                confidence=1.0,
            ),
        ]
        verifiers = [
            VerifierScoreByReasoning(
                verifier_id="versa_prm",
                target_agent_id="supported",
                verifier_score=0.40,
                metadata={
                    "candidate_key": "paris",
                    "target_run_index": 1,
                    "evidence_support": {
                        "status": "search_evidence_supported",
                        "priority": 3,
                    },
                    "process_verification": {
                        "critical_step_floor": 0.40,
                        "critical_step_geometric_mean": 0.40,
                        "average_probability": 0.40,
                    },
                },
            ),
            VerifierScoreByReasoning(
                verifier_id="versa_prm",
                target_agent_id="high_prm",
                verifier_score=0.99,
                metadata={
                    "candidate_key": "paris",
                    "target_run_index": 1,
                    "evidence_support": {"status": "no_support", "priority": 1},
                    "process_verification": {
                        "critical_step_floor": 0.99,
                        "critical_step_geometric_mean": 0.99,
                        "average_probability": 0.99,
                    },
                },
            ),
        ]
        network = Network("What is the capital?", configs)

        winner = network._select_winner(
            results,
            verifier_results=verifiers,
            evidence={},
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner.agent_id, "supported")
        selected = network._last_winner_selection_trace["candidates"][0]
        self.assertEqual(selected["critical_step_floor"], 0.40)
        self.assertNotEqual(selected["critical_step_floor"], 0.99)

    def test_candidate_path_scoring_keeps_run_identity(self) -> None:
        target = summary(
            "a1",
            [
                run("a1", 1, "Paris", "step 1. Paris."),
                run("a1", 2, "Lyon", "step 1. Lyon."),
            ],
            answer="Paris",
            confidence=0.33,
        )
        runner = Stage2Runner(
            question="Which city?",
            agents=[AgentConfig(agent_id="a1", model_name="test-model")],
            versa_scorer=SequenceVersaScorer([0.90]),
        )
        network = Network(
            "Which city?",
            [AgentConfig(agent_id="a1", model_name="test-model")],
        )

        scores = runner.run_candidate_paths(
            [target],
            candidate_key_builder=network.answer_candidate_clusterer.candidate_key,
        )

        self.assertEqual(len(scores), 2)
        self.assertEqual(
            [score.metadata["target_run_index"] for score in scores],
            [1, 2],
        )
        self.assertEqual(
            [score.metadata["candidate_key"] for score in scores],
            ["paris", "lyon"],
        )

    def test_exact_hierarchical_tie_requests_contrastive_review(self) -> None:
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        results = [
            summary(
                "a1",
                [run("a1", 1, "Paris", "step 1. Paris.")],
                answer="Paris",
                confidence=1.0,
            ),
            summary(
                "a2",
                [run("a2", 1, "Lyon", "step 1. Lyon.")],
                answer="Lyon",
                confidence=1.0,
            ),
        ]
        network = Network("Which city?", configs)
        candidates = network.answer_candidate_clusterer.cluster(results)

        selection = network.final_winner_selector.select(
            stage1_results=results,
            candidates=candidates,
            verifier_results=[],
            evidence={},
        )

        self.assertIsNone(selection.winner)
        self.assertEqual(selection.status, "review_required")

    def test_contrastive_review_can_only_return_a_candidate(self) -> None:
        token_usage: list[tuple[int, int]] = []
        runner = Stage1Runner(
            question="Which city is the capital of France?",
            agents=[AgentConfig(agent_id="a1", model_name="test-model")],
            get_agent=lambda config: ReviewAgent(),
            record_token_usage=lambda **usage: token_usage.append(
                (usage["prompt_tokens"], usage["completion_tokens"])
            ),
            stage1_runs_per_agent=1,
        )

        result = runner.review_final_candidates(
            candidate_answers=["Paris", "Lyon"],
            evidence_context="Paris is the capital of France.",
            preferred_agent_id="a1",
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(token_usage, [(10, 5)])


if __name__ == "__main__":
    unittest.main()
