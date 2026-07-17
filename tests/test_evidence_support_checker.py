from __future__ import annotations

import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network
from core.stage2_runner import Stage2Runner
from score.evidence_support_checker import EvidenceSupportChecker
from score.versa_prm_scorer import VersaPRMScoreResult, VersaPRMStepScore


def make_run(
    agent_id: str,
    answer: str,
    reasoning: str,
    *,
    tool_results: list[dict] | None = None,
) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=1,
        raw_reply="",
        reasoning=reasoning,
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
        tool_results=list(tool_results or []),
    )


def make_summary(
    agent_id: str,
    answer: str,
    reasoning: str,
    *,
    confidence: float = 0.33,
    tool_results: list[dict] | None = None,
) -> AgentReasoningSummary:
    run = make_run(
        agent_id,
        answer,
        reasoning,
        tool_results=tool_results,
    )
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=[run],
        compressed_answer=answer,
        compressed_reasoning=reasoning,
        confidence_score=confidence,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )


def trusted_final_evidence(answer: str) -> dict:
    return {
        "tool_usage": [
            {
                "tool_name": "deterministic_handler_router",
                "handler_name": "simple_math",
                "ok": True,
                "status": "ok",
                "output_type": "final_answer",
                "semantic_role": "final_answer",
                "evidence_valid": True,
                "output_text": f"Answer: {answer}",
                "handler_trust": {
                    "trusted": True,
                    "answer": answer,
                    "output_type": "final_answer",
                    "semantic_role": "final_answer",
                },
                "raw_result": {"answer": answer},
            }
        ]
    }


class FakeVersaScorer:
    def score_steps(self, *, question: str, reasoning_steps: list[tuple[int, str]]):
        return VersaPRMScoreResult(
            scorer_name="fake",
            model_id="fake",
            base_model_id="fake",
            step_scores=[
                VersaPRMStepScore(
                    step_index=index,
                    step_text=text,
                    reward_probability=0.97,
                )
                for index, text in reasoning_steps
            ],
        )


def verifier(
    agent_id: str,
    *,
    reward: float,
    support_status: str,
    support_priority: int,
) -> VerifierScoreByReasoning:
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent_id,
        verifier_score=reward,
        metadata={
            "evidence_support": {
                "agent_id": agent_id,
                "status": support_status,
                "priority": support_priority,
            }
        },
    )


def case_trusted_final_answer_supports_agent_and_step() -> None:
    summary = make_summary(
        "a1",
        "42",
        "step 1. The deterministic result is 42.",
    )

    result = EvidenceSupportChecker().check_agent(
        target=summary,
        reasoning_steps=[(1, "The deterministic result is 42.")],
        evidence=trusted_final_evidence("42"),
    )

    assert result.status == "tool_final_supported"
    assert result.priority == 5
    assert result.step_results[0].status == "supported"
    assert result.step_results[0].matched_tool_values == ["42"]


def case_conflicting_trusted_final_marks_agent_contradicted() -> None:
    summary = make_summary(
        "a1",
        "41",
        "step 1. Therefore the result is 41.",
    )

    result = EvidenceSupportChecker().check_agent(
        target=summary,
        reasoning_steps=[(1, "Therefore the result is 41.")],
        evidence=trusted_final_evidence("42"),
    )

    assert result.status == "contradicted"
    assert result.priority == -1
    assert result.step_results[0].status == "contradicted"


def case_stage1_intermediate_value_supports_reasoning() -> None:
    summary = make_summary(
        "a1",
        "84",
        "step 1. The calculator returned 42. step 2. Doubling it gives 84.",
        tool_results=[
            {
                "tool_name": "python_calculator",
                "ok": True,
                "status": "success",
                "evidence_valid": True,
                "output_text": "42",
            }
        ],
    )

    result = EvidenceSupportChecker().check_agent(
        target=summary,
        reasoning_steps=[
            (1, "The calculator returned 42."),
            (2, "Doubling it gives 84."),
        ],
        evidence={},
    )

    assert result.status == "tool_intermediate_supported"
    assert result.step_results[0].status == "supported"
    assert result.step_results[1].status == "unsupported"


def case_failed_attachment_allows_model_only_result_at_low_priority() -> None:
    summary = make_summary(
        "a1",
        "Tokyo",
        "step 1. I infer the location is Tokyo.",
        tool_results=[
            {
                "tool_name": "attachment_reader",
                "ok": False,
                "status": "retryable_failure",
                "evidence_valid": False,
                "error": "image decode failed",
            }
        ],
    )

    result = EvidenceSupportChecker().check_agent(
        target=summary,
        reasoning_steps=[(1, "I infer the location is Tokyo.")],
        evidence={},
    )

    assert result.status == "tool_failed_model_only"
    assert result.priority == 0
    assert result.step_results[0].status == "tool_failed"


def case_stage2_combines_support_status_with_versa_probability() -> None:
    summary = make_summary(
        "a1",
        "42",
        "step 1. The deterministic result is 42.",
    )
    runner = Stage2Runner(
        question="What is the result?",
        agents=[AgentConfig(agent_id="a1", model_name="test-model")],
        versa_scorer=FakeVersaScorer(),
    )

    result = runner.score_candidate(
        summary,
        evidence=trusted_final_evidence("42"),
    )

    assert result.verifier_score == 0.97
    assert result.step_scores[0]["support_status"] == "supported"
    assert result.step_scores[0]["reward_probability"] == 0.97
    assert result.metadata["evidence_support"]["status"] == "tool_final_supported"


def case_supported_low_consistency_candidate_beats_unsupported_candidate() -> None:
    configs = [
        AgentConfig(agent_id="supported", model_name="test-model"),
        AgentConfig(agent_id="unsupported", model_name="test-model"),
    ]
    supported = make_summary(
        "supported",
        "42",
        "step 1. The tool returned 42.",
        confidence=0.33,
    )
    unsupported = make_summary(
        "unsupported",
        "41",
        "step 1. I think the answer is 41.",
        confidence=1.0,
    )
    configs[0].total_score = 1.03
    configs[1].total_score = 1.99
    network = Network("What is the result?", configs)

    winner = network._select_winner(
        [supported, unsupported],
        verifier_results=[
            verifier(
                "supported",
                reward=0.70,
                support_status="tool_final_supported",
                support_priority=5,
            ),
            verifier(
                "unsupported",
                reward=0.99,
                support_status="no_support",
                support_priority=1,
            ),
        ],
    )

    assert winner is not None
    assert winner.agent_id == "supported"


class EvidenceSupportCheckerTests(unittest.TestCase):
    def test_trusted_final_answer_supports_agent_and_step(self) -> None:
        case_trusted_final_answer_supports_agent_and_step()

    def test_conflicting_trusted_final_marks_agent_contradicted(self) -> None:
        case_conflicting_trusted_final_marks_agent_contradicted()

    def test_stage1_intermediate_value_supports_reasoning(self) -> None:
        case_stage1_intermediate_value_supports_reasoning()

    def test_failed_attachment_allows_model_only_result_at_low_priority(self) -> None:
        case_failed_attachment_allows_model_only_result_at_low_priority()

    def test_stage2_combines_support_status_with_versa_probability(self) -> None:
        case_stage2_combines_support_status_with_versa_probability()

    def test_supported_low_consistency_candidate_beats_unsupported_candidate(self) -> None:
        case_supported_low_consistency_candidate_beats_unsupported_candidate()


if __name__ == "__main__":
    unittest.main()
