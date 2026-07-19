from __future__ import annotations

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network


def summary(*, answer: str = "42", confidence: float = 1.0, valid: bool = True):
    run = EachAgentReply(
        agent_id="a1",
        model_name="fake",
        run_index=1,
        raw_reply="",
        reasoning="step 1. Derive the answer.",
        final_answer=answer,
        tool_context="",
        parse_completed=valid,
        eligible_for_winner=valid,
        schema_valid=valid,
        reasoning_steps=[(1, "Derive the answer.")],
    )
    return AgentReasoningSummary(
        agent_id="a1",
        model_name="fake",
        runs=[run],
        compressed_answer=answer if valid else "",
        compressed_reasoning=run.reasoning,
        confidence_score=confidence,
        active=valid,
        valid_run_count=int(valid),
        eligible_run_count=int(valid),
        winner_selection_eligible=valid,
    )


def verifier(*, floor: float, support_priority: int, status: str):
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id="a1",
        verifier_score=floor,
        metadata={
            "process_verification": {"critical_step_floor": floor},
            "evidence_support": {
                "priority": support_priority,
                "status": status,
                "verification_status": status,
            },
        },
    )


def test_early_stop_requires_evidence_support_even_with_high_versa():
    network = Network(
        "What is the answer?",
        [AgentConfig(agent_id="a1", model_name="fake")],
        enable_stage1_early_stop=True,
    )
    result = summary()
    network._evaluate_early_candidates = lambda *args, **kwargs: [
        verifier(floor=0.99, support_priority=1, status="unknown")
    ]

    winner, _, reason = network._stage1_early_stop_decision([result])

    assert winner is None
    assert reason == "confidence_1.0_evidence_unsupported"
    assert network._stage1_retry_reason([result], []) == ""


def test_early_stop_accepts_supported_candidate_above_versa_threshold():
    network = Network(
        "What is the answer?",
        [AgentConfig(agent_id="a1", model_name="fake")],
        enable_stage1_early_stop=True,
    )
    result = summary()
    network._evaluate_early_candidates = lambda *args, **kwargs: [
        verifier(
            floor=0.96,
            support_priority=4,
            status="search_evidence_supported",
        )
    ]

    winner, _, reason = network._stage1_early_stop_decision([result])

    assert winner is result
    assert reason == "confidence_1.0_positive_versa_reward"


def test_retry_policy_only_marks_invalid_or_contradicted_results():
    network = Network(
        "What is the answer?",
        [AgentConfig(agent_id="a1", model_name="fake")],
        enable_stage1_early_stop=True,
    )

    assert network._stage1_retry_reason([summary(valid=False)], []) == (
        "all_stage1_answers_invalid"
    )
    assert network._stage1_retry_reason(
        [summary()],
        [verifier(floor=0.99, support_priority=0, status="contradicted")],
    ) == "all_candidates_contradicted"
    assert network._stage1_retry_reason(
        [summary()],
        [verifier(floor=0.50, support_priority=1, status="unknown")],
    ) == ""
