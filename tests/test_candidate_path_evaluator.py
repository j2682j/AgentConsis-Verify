from __future__ import annotations

from core.candidate_path_evaluator import CandidatePathEvaluator
from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply
from core.stage2_runner import Stage2Runner
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.evidence_support_checker import EvidenceSupportChecker
from score.versa_prm_scorer import VersaPRMScoreResult, VersaPRMStepScore


class FakeVersaScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score_steps(self, *, question: str, reasoning_steps: list[tuple[int, str]]):
        self.calls += 1
        return VersaPRMScoreResult(
            scorer_name="fake",
            model_id="fake",
            base_model_id="fake",
            step_scores=[
                VersaPRMStepScore(index, text, 0.96)
                for index, text in reasoning_steps
            ],
        )


class CountingSupportChecker(EvidenceSupportChecker):
    def __init__(self) -> None:
        super().__init__()
        self.path_calls = 0

    def check_path(self, **kwargs):
        self.path_calls += 1
        return super().check_path(**kwargs)


def make_summary() -> AgentReasoningSummary:
    run = EachAgentReply(
        agent_id="a1",
        model_name="test",
        run_index=1,
        raw_reply="",
        reasoning="step 1. Use the deterministic result 42.",
        final_answer="42",
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
        reasoning_parse_quality="valid",
        reasoning_versa_eligible=True,
        reasoning_steps=[(1, "Use the deterministic result 42.")],
    )
    return AgentReasoningSummary(
        agent_id="a1",
        model_name="test",
        runs=[run],
        compressed_answer="42",
        compressed_reasoning=run.reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )


def trusted_evidence() -> dict:
    return {
        "answer_requirement": "the numerical result",
        "tool_usage": [
            {
                "tool_name": "deterministic_handler_router",
                "handler_name": "simple_math",
                "ok": True,
                "status": "ok",
                "output_type": "final_answer",
                "semantic_role": "final_answer",
                "evidence_valid": True,
                "output_text": "42",
                "handler_trust": {
                    "trusted": True,
                    "answer": "42",
                    "output_type": "final_answer",
                },
            }
        ],
    }


def test_candidate_path_is_checked_and_scored_once_per_revision() -> None:
    summary = make_summary()
    clusterer = AnswerCandidateClusterer()
    candidates = clusterer.cluster([summary])
    checker = CountingSupportChecker()
    scorer = FakeVersaScorer()
    runner = Stage2Runner(
        question="What is the result?",
        agents=[AgentConfig(agent_id="a1", model_name="test")],
        versa_scorer=scorer,
    )
    evaluator = CandidatePathEvaluator(
        question="What is the result?",
        clusterer=clusterer,
        evidence_support_checker=checker,
        stage2_runner=runner,
    )

    evidence = trusted_evidence()
    first = evaluator.evaluate_candidates(
        candidates=candidates,
        stage1_results=[summary],
        evidence=evidence,
        enable_versa=True,
        evidence_revision=0,
    )
    second = evaluator.evaluate_candidates(
        candidates=candidates,
        stage1_results=[summary],
        evidence=trusted_evidence(),
        enable_versa=True,
        evidence_revision=0,
    )

    assert checker.path_calls == 1
    assert scorer.calls == 1
    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert first.path_evaluations[0].evidence_support_status == "tool_final_supported"
    assert first.path_evaluations[0].versa_available is True
    assert "_fact_store" not in evidence
    assert "fact_store" not in evidence


def test_unreliable_reasoning_keeps_versa_unavailable() -> None:
    summary = make_summary()
    summary.runs[0].reasoning_parse_quality = "unreliable"
    summary.runs[0].reasoning_versa_eligible = False
    clusterer = AnswerCandidateClusterer()
    scorer = FakeVersaScorer()
    evaluator = CandidatePathEvaluator(
        question="What is the result?",
        clusterer=clusterer,
        evidence_support_checker=EvidenceSupportChecker(),
        stage2_runner=Stage2Runner(
            question="What is the result?",
            agents=[AgentConfig(agent_id="a1", model_name="test")],
            versa_scorer=scorer,
        ),
    )

    bundle = evaluator.evaluate_candidates(
        candidates=clusterer.cluster([summary]),
        stage1_results=[summary],
        evidence=trusted_evidence(),
        enable_versa=True,
    )

    path = bundle.path_evaluations[0]
    assert path.versa_available is False
    assert path.versa_status == "unavailable_unreliable_reasoning"
    assert path.critical_step_floor is None
    assert scorer.calls == 0
