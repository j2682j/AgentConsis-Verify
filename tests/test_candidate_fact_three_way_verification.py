from core.config import AgentReasoningSummary, EachAgentReply
from score.candidate_fact_verifier import CandidateFactVerifier
from score.evidence_support_checker import EvidenceSupportChecker
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore


def _fact(*, value: str, polarity: str = "positive") -> EvidenceFact:
    return EvidenceFact(
        fact_id=f"F-{polarity}-{value}",
        subject="requested value",
        relation="has answer",
        object=value,
        qualifiers={
            "answer_binding": "direct",
            **(
                {"negation_type": "explicit_negative"}
                if polarity == "negative"
                else {}
            ),
        },
        polarity=polarity,
        role="ANSWER_SUPPORT",
        evidence_spans=[f"The requested value is {value}."],
        context=f"The requested value is {value}.",
        source_id="D1",
        source_type="search",
        grounding_status="grounded",
    )


def _summary(answer: str) -> AgentReasoningSummary:
    run = EachAgentReply(
        agent_id="a1",
        model_name="test",
        run_index=1,
        raw_reply="",
        reasoning="step 1. Return the grounded value.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )
    return AgentReasoningSummary(
        agent_id="a1",
        model_name="test",
        runs=[run],
        compressed_answer=answer,
        compressed_reasoning=run.reasoning,
        confidence_score=1.0,
        active=True,
    )


def test_no_matching_fact_is_unknown_not_contradicted() -> None:
    store = TaskFactStore()
    store.add(_fact(value="42"))

    result = CandidateFactVerifier().verify(
        candidate_answer="41",
        fact_store=store,
        answer_requirement="requested value",
    )

    assert result.status == "unknown"
    assert result.contradicting_fact_ids == []
    assert result.reason == "no_answer_bound_fact_matches_candidate"


def test_negative_answer_fact_explicitly_contradicts_candidate() -> None:
    store = TaskFactStore()
    store.add(_fact(value="41", polarity="negative"))

    result = CandidateFactVerifier().verify(
        candidate_answer="41",
        fact_store=store,
        answer_requirement="requested value",
    )

    assert result.status == "contradicted"
    assert result.contradicting_fact_ids == ["F-negative-41"]


def test_final_answer_support_is_independent_from_step_wording() -> None:
    store = TaskFactStore()
    store.add(_fact(value="42"))
    evidence = {"_fact_store": store, "answer_requirement": "requested value"}

    result = EvidenceSupportChecker().check_agent(
        target=_summary("42"),
        reasoning_steps=[(1, "Return the grounded value.")],
        evidence=evidence,
        question="What is the requested value?",
    )

    assert result.verification_status == "supported"
    assert result.supporting_fact_ids == ["F-positive-42"]
    assert result.step_results[0].status == "unsupported"
