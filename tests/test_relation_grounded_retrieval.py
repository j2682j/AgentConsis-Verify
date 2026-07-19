from __future__ import annotations

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidatePathEvaluation,
    CandidatePathIdentity,
    CandidateRun,
    EachAgentReply,
)
from score.candidate_fact_verifier import CandidateFactVerifier
from score.evidence_answer_resolver import EvidenceAnswerResolver
from score.final_winner_selector import FinalWinnerSelector
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore
from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)
from tools.search_result_builder.next_hop_query import (
    GoalCompletionEvaluator,
    NextHopQueryComposer,
    RelationGoalResolver,
)
from tools.search_result_builder.query import RelationPlan, RelationPlanValidator
from tools.search_result_builder.query.question_role_extractor import QuestionRole


def _q19_plan(*, complete: bool = False) -> RelationPlan:
    first_state = "resolved"
    second_state = "resolved" if complete else "active"
    return RelationPlan.from_dict(
        {
            "goals": [
                {
                    "goal_id": "G1",
                    "subject": "November 2016 Featured Articles",
                    "relation": "dinosaur article",
                    "target": "article title",
                    "state": first_state,
                    "resolved_values": ["Giganotosaurus"],
                },
                {
                    "goal_id": "G2",
                    "subject": "",
                    "relation": "nominated by",
                    "target": "person",
                    "state": second_state,
                    "resolved_values": ["FunkMonk"] if complete else [],
                },
            ],
            "active_goal_id": "" if complete else "G2",
        }
    )


def _fact(value: str, relation: str, *, fact_id: str) -> EvidenceFact:
    return EvidenceFact(
        fact_id=fact_id,
        subject="Giganotosaurus",
        relation=relation,
        object=value,
        qualifiers={"answer_binding": "direct"},
        role="ANSWER_SUPPORT",
        goal_id="G2",
        evidence_spans=[f"Giganotosaurus was {relation} {value}."],
        context=f"Giganotosaurus was {relation} {value}.",
        source_id="fac-archive",
        source_type="web",
        source_title="Featured article candidates archive",
        grounding_status="grounded",
    )


def _contract(value: str, relation: str, *, fact_id: str) -> dict[str, str]:
    return {
        "goal_id": "G2",
        "answer_span": value,
        "object": value,
        "subject": "Giganotosaurus",
        "relation": relation,
        "fact_id": fact_id,
        "grounding_status": "grounded",
        "document_id": "fac-archive",
    }


def _unsupported_stage1_candidate() -> tuple[
    list[AgentReasoningSummary],
    list[AnswerCandidate],
    list[CandidatePathEvaluation],
]:
    reply = EachAgentReply(
        agent_id="qwen",
        model_name="qwen3:4b",
        run_index=1,
        raw_reply="",
        reasoning="step 1. The archive says promoted by Ian Rose.",
        final_answer="Ian Rose",
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )
    summary = AgentReasoningSummary(
        agent_id="qwen",
        model_name="qwen3:4b",
        runs=[reply],
        compressed_answer="Ian Rose",
        compressed_reasoning=reply.reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )
    member = CandidateRun(
        agent_id="qwen",
        model_name="qwen3:4b",
        run_index=1,
        answer="Ian Rose",
        normalized_answer="ian rose",
        reasoning=reply.reasoning,
    )
    candidate = AnswerCandidate(
        candidate_key="ian rose",
        representative_answer="Ian Rose",
        members=[member],
    )
    path = CandidatePathEvaluation(
        identity=CandidatePathIdentity("ian rose", "qwen", 1),
        answer="Ian Rose",
        valid=True,
        eligible_for_winner=True,
        schema_valid=True,
        parse_completed=True,
        reasoning=reply.reasoning,
        evidence_support_status="no_support",
        evidence_support_level="unsupported",
        agent_answer_frequency=1,
        eligible_run_count=1,
        agent_confidence=1.0,
    )
    return [summary], [candidate], [path]


def test_relation_plan_validator_repairs_answer_placeholder_direction() -> None:
    raw = RelationPlan.from_specs(
        [
            {
                "subject": "Who",
                "relation": "nominated",
                "target": "the Featured Article about a dinosaur",
            }
        ]
    )
    result = RelationPlanValidator().validate(
        raw,
        question_role=QuestionRole(answer_role="person"),
    )

    assert result.valid
    assert result.plan.goals[0].subject == "the Featured Article about a dinosaur"
    assert result.plan.goals[0].target == "person"
    assert "G1:swap_answer_placeholder_subject_with_target" in result.repairs


def test_fact_goal_binding_keeps_nomination_and_promotion_distinct() -> None:
    goal = _q19_plan().active_goal
    assert goal is not None
    validator = FactGoalBindingValidator()

    promoted = validator.validate(
        fact=_fact("Ian Rose", "promoted by", fact_id="F-promoted"),
        goal=goal,
        effective_subjects=["Giganotosaurus"],
        answer_role="person",
    )
    nominated = validator.validate(
        fact=_fact("FunkMonk", "nominated by", fact_id="F-nominated"),
        goal=goal,
        effective_subjects=["Giganotosaurus"],
        answer_role="person",
    )

    assert promoted.status == "relation_mismatch"
    assert nominated.status == "bound"


def test_wrong_relation_cannot_resolve_goal_or_stop_retrieval() -> None:
    plan = _q19_plan()
    wrong = _contract("Ian Rose", "promoted by", fact_id="F-promoted")
    resolution = RelationGoalResolver().resolve_direct(plan, [wrong])
    completion = GoalCompletionEvaluator().evaluate(
        relation_plan=resolution.plan,
        documents=[{"direct_contracts": [wrong]}],
        answer_gate_sufficient=True,
    )

    assert resolution.resolved_goal_ids == []
    assert resolution.rejected_contracts[0]["status"] == "relation_mismatch"
    assert not completion.sufficient
    assert completion.unresolved_goal_ids == ["G2"]


def test_relation_next_hop_query_preserves_subject_and_relation() -> None:
    requests = NextHopQueryComposer().build_relation_requests(
        relation_plan=_q19_plan(),
        constraints=["Featured article candidate archive", "answer_support:person"],
    )

    assert len(requests) == 1
    query = requests[0].request.query.casefold()
    assert "giganotosaurus" in query
    assert "nominated by" in query
    assert "answer_support" not in query
    assert " person" not in query


def test_candidate_fact_verifier_rejects_wrong_relation_candidate() -> None:
    store = TaskFactStore()
    store.extend(
        [
            _fact("Ian Rose", "promoted by", fact_id="F-promoted"),
            _fact("FunkMonk", "nominated by", fact_id="F-nominated"),
        ]
    )
    verifier = CandidateFactVerifier()

    wrong = verifier.verify(
        candidate_answer="Ian Rose",
        fact_store=store,
        required_relation="nominated by",
        required_relation_goal_id="G2",
        answer_role="person",
    )
    correct = verifier.verify(
        candidate_answer="FunkMonk",
        fact_store=store,
        required_relation="nominated by",
        required_relation_goal_id="G2",
        answer_role="person",
    )

    assert wrong.status == "unknown"
    assert wrong.reason == "candidate_only_matches_wrong_relation"
    assert wrong.relation_mismatch_fact_ids == ["F-promoted"]
    assert correct.status == "supported"
    assert correct.supporting_fact_ids == ["F-nominated"]


def test_evidence_only_resolution_selects_unique_relation_bound_fact() -> None:
    store = TaskFactStore()
    store.extend(
        [
            _fact("Ian Rose", "promoted by", fact_id="F-promoted"),
            _fact("FunkMonk", "nominated by", fact_id="F-nominated"),
        ]
    )
    evidence = {
        "relation_plan": _q19_plan(complete=True).to_dict(),
        "required_relation": "nominated by",
        "required_relation_goal_id": "G2",
        "answer_role": "person",
        "fact_store": store.to_dict(),
        "routing": {"use_search": True},
    }

    resolution = EvidenceAnswerResolver().resolve(evidence)
    stage1, candidates, paths = _unsupported_stage1_candidate()
    selection = FinalWinnerSelector().select(
        stage1_results=stage1,
        candidates=candidates,
        path_evaluations=paths,
        evidence=evidence,
    )

    assert resolution.resolved
    assert resolution.answer == "FunkMonk"
    assert selection.winner is None
    assert selection.resolved_answer == "FunkMonk"
    assert selection.selection_origin == "evidence_only_resolution"
    assert selection.to_dict()["selected_answer"] == "FunkMonk"


def test_evidence_only_resolution_refuses_conflicting_relation_values() -> None:
    store = TaskFactStore()
    store.extend(
        [
            _fact("FunkMonk", "nominated by", fact_id="F-one"),
            _fact("AnotherEditor", "nominated by", fact_id="F-two"),
        ]
    )
    result = EvidenceAnswerResolver().resolve(
        {
            "relation_plan": _q19_plan(complete=True).to_dict(),
            "required_relation": "nominated by",
            "required_relation_goal_id": "G2",
            "answer_role": "person",
            "fact_store": store.to_dict(),
        }
    )

    assert result.status == "conflict"
    assert set(result.conflicting_values) == {"FunkMonk", "AnotherEditor"}
