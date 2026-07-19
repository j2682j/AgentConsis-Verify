from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook

from core.config import AgentReasoningSummary, EachAgentReply
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.answer_requirement_contract import TaskAnswerRequirementContract
from score.answer_requirement_gate import AnswerRequirementGate
from score.candidate_fact_verifier import CandidateFactVerifier
from score.evidence_support_checker import EvidenceSupportChecker
from tools.deterministic_handlers import (
    DeterministicHandlerRouter,
    HandlerInput,
    HandlerTrustGate,
)
from tools.deterministic_handlers.handlers.logic_equivalence import (
    LogicEquivalenceRouterHandler,
)
from tools.evidence.fact_extraction import (
    DirectEvidencePromoter,
    EvidenceFact,
    QuestionRuleFactExtractor,
    TaskFactStore,
)


def _summary(answer: str) -> AgentReasoningSummary:
    run = EachAgentReply(
        agent_id="agent",
        model_name="test",
        run_index=1,
        raw_reply="",
        reasoning="step 1. Apply the stated evidence.",
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )
    return AgentReasoningSummary(
        agent_id="agent",
        model_name="test",
        runs=[run],
        compressed_answer=answer,
        compressed_reasoning=run.reasoning,
        confidence_score=1.0,
        active=True,
    )


def test_task_contract_replaces_placeholder_with_question() -> None:
    contract = TaskAnswerRequirementContract.build(
        question="Who did not give a gift?",
        answer_requirement="specific information required",
        answer_role="person",
    )
    assert contract.resolved
    assert contract.requirement_text == "Who did not give a gift?"
    assert contract.source == "question_fallback"


def test_task_contract_uses_answer_bearing_question_clause() -> None:
    contract = TaskAnswerRequirementContract.build(
        question=(
            "The color indicates who owns each plot. "
            "Can Earl visit every owned plot and return to the start? "
            "Backtracking means revisiting a plot."
        )
    )

    assert contract.answer_role == "boolean"


def test_person_sentence_clusters_with_short_person_answer() -> None:
    clusterer = AnswerCandidateClusterer()
    summaries = [_summary("Fred"), _summary("Fred did not give a gift.")]
    summaries[1].agent_id = "agent-2"
    summaries[1].runs[0].agent_id = "agent-2"
    candidates = clusterer.cluster(
        summaries,
        answer_requirement="the person who did not give a gift",
        answer_role="person",
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_key == "fred"
    assert candidates[0].supporting_run_count == 2


def test_measurement_candidate_inherits_question_unit() -> None:
    store = TaskFactStore()
    store.add(
        EvidenceFact(
            fact_id="capacity",
            subject="fish bag",
            relation="has_capacity",
            object="0.1777 m3",
            qualifiers={"answer_binding": "direct"},
            role="ANSWER_SUPPORT",
            evidence_spans=["The bag has a capacity of 0.1777 m3."],
            grounding_status="grounded",
            extraction_method="explicit_local_relation",
        )
    )
    result = CandidateFactVerifier().verify(
        candidate_answer="0.1777",
        fact_store=store,
        answer_requirement="What is the volume in m^3?",
    )
    assert result.status == "supported"

    canonical, _ = AnswerRequirementGate().canonicalize(
        "0.1777",
        answer_requirement="What is the volume in m^3?",
        answer_role="measurement",
    )
    assert canonical == "0.1777"


def test_promoter_requires_relation_and_accepts_explicit_measurement() -> None:
    promoter = DirectEvidencePromoter()
    accepted = promoter.promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="0.1777 m3",
        context="Therefore, the bag has a capacity of 0.1777 m3.",
        question="What was the volume in m^3 of the fish bag?",
        answer_requirement="What was the volume in m^3 of the fish bag?",
        answer_target="fish bag",
        source_id="paper",
        source_title="Paper",
        document_id="p1",
        goal_id="G1",
        semantic_facts=[],
    )
    assert [fact.object for fact in accepted.promoted_facts] == [
        "0.1777 m3",
        "0.1777 m3",
    ]
    rejected = promoter.promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="Rodrigues",
        context="The article also mentions the extinct Rodrigues solitaire.",
        question="Who nominated the article?",
        answer_requirement="Who nominated the article?",
        answer_target="article nominator",
        source_id="page",
        source_title="Page",
        document_id="p2",
        goal_id="G1",
        semantic_facts=[],
    )
    assert not rejected.promoted_facts
    assert rejected.diagnostics[-1].reason == "direct_answer_requires_grounded_origin_fact"


def test_logic_handler_does_not_match_translation_quotes() -> None:
    question = 'Please translate "I like apples" to Tizin; "Pa" is nominative and "Mato" is accusative.'
    match = LogicEquivalenceRouterHandler().match_input(HandlerInput(question=question))
    assert not match.matched
    assert match.reason == "logic_operation_not_explicit"


def test_question_rule_composition_supports_translation_candidate() -> None:
    question = (
        "In Tizin, sentences use the Verb first, followed by the direct object, followed by the subject. "
        'The word that indicates oneself is "Pa" is the nominative form, "Mato" is the accusative form, '
        'and "Sing" is the genitive form. The root verb that indicates an intense like is "Maktay". '
        "The thing doing the liking is actually the object rather than the subject. "
        'The word for apples is "Apple" is the nominative form, "Zapple" is the accusative form, '
        'and "Izapple" is the genitive form. Please translate "I like apples" to Tizin.'
    )
    facts = QuestionRuleFactExtractor().extract(question=question)
    assert any(fact.object == "Maktay Mato Apple" for fact in facts)
    summary = _summary("Maktay Mato Apple")
    support = EvidenceSupportChecker().check_agent(
        target=summary,
        reasoning_steps=[(1, summary.compressed_reasoning)],
        evidence={},
        question=question,
    )
    assert support.status == "derived_evidence_supported"


def test_color_grid_hamiltonian_handler_returns_failure_witness() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "grid.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        for coordinate in ("A1", "B1", "C1"):
            sheet[coordinate].fill = __import__("openpyxl").styles.PatternFill(
                fill_type="solid",
                fgColor="FF00FF00",
            )
        workbook.save(path)
        question = (
            "Green cells are owned plots. Can Earl walk through every green plot and "
            "return to his starting plot without repeating a plot?"
        )
        result = DeterministicHandlerRouter().run(
            question=question,
            attachment={"file_path": str(path), "extension": ".xlsx"},
            handler_name="color_grid_hamiltonian",
        )
        assert result.answer == "no"
        assert result.structured_result["failure_witness"]


def test_short_no_does_not_match_inside_nominative() -> None:
    checker = EvidenceSupportChecker()
    assert not checker._value_mentioned("no", "Use the nominative form.")
