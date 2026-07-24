from tools.deterministic_handlers import HandlerInput, HandlerTrustGate
from tools.deterministic_handlers.handlers.table_aggregation import (
    TableAggregationRouterHandler,
)
from tools.evidence.fact_extraction import DirectEvidencePromoter, EvidenceFact


def _table_result(question: str):
    handler = TableAggregationRouterHandler()
    handler_input = HandlerInput(
        question=question,
        attachment_result="Burgers,Fries,Soda\n10,3,2\n20,4,5",
    )
    inputs = handler.build_input(handler_input)
    return handler.run(inputs)


def test_single_column_table_aggregation_remains_final() -> None:
    result = _table_result("What is the sum of Burgers?")
    trust = HandlerTrustGate().validate(result, question="What is the sum of Burgers?")

    assert trust.trusted
    assert trust.effective_output_type == "final_answer"


def test_partial_multi_column_table_aggregation_is_intermediate() -> None:
    question = "What is the sum of Burgers and Fries?"
    result = _table_result(question)
    trust = HandlerTrustGate().validate(result, question=question)

    assert not trust.trusted
    assert trust.usable_as_intermediate
    assert trust.effective_output_type == "intermediate_value"
    assert "finality_downgraded_to_intermediate" in trust.reasons


def test_local_maximum_is_not_promoted_as_global_answer() -> None:
    context = "The observed frame has a count of 2, which is the highest in this frame."
    origin = EvidenceFact(
        fact_id="V1",
        subject="observed frame",
        relation="has count",
        object="2",
        qualifiers={},
        role="ANSWER_SUPPORT",
        evidence_spans=[context],
        context=context,
        source_id="V1",
        source_type="video",
        grounding_status="grounded",
    )
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="2",
        context=context,
        question="What is the highest number visible simultaneously in the video?",
        answer_requirement="highest number visible simultaneously in the video",
        answer_target="number visible simultaneously",
        source_id="V1",
        source_title="Video",
        document_id="V1",
        goal_id="G1",
        semantic_facts=[origin],
    )

    promoted = next(fact for fact in result.promoted_facts if fact.parent_fact_ids)
    assert promoted.role == "BRIDGE"
    assert promoted.qualifiers["answer_binding"] == "bridge"
    assert promoted.derived_contract["verification_status"] == "unverified"
