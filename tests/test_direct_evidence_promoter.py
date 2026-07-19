from tools.evidence.fact_extraction import DirectEvidencePromoter, EvidenceFact


def _fact(
    *,
    object_value: str,
    context: str,
    qualifiers: dict[str, str] | None = None,
    polarity: str = "positive",
) -> EvidenceFact:
    return EvidenceFact(
        fact_id="F1",
        subject="fish bag",
        relation="capacity",
        object=object_value,
        qualifiers=dict(qualifiers or {}),
        polarity=polarity,
        role="ANSWER_SUPPORT",
        evidence_spans=[context],
        context=context,
        source_id="D1",
        source_type="web",
        source_title="Paper",
        grounding_status="grounded",
    )


def test_promotes_grounded_measurement_from_fact_qualifier() -> None:
    context = "Therefore, the fish bag has a capacity of 0.1777 m3."
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="0.1777 m3",
        context=context,
        question="What was the volume in m^3 of the fish bag?",
        answer_requirement="volume in m^3 of the fish bag",
        answer_target="fish bag",
        source_id="D1",
        source_title="Paper",
        document_id="D1",
        goal_id="G1",
        semantic_facts=[
            _fact(
                object_value="Paper title",
                context=context,
                qualifiers={"reported_volume": "0.1777 m3"},
            )
        ],
    )

    assert [item.value for item in result.promoted_values] == ["0.1777 m3"]
    assert result.promoted_facts[0].qualifiers["answer_binding"] == "direct"
    assert result.promoted_facts[0].derivation_type == "answer_value_promotion"


def test_rejects_year_as_count_answer() -> None:
    context = "2005 Album: Corazon Libre."
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="2005",
        context=context,
        question="How many studio albums were released?",
        answer_requirement="how many studio albums",
        answer_target="studio albums",
        source_id="D2",
        source_title="Discography",
        document_id="D2",
        goal_id="G1",
        semantic_facts=[_fact(object_value="2005", context=context)],
    )

    assert not result.promoted_values
    assert any(item.failed_gate == "answer_type" for item in result.diagnostics)


def test_rejects_local_value_for_global_maximum() -> None:
    context = "At 00:26, two bird species are visible."
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="2",
        context=context,
        question="What is the highest number of bird species visible simultaneously?",
        answer_requirement="highest number of bird species visible simultaneously",
        answer_target="bird species visible simultaneously",
        source_id="V1",
        source_title="Video",
        document_id="V1",
        goal_id="G1",
        semantic_facts=[_fact(object_value="2", context=context)],
    )

    assert not result.promoted_values
    assert any(item.failed_gate == "scope_completion" for item in result.diagnostics)


def test_question_preserves_count_type_when_answer_requirement_is_truncated() -> None:
    context = (
        "Record Type: article Title: 10 Questions We Need Answered "
        "Source: Highest Number Of Bird Species On Camera Simultaneously"
    )
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span=context,
        context=context,
        question="What is the highest number of bird species on camera simultaneously?",
        answer_requirement="camera simultaneously",
        answer_target="camera simultaneously highest number",
        source_id="D4",
        source_title="Unrelated article",
        document_id="D4",
        goal_id="G1",
        semantic_facts=[],
    )

    assert not result.promoted_values
    assert any(item.failed_gate == "context_binding" for item in result.diagnostics)


def test_negative_fact_never_uses_positive_promotion_path() -> None:
    context = "The article does not mention plasmons."
    result = DirectEvidencePromoter().promote(
        model_role="ANSWER_SUPPORT",
        candidate_span="plasmons",
        context=context,
        question="Which article does not mention plasmons?",
        answer_requirement="article that does not mention plasmons",
        answer_target="article",
        source_id="D3",
        source_title="Article",
        document_id="D3",
        goal_id="G1",
        semantic_facts=[
            _fact(
                object_value="plasmons",
                context=context,
                polarity="negative",
            )
        ],
    )

    assert not result.promoted_values
    assert any(item.failed_gate == "positive_polarity" for item in result.diagnostics)
