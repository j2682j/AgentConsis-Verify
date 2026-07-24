from tools.validation import ToolFinalityValidator


def test_legacy_final_without_metadata_keeps_existing_behavior() -> None:
    result = ToolFinalityValidator().validate(
        declared_output_type="final_answer",
        result_ok=True,
        answer="42",
    )

    assert result.final
    assert result.legacy_accepted
    assert result.effective_output_type == "final_answer"


def test_explicit_complete_contract_is_final() -> None:
    result = ToolFinalityValidator().validate(
        declared_output_type="final_answer",
        result_ok=True,
        answer="42",
        finality_payload={
            "operation_status": "complete",
            "scope_status": "complete",
            "required_constraints": ["column:total"],
            "satisfied_constraints": ["column:total"],
            "provenance_ids": ["row:1"],
        },
    )

    assert result.final
    assert not result.legacy_accepted


def test_incomplete_scope_or_constraints_downgrades_to_intermediate() -> None:
    result = ToolFinalityValidator().validate(
        declared_output_type="final_answer",
        result_ok=True,
        answer="18028",
        finality_payload={
            "operation_status": "complete",
            "scope_status": "incomplete",
            "required_constraints": ["column:burgers", "column:fries"],
            "satisfied_constraints": ["column:burgers"],
            "provenance_ids": ["row:1"],
        },
    )

    assert result.status == "intermediate"
    assert result.effective_output_type == "intermediate_value"
    assert "column:fries" in result.missing_constraints


def test_explicit_contract_requires_provenance() -> None:
    result = ToolFinalityValidator().validate(
        declared_output_type="final_answer",
        result_ok=True,
        answer="42",
        finality_payload={
            "operation_status": "complete",
            "scope_status": "complete",
        },
    )

    assert result.status == "invalid"
    assert result.effective_output_type == "invalid"
    assert "finality_provenance_invalid" in result.reasons
