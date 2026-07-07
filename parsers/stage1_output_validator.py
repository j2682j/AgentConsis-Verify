from __future__ import annotations

import re

from .stage1_output_schema import Stage1StructuredOutput, Stage1ValidationResult
from score.answer_validator import AnswerValidator


class Stage1OutputValidator:
    """
    Validate Stage1 structured output without assigning numeric penalties.
    """

    ANSWER_TYPES = {
        "number",
        "date",
        "person",
        "organization",
        "location",
        "title",
        "list",
        "short_text",
        "short_phrase",
        "boolean",
        "unknown",
    }

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def validate(self, output: Stage1StructuredOutput) -> Stage1ValidationResult:
        schema_errors: list[str] = []
        validity_labels: list[str] = []

        if output.confidence is not None and not 0.0 <= output.confidence <= 1.0:
            schema_errors.append("confidence_out_of_range")

        if output.answer_type not in self.ANSWER_TYPES:
            schema_errors.append("unknown_answer_type")

        invalid_evidence_ids = [
            evidence_id
            for evidence_id in output.used_evidence_ids
            if not re.fullmatch(r"E\d+", str(evidence_id or ""))
        ]
        if invalid_evidence_ids:
            schema_errors.append("invalid_evidence_id")

        malformed_steps = [
            step
            for step in output.reasoning_steps
            if not re.match(r"^step\s+\d+\.", str(step or "").strip(), re.IGNORECASE)
        ]
        if malformed_steps:
            schema_errors.append("malformed_reasoning_steps")

        if output.tool_request is not None:
            if not output.tool_request.tool_name:
                schema_errors.append("missing_tool_name")
            if output.final_answer:
                schema_errors.append("tool_request_with_final_answer")
            return Stage1ValidationResult(
                schema_valid=not schema_errors,
                eligible_for_winner=False,
                schema_errors=schema_errors,
                validity_labels=["tool_request_pending"],
            )

        final_answer = self.answer_validator.clean(output.final_answer)
        if not final_answer:
            validity_labels.append("empty_final_answer")
        elif self.answer_validator.is_tool_call_like(final_answer):
            validity_labels.append("tool_call_as_final_answer")
        elif self.answer_validator.is_refusal_like(final_answer):
            validity_labels.append("refusal_like_final_answer")
        elif self.answer_validator.is_too_verbose(final_answer):
            validity_labels.append("too_verbose_final_answer")
        elif self.answer_validator.is_uncertain(final_answer):
            validity_labels.append("uncertain_final_answer")
        elif not self.answer_validator.is_valid(final_answer):
            validity_labels.append("invalid_final_answer")

        schema_valid = not schema_errors
        eligible = schema_valid and not validity_labels
        return Stage1ValidationResult(
            schema_valid=schema_valid,
            eligible_for_winner=eligible,
            schema_errors=schema_errors,
            validity_labels=validity_labels,
        )


__all__ = ["Stage1OutputValidator"]
