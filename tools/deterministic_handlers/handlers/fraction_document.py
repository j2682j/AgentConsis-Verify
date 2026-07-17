from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from tools.attachment_reader.specialized import FractionDocumentExtractor

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field


class FractionDocumentRouterHandler:
    """Extract ordered fractions from an image and solve its sample reductions."""

    name = "fraction_document"
    uses_specialized_attachment_parser = True
    handler_role = "fraction_document_reasoning"
    capability_description = (
        "Read a worksheet or document screenshot containing slash fractions and vertically "
        "typeset fraction exercises, then simplify the exercises in display order."
    )
    supported_attachment_types: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    supported_task_roles: set[str] = {"fraction_document_reasoning"}
    supported_answer_roles: set[str] = {"fraction_list", "list"}
    routing_terms = {"fraction", "fractions", "numerator", "denominator", "simplify"}
    input_schema = io_contract(
        name,
        [
            input_field("file_path", "str", True, "Image attachment path.", "attachment"),
            input_field("question", "str", True, "Original task question.", "question"),
        ],
        [
            *default_outputs(),
            output_field("literal_fractions", "list[str]", True, "Slash fractions in reading order."),
            output_field("sample_answers", "list[str]", True, "Reduced exercise answers."),
            output_field("ordered_answer", "list[str]", True, "Combined final output sequence."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self, extractor: FractionDocumentExtractor | None = None) -> None:
        self.extractor = extractor

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        file_path = str(inputs.get("file_path") or "")
        extension = Path(file_path).suffix.casefold()
        question = handler_input.question.casefold()
        task_match = "fraction" in question and extension in self.supported_attachment_types
        missing = [] if file_path and Path(file_path).is_file() else ["file_path"]
        return HandlerMatch(
            handler_name=self.name,
            handler_role=self.handler_role,
            matched=task_match and not missing,
            confidence=0.99 if task_match and not missing else 0.15,
            reason="fraction_image_attachment",
            missing_inputs=missing,
            required_inputs=["file_path", "question"],
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        adapted = handler_input.adapted_inputs()
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        return {
            "file_path": str(
                adapted.get("file_path")
                or attachment.get("file_path")
                or attachment.get("path")
                or ""
            ),
            "question": handler_input.question,
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        file_path = str(inputs.get("file_path") or "").strip()
        if not file_path or not Path(file_path).is_file():
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["file_path"],
                next_action_hint="Provide the original image attachment.",
            )
        try:
            extractor = self.extractor or FractionDocumentExtractor()
            artifact = extractor.extract(file_path)
        except Exception as exc:
            return HandlerResult.error_result(handler_name=self.name, error=str(exc))

        literal = [item.text for item in artifact.literal_fractions]
        sample_answers: list[str] = []
        validation_errors: list[str] = []
        for problem in artifact.sample_problems:
            if problem.denominator == 0:
                validation_errors.append(f"zero_denominator:{problem.order}")
                continue
            value = Fraction(problem.numerator, problem.denominator)
            sample_answers.append(f"{value.numerator}/{value.denominator}")
        if not literal:
            validation_errors.append("no_literal_slash_fractions")
        if not artifact.sample_problems:
            validation_errors.append("no_sample_fractions")
        if validation_errors:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=validation_errors,
                structured_result={"artifact": artifact.to_dict()},
                next_action_hint="Retry OCR on the fraction regions before answering.",
            )

        ordered = [*literal, *sample_answers]
        answer = ",".join(ordered)
        structured = {
            "task_type": "fraction_document_extraction_and_reduction",
            "literal_fractions": literal,
            "sample_answers": sample_answers,
            "ordered_answer": ordered,
            "artifact": artifact.to_dict(),
            "validation": {
                "valid": True,
                "literal_count": len(literal),
                "sample_count": len(sample_answers),
                "denominators_nonzero": True,
                "answers_reduced": True,
            },
            "calculation_trace": {
                "sample_inputs": [
                    f"{item.numerator}/{item.denominator}"
                    for item in artifact.sample_problems
                ],
                "sample_outputs": sample_answers,
            },
            "input_provenance": {
                "source": "specialized_attachment_input",
                "file_path": file_path,
                "parse_status": "success",
            },
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            structured_result=structured,
            confidence=1.0,
            output_type="final_answer",
            semantic_role="ordered_fraction_document_answer",
            supporting_inputs=[file_path, *literal],
        )


__all__ = ["FractionDocumentRouterHandler"]
