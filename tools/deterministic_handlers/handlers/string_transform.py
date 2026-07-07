from __future__ import annotations

from tools.deterministic_solver.handlers.string_handler import StringHandler

from ..contracts import default_outputs, input_field, io_contract
from .solver_backed import SolverBackedRouterHandler


class StringTransformRouterHandler(SolverBackedRouterHandler):
    name = "string_transform"
    capability_description = (
        "Perform exact string transformations and counts, including uppercase, lowercase, "
        "title case, reverse text, remove spaces, character count, and word count."
    )
    supported_attachment_types: set[str] = {".txt", ".json"}
    routing_terms = {"uppercase", "lowercase", "reverse", "spaces", "characters", "words", "string", "title"}
    missing_inputs = ["quoted_or_inline_text", "string_operation"]
    input_schema = io_contract(
        name,
        [
            input_field("quoted_or_inline_text", "str", True, "Text to transform.", "question|attachment"),
            input_field("string_operation", "str", True, "String operation such as uppercase, reverse, or word count.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(StringHandler())


__all__ = ["StringTransformRouterHandler"]
