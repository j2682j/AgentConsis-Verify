from __future__ import annotations

import re

from tools.deterministic_solver.handlers.sexagesimal_handler import SexagesimalHandler

from ..base import HandlerInput, HandlerMatch
from ..contracts import default_outputs, input_field, io_contract, output_field
from .solver_backed import SolverBackedRouterHandler


class SexagesimalConversionRouterHandler(SolverBackedRouterHandler):
    name = "sexagesimal_conversion"
    capability_description = (
        "Convert between decimal degrees and sexagesimal degree-minute-second notation, "
        "including DMS, base-60 coordinates, latitude, and longitude values."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    routing_terms = {"sexagesimal", "dms", "degrees", "minutes", "seconds", "decimal"}
    missing_inputs = ["sexagesimal_or_decimal_degree_value"]
    input_schema = io_contract(
        name,
        [
            input_field("sexagesimal_or_decimal_degree_value", "str", True, "DMS or decimal degree value to convert.", "question|attachment"),
            input_field("conversion_direction", "str", False, "Decimal-to-DMS or DMS-to-decimal direction.", "question"),
        ],
        [
            *default_outputs(),
            output_field("converted_value", "str|float", False, "Converted coordinate value."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(SexagesimalHandler())

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        text = handler_input.combined_text()
        if (
            re.search(r"\b(distance|between|haversine|coordinate|coordinates)\b", text, re.IGNORECASE)
            and len(re.findall(r"\b(?:degrees?|deg|°)\b", text, re.IGNORECASE)) >= 4
        ):
            return HandlerMatch(
                handler_name=self.name,
                matched=False,
                confidence=0.2,
                reason="sexagesimal_value_is_part_of_coordinate_distance_task",
                missing_inputs=["pure_conversion_request"],
            )
        return super().match_input(handler_input)


__all__ = ["SexagesimalConversionRouterHandler"]
