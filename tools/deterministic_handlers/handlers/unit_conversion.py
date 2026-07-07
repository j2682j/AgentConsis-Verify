from __future__ import annotations

from tools.deterministic_solver.handlers.unit_handler import UnitHandler

from ..contracts import default_outputs, input_field, io_contract, output_field
from .solver_backed import SolverBackedRouterHandler


class UnitConversionRouterHandler(SolverBackedRouterHandler):
    name = "unit_conversion"
    capability_description = (
        "Convert units exactly, including length, mass, time, and Celsius/Fahrenheit "
        "temperature conversion when the numeric value and target unit are provided."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    routing_terms = {"convert", "conversion", "unit", "units", "celsius", "fahrenheit", "meters", "grams"}
    missing_inputs = ["numeric_value", "source_unit", "target_unit"]
    input_schema = io_contract(
        name,
        [
            input_field("numeric_value", "float", True, "Value to convert.", "question|attachment"),
            input_field("source_unit", "str", True, "Original unit.", "question|attachment"),
            input_field("target_unit", "str", True, "Target unit.", "question"),
        ],
        [
            *default_outputs(),
            output_field("converted_value", "float|str", False, "Converted value."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(UnitHandler())


__all__ = ["UnitConversionRouterHandler"]
