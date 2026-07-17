from __future__ import annotations

from tools.deterministic_solver.handlers.list_handler import ListHandler

from ..contracts import default_outputs, input_field, io_contract
from .solver_backed import SolverBackedRouterHandler


class ListOperationsRouterHandler(SolverBackedRouterHandler):
    name = "list_operations"
    capability_description = (
        "Perform exact list operations such as count items, select the nth item, sort "
        "alphabetically, sort numbers, reverse order, or order comma-separated items."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json", ".docx"}
    routing_terms = {"list", "items", "sort", "order", "alphabetically", "count", "first", "second", "third"}
    missing_inputs = ["list_items", "list_operation"]
    input_schema = io_contract(
        name,
        [
            input_field("list_items", "list[str]", True, "Items to operate on.", "question|attachment"),
            input_field("list_operation", "str", True, "Count, nth item, sort, reverse, or similar operation.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(ListHandler())


__all__ = ["ListOperationsRouterHandler"]
