from __future__ import annotations

from tools.deterministic_solver.handlers.grid_word_handler import GridWordHandler

from ..contracts import default_outputs, input_field, io_contract, output_field
from .solver_backed import SolverBackedRouterHandler


class BoggleDFSRouterHandler(SolverBackedRouterHandler):
    name = "boggle_dfs"
    capability_description = (
        "Solve Boggle, word search, or letter-grid tasks by checking whether candidate "
        "words can be formed through adjacent grid cells."
    )
    supported_attachment_types: set[str] = {".txt", ".json"}
    routing_terms = {"boggle", "grid", "letter", "word", "search", "formed"}
    missing_inputs = ["grid", "candidate_words"]
    input_schema = io_contract(
        name,
        [
            input_field("grid", "list[list[str]]", True, "Letter grid.", "question|attachment"),
            input_field("candidate_words", "list[str]", True, "Words to test against the grid.", "question|attachment"),
        ],
        [
            *default_outputs(),
            output_field("found_words", "list[str]", False, "Candidate words that can be formed."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(GridWordHandler())


__all__ = ["BoggleDFSRouterHandler"]
