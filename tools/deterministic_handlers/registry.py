from __future__ import annotations

from typing import Iterable

from .base import DeterministicHandler
from .capability import HandlerCapability
from .handlers import (
    BinaryOperationTableRouterHandler,
    BoggleDFSRouterHandler,
    ChessTacticsRouterHandler,
    ChessImageSolverRouterHandler,
    CoordinateDistanceRouterHandler,
    ColorGridHamiltonianRouterHandler,
    DateTimeRouterHandler,
    FractionDocumentRouterHandler,
    GraphShortestPathRouterHandler,
    ListOperationsRouterHandler,
    LogicEquivalenceRouterHandler,
    MultiStepCountingRouterHandler,
    NumericReasoningRouterHandler,
    ProbabilitySimulationRouterHandler,
    SexagesimalConversionRouterHandler,
    SimpleMathRouterHandler,
    StringTransformRouterHandler,
    TableAggregationRouterHandler,
    TableExactRouterHandler,
    TextExtractionRouterHandler,
    UnitConversionRouterHandler,
)
from .schema import HandlerIOContract


class HandlerRegistry:
    """
    Register deterministic handlers that can be selected by the router.
    """

    def __init__(self, handlers: Iterable[DeterministicHandler] | None = None) -> None:
        self._handlers: dict[str, DeterministicHandler] = {}
        self._role_aliases: dict[str, set[str]] = {
            "binary_operation_reasoning": {"binary_operation_table"},
            "boggle_dfs": {"boggle_dfs"},
            "chess_tactics": {"chess_image_solver", "chess_tactics"},
            "coordinate_distance": {"coordinate_distance"},
            "date_time": {"date_time"},
            "fraction_document_reasoning": {"fraction_document"},
            "graph_search": {"graph_shortest_path"},
            "grid_hamiltonian_cycle": {"color_grid_hamiltonian"},
            "list_operation": {"list_operations"},
            "logic_equivalence": {"logic_equivalence"},
            "multi_step_counting": {"multi_step_counting"},
            "numeric_arithmetic": {"numeric_reasoning", "simple_math"},
            "probability_simulation": {"probability_simulation"},
            "simple_math": {"simple_math"},
            "sexagesimal_conversion": {"sexagesimal_conversion"},
            "string_transform": {"string_transform"},
            "table_reasoning": {"table_exact_operations", "table_aggregation"},
            "text_extraction": {"text_extraction"},
            "unit_conversion": {"unit_conversion"},
        }
        for handler in handlers or []:
            self.register(handler)

    def register(self, handler: DeterministicHandler) -> None:
        self._handlers[handler.name] = handler

    def list_handlers(self) -> list[DeterministicHandler]:
        return list(self._handlers.values())

    def get(self, name: str) -> DeterministicHandler | None:
        return self._handlers.get(name)

    def find_by_role(self, role: str) -> list[DeterministicHandler]:
        key = str(role or "").strip().lower()
        if not key:
            return []
        return [
            handler
            for handler in self._handlers.values()
            if handler.name in self._role_aliases.get(key, set())
            or str(getattr(handler, "handler_role", "") or "").strip().lower() == key
            or key in {
                str(item or "").strip().lower()
                for item in getattr(handler, "supported_task_roles", set()) or set()
            }
        ]

    def role_for_handler(self, handler_name: str) -> str:
        name = str(handler_name or "").strip()
        for role, names in self._role_aliases.items():
            if name in names:
                return role
        handler = self.get(name)
        return str(getattr(handler, "handler_role", "") or "") if handler else ""

    def capability_for(
        self,
        handler_name: str,
        *,
        available_inputs: Iterable[str] = (),
    ) -> HandlerCapability | None:
        handler = self.get(handler_name)
        if handler is None:
            return None
        contract = getattr(handler, "input_schema", None)
        required_inputs: list[str] = []
        optional_inputs: list[str] = []
        supported_types = set(getattr(handler, "supported_attachment_types", set()) or set())
        if isinstance(contract, HandlerIOContract):
            required_inputs = contract.required_input_names()
            optional_inputs = [
                field.name for field in contract.input_fields if not field.required
            ]
            supported_types.update(contract.supported_attachment_types)
        return HandlerCapability(
            handler_name=handler.name,
            handler_role=self.role_for_handler(handler.name),
            capability=str(getattr(handler, "capability_description", "") or ""),
            required_inputs=required_inputs,
            optional_inputs=optional_inputs,
            available_inputs=sorted({str(item) for item in available_inputs if str(item)}),
            supported_attachment_types=sorted(supported_types),
            output_type="final_answer",
        )


def default_deterministic_registry() -> HandlerRegistry:
    return HandlerRegistry(
        [
            BinaryOperationTableRouterHandler(),
            BoggleDFSRouterHandler(),
            ChessImageSolverRouterHandler(),
            ChessTacticsRouterHandler(),
            CoordinateDistanceRouterHandler(),
            ColorGridHamiltonianRouterHandler(),
            GraphShortestPathRouterHandler(),
            DateTimeRouterHandler(),
            FractionDocumentRouterHandler(),
            LogicEquivalenceRouterHandler(),
            MultiStepCountingRouterHandler(),
            ProbabilitySimulationRouterHandler(),
            SexagesimalConversionRouterHandler(),
            UnitConversionRouterHandler(),
            TableExactRouterHandler(),
            TableAggregationRouterHandler(),
            ListOperationsRouterHandler(),
            StringTransformRouterHandler(),
            TextExtractionRouterHandler(),
            NumericReasoningRouterHandler(),
            SimpleMathRouterHandler(),
        ]
    )


__all__ = ["HandlerRegistry", "default_deterministic_registry"]
