from __future__ import annotations

from typing import Iterable

from .base import DeterministicHandler
from .handlers import (
    BoggleDFSRouterHandler,
    CoordinateDistanceRouterHandler,
    DateTimeRouterHandler,
    GraphShortestPathRouterHandler,
    ListOperationsRouterHandler,
    NumericReasoningRouterHandler,
    SexagesimalConversionRouterHandler,
    SimpleMathRouterHandler,
    StringTransformRouterHandler,
    TableAggregationRouterHandler,
    TableExactRouterHandler,
    TextExtractionRouterHandler,
    UnitConversionRouterHandler,
)


class HandlerRegistry:
    """
    Register deterministic handlers that can be selected by the router.
    """

    def __init__(self, handlers: Iterable[DeterministicHandler] | None = None) -> None:
        self._handlers: dict[str, DeterministicHandler] = {}
        self._role_aliases: dict[str, set[str]] = {
            "boggle_dfs": {"boggle_dfs"},
            "coordinate_distance": {"coordinate_distance"},
            "date_time": {"date_time"},
            "graph_search": {"graph_shortest_path"},
            "list_operation": {"list_operations"},
            "numeric_arithmetic": {"numeric_reasoning", "simple_math"},
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


def default_deterministic_registry() -> HandlerRegistry:
    return HandlerRegistry(
        [
            BoggleDFSRouterHandler(),
            CoordinateDistanceRouterHandler(),
            GraphShortestPathRouterHandler(),
            DateTimeRouterHandler(),
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
