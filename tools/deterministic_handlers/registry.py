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
        for handler in handlers or []:
            self.register(handler)

    def register(self, handler: DeterministicHandler) -> None:
        self._handlers[handler.name] = handler

    def list_handlers(self) -> list[DeterministicHandler]:
        return list(self._handlers.values())

    def get(self, name: str) -> DeterministicHandler | None:
        return self._handlers.get(name)


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
