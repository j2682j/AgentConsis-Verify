from __future__ import annotations

from ..base import HandlerInput
from .base import AdapterResult, HandlerInputAdapter
from .attachment_file_adapter import AttachmentFileInputAdapter
from .coordinate_adapter import CoordinateInputAdapter
from .relation_adapter import RelationInputAdapter
from .table_adapter import TableInputAdapter
from .visual_adapter import VisualInputAdapter


class HandlerInputAdapterRegistry:
    def __init__(self, adapters: list[HandlerInputAdapter] | None = None) -> None:
        self._adapters: dict[str, HandlerInputAdapter] = {}
        for adapter in adapters or [
            AttachmentFileInputAdapter(),
            TableInputAdapter(),
            CoordinateInputAdapter(),
            RelationInputAdapter(),
            VisualInputAdapter(),
        ]:
            for handler_name in adapter.handler_names:
                self._adapters[handler_name] = adapter

    def has_adapter(self, handler_name: str) -> bool:
        return str(handler_name or "") in self._adapters

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        adapter = self._adapters.get(str(handler_name or ""))
        if adapter is None:
            return AdapterResult(status="not_applicable")
        return adapter.adapt(handler_name, handler_input)


__all__ = ["HandlerInputAdapterRegistry"]
