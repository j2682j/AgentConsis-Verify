from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..base import HandlerInput


@dataclass
class AdapterResult:
    status: str = "not_applicable"
    inputs: dict[str, Any] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    input_provenance: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HandlerInputAdapter(Protocol):
    handler_names: set[str]

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        ...


def parsed_payload(handler_input: HandlerInput) -> dict[str, Any]:
    metadata = handler_input.metadata if isinstance(handler_input.metadata, dict) else {}
    payload = metadata.get("parsed_payload")
    return payload if isinstance(payload, dict) else {}


def payload_provenance(handler_input: HandlerInput) -> dict[str, Any]:
    provenance = parsed_payload(handler_input).get("provenance")
    return dict(provenance) if isinstance(provenance, dict) else {}


__all__ = [
    "AdapterResult",
    "HandlerInputAdapter",
    "parsed_payload",
    "payload_provenance",
]
