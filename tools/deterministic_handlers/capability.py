from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HandlerCapability:
    handler_name: str
    handler_role: str = ""
    capability: str = ""
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    available_inputs: list[str] = field(default_factory=list)
    supported_attachment_types: list[str] = field(default_factory=list)
    output_type: str = "final_answer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandlerPreflightResult:
    handler_name: str = ""
    handler_role: str = ""
    status: str = "handler_unavailable"
    required_inputs: list[str] = field(default_factory=list)
    available_inputs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    adapted_inputs: dict[str, Any] = field(default_factory=dict)
    input_provenance: dict[str, Any] = field(default_factory=dict)
    attachment_bound: bool = False
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapted_inputs"] = {
            key: _summarize(value) for key, value in self.adapted_inputs.items()
        }
        return payload


def _summarize(value: Any) -> Any:
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": [str(key) for key in list(value)[:12]]}
    if isinstance(value, str):
        return value[:180] + " ..." if len(value) > 180 else value
    return value


__all__ = ["HandlerCapability", "HandlerPreflightResult"]
