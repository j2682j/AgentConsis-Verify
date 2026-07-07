from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "deterministic-handler-v1"


@dataclass
class HandlerInputField:
    name: str
    type: str
    required: bool = True
    description: str = ""
    source: str = "question"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandlerOutputField:
    name: str
    type: str
    required: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HandlerIOContract:
    handler_name: str
    input_fields: list[HandlerInputField] = field(default_factory=list)
    output_fields: list[HandlerOutputField] = field(default_factory=list)
    supported_attachment_types: set[str] = field(default_factory=set)
    examples: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def required_input_names(self) -> list[str]:
        return [field.name for field in self.input_fields if field.required]

    def required_output_names(self) -> list[str]:
        return [field.name for field in self.output_fields if field.required]

    def required_missing(self, available: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for field_info in self.input_fields:
            if not field_info.required:
                continue
            value = available.get(field_info.name)
            if value is None or value == "" or value == [] or value == {}:
                missing.append(field_info.name)
        return missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_name": self.handler_name,
            "schema_version": self.schema_version,
            "input_fields": [field_info.to_dict() for field_info in self.input_fields],
            "output_fields": [field_info.to_dict() for field_info in self.output_fields],
            "supported_attachment_types": sorted(self.supported_attachment_types),
            "examples": list(self.examples),
        }


__all__ = [
    "SCHEMA_VERSION",
    "HandlerIOContract",
    "HandlerInputField",
    "HandlerOutputField",
]
