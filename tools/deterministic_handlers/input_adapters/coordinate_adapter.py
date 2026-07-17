from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput
from .base import AdapterResult, parsed_payload, payload_provenance


class CoordinateInputAdapter:
    handler_names = {"coordinate_distance"}

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        del handler_name
        payload = parsed_payload(handler_input)
        records = [
            item for item in list(payload.get("coordinates") or []) if isinstance(item, dict)
        ]
        if len(records) < 2:
            return AdapterResult(
                status="missing_inputs",
                missing_inputs=["coordinates"],
                input_provenance=payload_provenance(handler_input),
                reason="typed_coordinates_missing",
            )

        metadata = handler_input.metadata if isinstance(handler_input.metadata, dict) else {}
        strategy = metadata.get("attachment_strategy")
        strategy = strategy if isinstance(strategy, dict) else {}
        target_text = "\n".join(
            value
            for value in [
                handler_input.question,
                str(strategy.get("information_need") or ""),
            ]
            if value
        )
        selected = self._select_records(records, target_text)
        if len(selected) != 2:
            return AdapterResult(
                status="ambiguous_inputs" if selected else "missing_inputs",
                missing_inputs=["two_coordinate_targets"],
                input_provenance=payload_provenance(handler_input),
                reason=(
                    f"coordinate_target_count={len(selected)}; "
                    "two uniquely identified records are required"
                ),
            )
        pairs = [self._point(record) for record in selected]
        return AdapterResult(
            status="ready",
            inputs={
                "pairs": pairs,
                "coordinate_records": selected,
                "use_haversine": False,
            },
            input_provenance=payload_provenance(handler_input),
            reason="two_typed_coordinate_targets",
        )

    def _select_records(
        self,
        records: list[dict[str, Any]],
        target_text: str,
    ) -> list[dict[str, Any]]:
        lowered = target_text.casefold()
        selected: list[dict[str, Any]] = []
        explicit_ids = {
            match.group(1)
            for match in re.finditer(
                r"\batom(?:\s+(?:serial|number|id))?\s*#?\s*(\d+)\b",
                target_text,
                flags=re.IGNORECASE,
            )
        }
        for record in records:
            identifier = str(record.get("identifier") or "")
            attributes = record.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            aliases = self._aliases(identifier, attributes)
            if identifier in explicit_ids or any(alias in lowered for alias in aliases):
                selected.append(record)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in selected:
            key = str(record.get("identifier") or id(record))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    def _aliases(self, identifier: str, attributes: dict[str, Any]) -> list[str]:
        atom = str(attributes.get("atom") or "").casefold()
        residue = str(attributes.get("residue") or "").casefold()
        residue_index = str(attributes.get("residue_index") or "").casefold()
        chain = str(attributes.get("chain") or "").casefold()
        aliases: list[str] = []
        if atom and residue_index:
            aliases.extend(
                [
                    f"atom {atom} residue {residue_index}",
                    f"{atom} of residue {residue_index}",
                    f"{residue} {residue_index} {atom}".strip(),
                ]
            )
        if chain and atom and residue_index:
            aliases.append(f"chain {chain} residue {residue_index} atom {atom}")
        return [alias for alias in aliases if len(alias.split()) >= 3]

    @staticmethod
    def _point(record: dict[str, Any]) -> tuple[float, ...]:
        point = (float(record["x"]), float(record["y"]))
        if record.get("z") is not None:
            return (*point, float(record["z"]))
        return point


__all__ = ["CoordinateInputAdapter"]
