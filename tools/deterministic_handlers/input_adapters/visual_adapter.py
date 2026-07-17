from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput
from .base import AdapterResult, parsed_payload, payload_provenance


class VisualInputAdapter:
    handler_names = {"text_extraction", "boggle_dfs", "numeric_reasoning"}

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        payload = parsed_payload(handler_input)
        visual_blocks = [
            block for block in list(payload.get("visual_blocks") or []) if isinstance(block, dict)
        ]
        text_blocks = [
            block for block in list(payload.get("text_blocks") or []) if isinstance(block, dict)
        ]
        source_text = "\n".join(
            str(block.get("text") or "").strip()
            for block in [*visual_blocks, *text_blocks]
            if str(block.get("text") or "").strip()
        )
        provenance = payload_provenance(handler_input)

        if handler_name == "text_extraction":
            return AdapterResult(
                status="ready" if source_text else "missing_inputs",
                inputs={"source_text": source_text} if source_text else {},
                missing_inputs=[] if source_text else ["source_text"],
                input_provenance=provenance,
                reason="typed_text_or_visual_blocks",
            )

        attributes = self._attributes(visual_blocks)
        if handler_name == "boggle_dfs":
            grid = attributes.get("grid")
            words = attributes.get("candidate_words") or attributes.get("words")
            ready = isinstance(grid, list) and bool(grid) and isinstance(words, list) and bool(words)
            return AdapterResult(
                status="ready" if ready else "missing_inputs",
                inputs={"grid": grid, "candidate_words": words} if ready else {},
                missing_inputs=[] if ready else ["grid", "candidate_words"],
                input_provenance=provenance,
                reason="typed_visual_grid" if ready else "typed_visual_grid_missing",
            )

        numbers = attributes.get("numbers")
        if not isinstance(numbers, list):
            numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", source_text)]
        return AdapterResult(
            status="ready" if numbers else "missing_inputs",
            inputs={"numbers": numbers} if numbers else {},
            missing_inputs=[] if numbers else ["numbers"],
            input_provenance=provenance,
            reason="typed_visual_numbers" if numbers else "typed_visual_numbers_missing",
        )

    @staticmethod
    def _attributes(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for block in blocks:
            attributes = block.get("attributes")
            if isinstance(attributes, dict):
                merged.update(attributes)
        return merged


__all__ = ["VisualInputAdapter"]
