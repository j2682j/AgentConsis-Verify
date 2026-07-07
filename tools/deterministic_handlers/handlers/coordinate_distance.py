from __future__ import annotations

import math
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field


class CoordinateDistanceRouterHandler:
    name = "coordinate_distance"
    capability_description = (
        "Compute exact Euclidean or haversine distance between two coordinate pairs, "
        "latitude/longitude pairs, or points with decimal coordinates."
    )
    supported_attachment_types: set[str] = {".txt", ".csv", ".tsv", ".json"}
    routing_terms = {"coordinate", "coordinates", "distance", "latitude", "longitude", "haversine", "euclidean"}
    input_schema = io_contract(
        name,
        [
            input_field("pairs", "list[tuple[float,float]]", True, "Two coordinate pairs.", "question|attachment|search"),
            input_field("use_haversine", "bool", False, "Whether to use Earth-distance calculation.", "question"),
        ],
        [
            *default_outputs(),
            output_field("value", "float", True, "Computed distance."),
            output_field("unit", "str", False, "Distance unit."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    PAIR_RE = re.compile(
        r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)?"
    )
    DMS_COMPONENT_RE = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:minutes?|min|')\s*"
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|sec|\")?\s*([NSEW])?",
        re.IGNORECASE,
    )

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        missing = [] if len(inputs.get("pairs") or []) >= 2 else ["two_coordinate_pairs"]
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.98 if not missing else 0.3,
            reason="coordinate_pair_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        text = handler_input.combined_text()
        pairs = [
            (float(left), float(right))
            for left, right in self.PAIR_RE.findall(text)
        ]
        if len(pairs) < 2:
            pairs = self._dms_pairs(text)
        return {
            "pairs": pairs[:2],
            "use_haversine": bool(
                re.search(r"\b(latitude|longitude|haversine|earth|km|kilometer|degrees?|minutes?|seconds?)\b", text, re.IGNORECASE)
            ),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        pairs = list(inputs.get("pairs") or [])
        if len(pairs) < 2:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["two_coordinate_pairs"],
                structured_result={"pairs": pairs},
            )
        first, second = pairs[:2]
        if inputs.get("use_haversine"):
            value = self._haversine(first, second)
            unit = "km"
            task_type = "coordinate_haversine_distance"
        else:
            value = math.dist(first, second)
            unit = ""
            task_type = "coordinate_euclidean_distance"
        answer = self._format(value, unit)
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            structured_result={
                "task_type": task_type,
                "point_a": first,
                "point_b": second,
                "value": value,
                "unit": unit,
            },
            confidence=0.95,
        )

    def _dms_pairs(self, text: str) -> list[tuple[float, float]]:
        values = [
            self._dms_to_decimal(match.groups())
            for match in self.DMS_COMPONENT_RE.finditer(text or "")
        ]
        pairs: list[tuple[float, float]] = []
        for index in range(0, len(values) - 1, 2):
            pairs.append((values[index], values[index + 1]))
        return pairs

    def _dms_to_decimal(self, groups: tuple[str, str, str, str | None]) -> float:
        degrees = float(groups[0])
        minutes = float(groups[1])
        seconds = float(groups[2])
        direction = str(groups[3] or "").upper()
        sign = -1 if degrees < 0 or direction in {"S", "W"} else 1
        return sign * (abs(degrees) + minutes / 60 + seconds / 3600)

    def _haversine(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        lat1, lon1 = map(math.radians, first)
        lat2, lon2 = map(math.radians, second)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        value = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return 6371.0088 * 2 * math.asin(math.sqrt(value))

    def _format(self, value: float, unit: str) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{text} {unit}".strip()


__all__ = ["CoordinateDistanceRouterHandler"]
