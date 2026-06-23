from __future__ import annotations

import math
import re

from ..schemas import DeterministicSolverResult
from .common import lower_text


class CoordinateHandler:
    """Calculate Euclidean or Haversine distance between two coordinate pairs."""

    PAIR_RE = re.compile(
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)"
    )

    def solve(self, question: str, **_: object) -> DeterministicSolverResult:
        lowered = lower_text(question)
        if not any(term in lowered for term in ("coordinate", "latitude", "longitude", "distance between")):
            return DeterministicSolverResult.miss("coordinate")
        pairs = [
            (float(left), float(right))
            for left, right in self.PAIR_RE.findall(question)
        ]
        if len(pairs) < 2:
            return DeterministicSolverResult.miss("coordinate", "two coordinate pairs are required")

        first, second = pairs[:2]
        if "latitude" in lowered or "longitude" in lowered or "haversine" in lowered:
            value = self._haversine(first, second)
            unit = "km"
            task_type = "coordinate_haversine_distance"
        else:
            value = math.dist(first, second)
            unit = ""
            task_type = "coordinate_euclidean_distance"
        answer = self._format(value, unit)
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type=task_type,
            answer=value,
            answer_text=answer,
            confidence=0.96,
            evidence={"point_a": first, "point_b": second, "unit": unit},
        )

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


__all__ = ["CoordinateHandler"]
