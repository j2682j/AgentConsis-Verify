from __future__ import annotations

import re

from ..schemas import DeterministicSolverResult
from .common import lower_text


class SexagesimalHandler:
    """Convert between decimal degrees and degree-minute-second notation."""

    DMS_RE = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:degrees?|°)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:minutes?|'|′)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|\"|″)?\s*([NSEW])?",
        flags=re.IGNORECASE,
    )

    def solve(self, question: str, **_: object) -> DeterministicSolverResult:
        lowered = lower_text(question)
        if not any(term in lowered for term in ("sexagesimal", "degrees", "dms", "base 60")):
            return DeterministicSolverResult.miss("sexagesimal")

        match = self.DMS_RE.search(question)
        if match:
            degrees = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            direction = (match.group(4) or "").upper()
            sign = -1 if degrees < 0 or direction in {"S", "W"} else 1
            decimal = sign * (abs(degrees) + minutes / 60 + seconds / 3600)
            answer = f"{decimal:.10f}".rstrip("0").rstrip(".")
            return DeterministicSolverResult(
                used_deterministic_solver=True,
                task_type="sexagesimal_to_decimal",
                answer=decimal,
                answer_text=answer,
                confidence=0.98,
                evidence={
                    "degrees": degrees,
                    "minutes": minutes,
                    "seconds": seconds,
                    "direction": direction,
                },
            )

        decimal_match = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s*(?:decimal\s+degrees?|degrees?)",
            question,
            flags=re.IGNORECASE,
        )
        if decimal_match and ("dms" in lowered or "sexagesimal" in lowered):
            value = float(decimal_match.group(1))
            sign = "-" if value < 0 else ""
            absolute = abs(value)
            degrees = int(absolute)
            minute_value = (absolute - degrees) * 60
            minutes = int(minute_value)
            seconds = (minute_value - minutes) * 60
            answer = f"{sign}{degrees}° {minutes}' {seconds:.6f}\""
            return DeterministicSolverResult(
                used_deterministic_solver=True,
                task_type="decimal_to_sexagesimal",
                answer=answer,
                answer_text=answer,
                confidence=0.98,
                evidence={"decimal_degrees": value},
            )
        return DeterministicSolverResult.miss("sexagesimal", "no supported sexagesimal value found")


__all__ = ["SexagesimalHandler"]
