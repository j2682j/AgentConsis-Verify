from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


_SMALL_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


class RoadIntervalCoverageRouterHandler:
    """以貪婪演算法求 ASCII 路線圖上覆蓋所有標記點的最少設施數。"""

    name = "road_interval_coverage"
    handler_role = "interval_point_coverage"
    uses_specialized_attachment_parser = True
    capability_description = (
        "Compute the minimum number of facilities with a fixed coverage radius "
        "needed to cover every marker position in an ASCII road-layout file "
        "using the optimal greedy interval sweep."
    )
    supported_attachment_types = {".txt"}
    supported_task_roles = {"interval_point_coverage", "minimum_coverage"}
    supported_answer_roles = {"count", "number"}
    input_schema = io_contract(
        name,
        [
            input_field("file_path", "str", True, "Road layout text file.", "attachment"),
            input_field("coverage_radius", "int", True, "Coverage radius in road units.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        path = self._file_path(handler_input)
        question = handler_input.question.casefold()
        ready = bool(
            path
            and Path(path).suffix.casefold() in self.supported_attachment_types
            and self._coverage_radius(question)
            and re.search(r"\bminimum\b", question)
            and re.search(r"\bcover", question)
        )
        return HandlerMatch(
            handler_name=self.name,
            matched=ready,
            confidence=0.99 if ready else 0.0,
            reason="minimum_radius_coverage_over_road_layout",
            handler_role=self.handler_role,
            missing_inputs=[] if ready else ["road_layout_or_radius"],
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        return {
            "file_path": self._file_path(handler_input),
            "coverage_radius": self._coverage_radius(handler_input.question.casefold()),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        file_path = str(inputs.get("file_path") or "").strip()
        radius = int(inputs.get("coverage_radius") or 0)
        if not file_path or not Path(file_path).is_file() or radius <= 0:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["file_path", "coverage_radius"],
            )
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        positions = self._marker_positions(text)
        if not positions:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["marker_positions"],
            )
        # Greedy sweep is optimal for interval point coverage: put each
        # facility at (leftmost uncovered marker + radius), which covers every
        # marker within 2*radius of that leftmost one.
        count = 0
        facilities: list[int] = []
        covered_until: int | None = None
        for position in positions:
            if covered_until is None or position > covered_until:
                count += 1
                facilities.append(position + radius)
                covered_until = position + 2 * radius
        evidence = (
            f"Marker positions at road units {positions} with coverage radius "
            f"{radius} require {count} facilities (greedy placements at units "
            f"{facilities})."
        )
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=str(count),
            evidence_text=evidence,
            confidence=1.0,
            output_type="final_answer",
            semantic_role="minimum_coverage_count",
            supporting_inputs=[file_path, str(radius), *map(str, positions)],
            structured_result={
                "task_type": "interval_point_coverage",
                "operation": "greedy_interval_coverage",
                "coverage_radius": radius,
                "marker_positions": positions,
                "facility_positions": facilities,
                "input_provenance": {
                    "source": "specialized_attachment_input",
                    "file_path": file_path,
                    "parse_status": "success",
                },
            },
        )

    @staticmethod
    def _marker_positions(text: str) -> list[int]:
        """Column indices of marker characters on non-road lines.

        The road is the line made of dashes (mile markers); every other line
        contributes the column index of each non-space character as a marker
        position measured in road units.
        """
        lines = [line.rstrip("\n") for line in text.splitlines()]
        road_lines = [
            line for line in lines if len(line.strip()) >= 3 and set(line.strip()) == {"-"}
        ]
        if not road_lines:
            return []
        positions: set[int] = set()
        for line in lines:
            if line in road_lines:
                continue
            for index, char in enumerate(line):
                if not char.isspace():
                    positions.add(index)
        return sorted(positions)

    @staticmethod
    def _coverage_radius(question: str) -> int:
        match = re.search(
            r"\b(\d+|" + "|".join(_SMALL_NUMBERS) + r")[- ](?:mile|kilometer|km|unit)s?\b[^.]*?\bradius\b"
            r"|\bradius\b[^.]*?\b(\d+|" + "|".join(_SMALL_NUMBERS) + r")[- ](?:mile|kilometer|km|unit)s?\b",
            question,
        )
        if not match:
            return 0
        token = next((group for group in match.groups() if group), "")
        return int(token) if token.isdigit() else _SMALL_NUMBERS.get(token, 0)

    @staticmethod
    def _file_path(handler_input: HandlerInput) -> str:
        adapted = handler_input.adapted_inputs()
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        return str(
            adapted.get("file_path")
            or attachment.get("file_path")
            or attachment.get("path")
            or ""
        ).strip()


__all__ = ["RoadIntervalCoverageRouterHandler"]
