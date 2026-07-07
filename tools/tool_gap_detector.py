from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CapabilityRequirement:
    capability: str
    reason: str


@dataclass
class ToolGapReport:
    required: list[CapabilityRequirement] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def has_gap(self) -> bool:
        return bool(self.missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": [asdict(item) for item in self.required],
            "matched": dict(self.matched),
            "missing": list(self.missing),
            "has_gap": self.has_gap,
        }


class ToolGapDetector:
    """
    從問題與附件型別辨識所需工具能力，並與 registry 比對。

    Args:
        - registry: 支援 find_by_capability 的 ToolRegistry。

    Returns:
        - ToolGapDetector: 能力需求與缺口偵測器。
    """

    CAPABILITY_PATTERNS = (
        (
            "grid.word_search",
            r"\b(boggle|word search|letter grid|find words? in (?:the )?grid)\b",
            "question requires grid-based word search or DFS",
        ),
        (
            "graph.shortest_path",
            r"\b(shortest (?:path|route)|fewest (?:stops|stations|hops)|minimum (?:stops|stations|hops))\b",
            "question requires graph shortest-path computation",
        ),
        (
            "graph.traversal",
            r"\b(graph|nodes?|edges?|connected|reachable|route map|network)\b",
            "question requires graph traversal",
        ),
        (
            "graph.station_count",
            r"\b(stations?|stops?)\b.*\b(count|how many|between|route)\b|\bhow many\b.*\b(stations?|stops?)\b",
            "question requires station or stop counting",
        ),
        (
            "geometry.coordinate_distance",
            r"\b(coordinates?|latitude|longitude|euclidean distance|haversine|distance between points?)\b",
            "question requires coordinate distance calculation",
        ),
        (
            "conversion.sexagesimal",
            r"\b(sexagesimal|dms)\b|\bdegrees?\b.*\bminutes?\b.*\bseconds?\b|\bhours?\b.*\bminutes?\b.*\bseconds?\b",
            "question requires base-60 or degree-minute-second conversion",
        ),
        (
            "table.filter",
            r"\b(table|spreadsheet|csv|xlsx|rows?)\b.*\b(filter|where|matching|greater than|less than)\b|\b(filter|where|matching|greater than|less than)\b.*\b(table|spreadsheet|csv|xlsx|rows?)\b",
            "question requires table filtering",
        ),
        (
            "table.statistics",
            r"\b(table|spreadsheet|csv|xlsx|column)\b.*\b(count|sum|average|mean|median|unique|duplicate|group)\b|\b(count|sum|average|mean|median|unique|duplicate|group)\b.*\b(table|spreadsheet|csv|xlsx|column)\b",
            "question requires table aggregation or statistics",
        ),
        (
            "table.cell_lookup",
            r"\b(table|spreadsheet|csv|xlsx)\b.*\b(cell|row|column)\b",
            "question requires table cell lookup",
        ),
        (
            "math.statistics",
            r"\b(average|mean|median|maximum|minimum|sum|total)\b",
            "question requires deterministic statistics",
        ),
        (
            "math.arithmetic",
            r"\b(calculate|compute|arithmetic|percentage|percent)\b|[-+]?\d+(?:\.\d+)?\s*[%+\-*/]\s*[-+]?\d+",
            "question requires arithmetic",
        ),
        (
            "physics.density",
            r"\b(density|mass per unit volume|determine_density)\b",
            "question requires density calculation with mass and volume units",
        ),
        (
            "list.sort",
            r"\b(sort|order alphabetically|ascending|descending)\b",
            "question requires deterministic sorting",
        ),
        (
            "string.transform",
            r"\b(reverse string|uppercase|lowercase|title case|remove spaces)\b",
            "question requires string transformation",
        ),
        (
            "video.transcript",
            r"\b(video|youtube|film|documentary|watch|listen|transcript|caption|subtitles?)\b",
            "question requires video captions, transcript, or audio ASR",
        ),
        (
            "web.search",
            r"\b(who|when|where|website|published|according to|official source|latest)\b",
            "question may require external factual lookup",
        ),
    )

    ATTACHMENT_CAPABILITIES = {
        "csv": "attachment.table",
        "tsv": "attachment.table",
        "xls": "attachment.table",
        "xlsx": "attachment.table",
        "jpg": "attachment.media",
        "jpeg": "attachment.media",
        "png": "attachment.media",
        "mp3": "attachment.media",
        "wav": "attachment.media",
        "mp4": "attachment.media",
        "mov": "attachment.media",
        "zip": "attachment.archive",
    }

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def detect(
        self,
        question: str,
        *,
        attachment_type: str | None = None,
        enabled_tools: set[str] | None = None,
        requested_tool_name: str = "",
    ) -> ToolGapReport:
        text = " ".join(
            part for part in (str(question or ""), str(requested_tool_name or "")) if part
        ).lower()
        requirements: list[CapabilityRequirement] = []
        seen: set[str] = set()

        for capability, pattern, reason in self.CAPABILITY_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                self._append_requirement(requirements, seen, capability, reason)

        extension = str(attachment_type or "").lower().lstrip(".")
        if extension:
            capability = self.ATTACHMENT_CAPABILITIES.get(extension, "attachment.read")
            self._append_requirement(
                requirements,
                seen,
                capability,
                f"attachment type .{extension} requires a compatible reader",
            )

        matched: dict[str, list[str]] = {}
        missing: list[str] = []
        for requirement in requirements:
            tools = self.registry.find_by_capability(requirement.capability)
            names = [
                tool.name
                for tool in tools
                if enabled_tools is None or tool.name in enabled_tools
            ]
            if names:
                matched[requirement.capability] = names
            else:
                missing.append(requirement.capability)

        return ToolGapReport(
            required=requirements,
            matched=matched,
            missing=missing,
        )

    def _append_requirement(
        self,
        requirements: list[CapabilityRequirement],
        seen: set[str],
        capability: str,
        reason: str,
    ) -> None:
        if capability in seen:
            return
        requirements.append(CapabilityRequirement(capability=capability, reason=reason))
        seen.add(capability)


__all__ = ["CapabilityRequirement", "ToolGapDetector", "ToolGapReport"]
