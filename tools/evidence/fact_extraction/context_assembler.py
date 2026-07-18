from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Iterable

from utils.network_utils import normalize_text

from .models import SemanticSourceUnit


@dataclass(frozen=True)
class CrossContextWindow:
    """保存同一來源中可共同進行事實抽取的相鄰語意單位。"""

    window_id: str
    source_id: str
    source_type: str
    text: str
    unit_ids: list[str]
    units: list[SemanticSourceUnit]
    boundary_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossContextAssembler:
    """
    將同一來源中相鄰且具有語意延續性的單位組成有限長度視窗。

    Args:
     - max_windows: 每次任務最多建立的跨上下文視窗數。
     - max_units: 每個視窗最多包含的來源單位數。
     - max_chars: 每個視窗允許的最大字元數。

    Returns:
     - CrossContextAssembler: 可由檢索錨點建立跨 passage 視窗的組裝器。
    """

    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
    _CAPITALIZED_RE = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,})){0,5}\b"
    )
    _REFERENCE_RE = re.compile(
        r"^(?:this|that|these|those|it|its|they|their|he|his|she|her|the\s+(?:former|latter|same))\b",
        re.IGNORECASE,
    )
    _TABLE_MARKER_RE = re.compile(
        r"\b(?:table|row|column|headers?|record\s+type|authors?|date|source)\s*:",
        re.IGNORECASE,
    )
    _STOPWORDS = {
        "about", "after", "also", "and", "are", "before", "between",
        "from", "have", "into", "more", "that", "the", "their", "then",
        "there", "these", "this", "those", "was", "were", "which", "with",
    }

    def __init__(
        self,
        *,
        max_windows: int = 6,
        max_units: int = 3,
        max_chars: int = 1600,
    ) -> None:
        self.max_windows = max(1, int(max_windows))
        self.max_units = max(2, min(3, int(max_units)))
        self.max_chars = max(400, int(max_chars))

    def assemble(
        self,
        units: Iterable[SemanticSourceUnit],
        *,
        anchor_unit_ids: Iterable[str] = (),
    ) -> list[CrossContextWindow]:
        grouped: dict[str, list[SemanticSourceUnit]] = {}
        for unit in units:
            source_id = normalize_text(unit.source_id)
            if not source_id or not normalize_text(unit.text):
                continue
            grouped.setdefault(source_id, []).append(unit)

        anchors = {
            normalize_text(unit_id)
            for unit_id in anchor_unit_ids
            if normalize_text(unit_id)
        }
        windows: list[CrossContextWindow] = []
        seen: set[tuple[str, ...]] = set()
        for source_id, source_units in grouped.items():
            ordered = sorted(source_units, key=self._order_key)
            anchor_indexes = [
                index
                for index, unit in enumerate(ordered)
                if not anchors or unit.unit_id in anchors
            ]
            for anchor_index in anchor_indexes:
                for candidate in self._candidate_groups(ordered, anchor_index):
                    unit_key = tuple(unit.unit_id for unit in candidate)
                    if unit_key in seen:
                        continue
                    boundary_reasons = [
                        self._boundary_reason(left, right)
                        for left, right in zip(candidate, candidate[1:])
                    ]
                    if not boundary_reasons or any(not reason for reason in boundary_reasons):
                        continue
                    text = self._render_bounded(candidate)
                    windows.append(
                        CrossContextWindow(
                            window_id=self._window_id(source_id, unit_key),
                            source_id=source_id,
                            source_type=normalize_text(candidate[0].source_type),
                            text=text,
                            unit_ids=list(unit_key),
                            units=list(candidate),
                            boundary_reason="+".join(self._dedupe(boundary_reasons)),
                            metadata={
                                "unit_count": len(candidate),
                                "anchor_unit_ids": [
                                    unit.unit_id for unit in candidate if unit.unit_id in anchors
                                ],
                            },
                        )
                    )
                    seen.add(unit_key)
                    if len(windows) >= self.max_windows:
                        return windows
        return windows

    def _candidate_groups(
        self,
        units: list[SemanticSourceUnit],
        anchor_index: int,
    ) -> list[list[SemanticSourceUnit]]:
        groups: list[list[SemanticSourceUnit]] = []
        for size in range(2, self.max_units + 1):
            for start in range(max(0, anchor_index - size + 1), anchor_index + 1):
                end = start + size
                if end <= len(units) and start <= anchor_index < end:
                    groups.append(units[start:end])
        return groups

    def _boundary_reason(
        self,
        left: SemanticSourceUnit,
        right: SemanticSourceUnit,
    ) -> str:
        if normalize_text(left.source_id) != normalize_text(right.source_id):
            return ""
        if not self._structurally_adjacent(left, right):
            return ""
        if self._table_continuity(left, right):
            return "table_header_and_row"
        if self._shared_entity(left.text, right.text):
            return "shared_entity"
        if self._reference_continuity(left.text, right.text):
            return "reference_continuity"
        return ""

    def _structurally_adjacent(
        self,
        left: SemanticSourceUnit,
        right: SemanticSourceUnit,
    ) -> bool:
        left_record = normalize_text(str(left.metadata.get("record_id") or ""))
        right_record = normalize_text(str(right.metadata.get("record_id") or ""))
        if left_record and left_record == right_record:
            return True
        left_order = self._numeric_order(left)
        right_order = self._numeric_order(right)
        return left_order is not None and right_order is not None and right_order - left_order == 1

    def _table_continuity(
        self,
        left: SemanticSourceUnit,
        right: SemanticSourceUnit,
    ) -> bool:
        left_table = normalize_text(str(left.metadata.get("table_id") or ""))
        right_table = normalize_text(str(right.metadata.get("table_id") or ""))
        if left_table and left_table == right_table:
            return True
        record_type = {
            normalize_text(str(left.metadata.get("record_type") or "")),
            normalize_text(str(right.metadata.get("record_type") or "")),
        }
        return bool(
            record_type & {"table", "table_row", "database_row", "publication"}
            and (
                self._TABLE_MARKER_RE.search(left.text)
                or self._TABLE_MARKER_RE.search(right.text)
            )
        )

    def _shared_entity(self, left: str, right: str) -> bool:
        left_entities = self._entity_terms(left)
        right_entities = self._entity_terms(right)
        return bool(left_entities & right_entities)

    def _entity_terms(self, text: str) -> set[str]:
        entities = {
            normalize_text(match.group(0)).casefold()
            for match in self._CAPITALIZED_RE.finditer(text)
            if len(normalize_text(match.group(0))) >= 3
        }
        words = {
            match.group(0).casefold()
            for match in self._WORD_RE.finditer(text)
            if len(match.group(0)) >= 5
            and match.group(0).casefold() not in self._STOPWORDS
        }
        return entities | words

    def _reference_continuity(self, left: str, right: str) -> bool:
        right_text = normalize_text(right)
        if not right_text or not self._REFERENCE_RE.search(right_text):
            return False
        return bool(self._entity_terms(left))

    def _order_key(self, unit: SemanticSourceUnit) -> tuple[int, str]:
        order = self._numeric_order(unit)
        return (order if order is not None else 10**9, unit.unit_id)

    @staticmethod
    def _numeric_order(unit: SemanticSourceUnit) -> int | None:
        value = unit.metadata.get("order")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        matches = re.findall(r"\d+", unit.unit_id)
        return int(matches[-1]) if matches else None

    def _render_bounded(self, units: list[SemanticSourceUnit]) -> str:
        header_chars = sum(len(unit.unit_id) + 10 for unit in units)
        separator_chars = max(0, len(units) - 1) * 2
        per_unit = max(
            120,
            (self.max_chars - header_chars - separator_chars) // len(units),
        )
        rendered: list[str] = []
        for index, unit in enumerate(units):
            if index == 0:
                side = "tail"
            elif index == len(units) - 1:
                side = "head"
            else:
                side = "both"
            rendered.append(
                f"[Unit {unit.unit_id}]\n{self._clip(unit.text, per_unit, side=side)}"
            )
        return "\n\n".join(rendered)[: self.max_chars]

    @staticmethod
    def _clip(value: str, max_chars: int, *, side: str) -> str:
        text = normalize_text(value)
        if len(text) <= max_chars:
            return text
        if side == "tail":
            clipped = text[-max_chars:]
            boundary = clipped.find(" ")
            return clipped[boundary + 1 :] if boundary >= 0 else clipped
        if side == "both":
            half = max(40, (max_chars - 5) // 2)
            return f"{text[:half].rstrip()} ... {text[-half:].lstrip()}"
        clipped = text[:max_chars]
        boundary = clipped.rfind(" ")
        return clipped[:boundary] if boundary > max_chars // 2 else clipped

    @staticmethod
    def _window_id(source_id: str, unit_ids: tuple[str, ...]) -> str:
        raw = "\x1f".join([source_id, *unit_ids])
        return "CW-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            if value and value not in output:
                output.append(value)
        return output


__all__ = ["CrossContextAssembler", "CrossContextWindow"]
