from __future__ import annotations

from dataclasses import dataclass, field
import re

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class AtomizationResult:
    steps: list[tuple[int, str]]
    compound_step_indices: list[int] = field(default_factory=list)
    atomized_step_indices: list[int] = field(default_factory=list)


class ReasoningStepAtomizer:
    """Split only structurally clear multi-action reasoning steps."""

    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    _EXPLICIT_LIST = re.compile(r"\s+(?=(?:\(?[a-z0-9]+\)|\d+[.)])\s+)", re.IGNORECASE)
    _ACTION_START = re.compile(
        r"^(?:identify|find|extract|read|look up|compare|calculate|compute|convert|"
        r"divide|multiply|subtract|add|apply|round|check|verify|determine|use|"
        r"select|count|derive|conclude|format|return|infer|match|combine)\b",
        re.IGNORECASE,
    )
    _MULTI_ACTION = re.compile(
        r"\b(?:identify|find|extract|compare|calculate|compute|convert|divide|"
        r"multiply|subtract|add|apply|round|check|verify|determine|conclude|infer)\b",
        re.IGNORECASE,
    )

    def atomize(self, steps: list[tuple[int, str]]) -> AtomizationResult:
        output: list[tuple[int, str]] = []
        compound: list[int] = []
        atomized: list[int] = []
        for original_index, raw_text in steps:
            text = normalize_text(raw_text)
            parts = self._safe_parts(text)
            if len(parts) > 1:
                atomized.append(original_index)
                output.extend((0, part) for part in parts)
            else:
                output.append((0, text))
                if self._is_unresolved_compound(text):
                    compound.append(original_index)
        renumbered = [
            (index, text)
            for index, (_, text) in enumerate(output, start=1)
            if normalize_text(text)
        ]
        return AtomizationResult(
            steps=renumbered,
            compound_step_indices=compound,
            atomized_step_indices=atomized,
        )

    def _safe_parts(self, text: str) -> list[str]:
        sentence_parts = [
            normalize_text(item)
            for item in self._SENTENCE_BOUNDARY.split(text)
            if normalize_text(item)
        ]
        if len(sentence_parts) > 1:
            return sentence_parts

        semicolon_parts = [
            normalize_text(item)
            for item in re.split(r"\s*;\s*", text)
            if normalize_text(item)
        ]
        if len(semicolon_parts) > 1 and all(self._has_action(item) for item in semicolon_parts):
            return semicolon_parts

        listed = [
            self._strip_list_marker(item)
            for item in self._EXPLICIT_LIST.split(text)
            if normalize_text(item)
        ]
        if len(listed) > 1 and all(self._has_action(item) for item in listed):
            return listed

        comma_parts = [normalize_text(item) for item in re.split(r",\s+(?:and\s+)?", text)]
        if (
            2 <= len(comma_parts) <= 4
            and all(self._ACTION_START.search(item) for item in comma_parts)
        ):
            return comma_parts
        return [text] if text else []

    def _is_unresolved_compound(self, text: str) -> bool:
        actions = self._MULTI_ACTION.findall(text)
        return len(actions) >= 3 and bool(re.search(r"\b(?:and|then)\b|,", text, re.IGNORECASE))

    def _has_action(self, text: str) -> bool:
        return bool(self._MULTI_ACTION.search(text))

    @staticmethod
    def _strip_list_marker(text: str) -> str:
        return normalize_text(re.sub(r"^(?:\(?[a-z0-9]+\)|\d+[.)])\s*", "", text, flags=re.IGNORECASE))


__all__ = ["AtomizationResult", "ReasoningStepAtomizer"]
