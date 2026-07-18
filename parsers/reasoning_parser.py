from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Pattern

from .reasoning_step_atomizer import ReasoningStepAtomizer


_EXPLICIT_STEP_PATTERN = re.compile(
    r"(?is)"
    r"(?:^|[\n\r]|(?<=\s))"
    r"(?P<prefix>[^\w\n\r]*)"
    r"(?:\*\*)?\s*step\s*(?P<index>\d{1,3})\s*"
    r"(?P<suffix>[\.\):：\-、]?)(?:\*\*)?\s*"
)

_NUMBERED_STEP_PATTERN = re.compile(
    r"(?im)"
    r"(?P<prefix>^\s*(?:[-*]\s*)?)"
    r"(?P<index>\d{1,3})"
    r"(?P<suffix>[\.\):：、])"
    r"(?!\d)\s+"
)

_MARKDOWN_RULE_PATTERN = re.compile(r"(?m)^\s*-{3,}\s*$")
_MARKDOWN_HEADING_PATTERN = re.compile(r"(?im)^\s*#{2,6}\s+(.+?)\s*$")
_FINAL_ANSWER_TAIL_PATTERN = re.compile(
    r"(?is)(?:^|[\s#>\*\-_`])final[_\s-]*answer\s*[:：]?"
)
_PLAIN_FINAL_ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*final\s+answer\s*(?:\*\*)?\s*[:：]"
)


@dataclass(frozen=True)
class _StepMarker:
    index: int
    start: int
    content_start: int


class ReasoningParseQuality(str, Enum):
    VALID = "valid"
    REPAIRED = "repaired"
    UNRELIABLE = "unreliable"


@dataclass
class ReasoningParseDiagnostics:
    source_format: str = ""
    explicit_marker_count: int = 0
    original_step_count: int = 0
    canonical_step_count: int = 0
    final_answer_marker_found: bool = False
    final_answer_removed: bool = False
    final_answer_leak_step_indices: list[int] = field(default_factory=list)
    compound_step_indices: list[int] = field(default_factory=list)
    atomized_step_indices: list[int] = field(default_factory=list)
    renumbered: bool = False
    repair_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "explicit_marker_count": self.explicit_marker_count,
            "original_step_count": self.original_step_count,
            "canonical_step_count": self.canonical_step_count,
            "final_answer_marker_found": self.final_answer_marker_found,
            "final_answer_removed": self.final_answer_removed,
            "final_answer_leak_step_indices": list(self.final_answer_leak_step_indices),
            "compound_step_indices": list(self.compound_step_indices),
            "atomized_step_indices": list(self.atomized_step_indices),
            "renumbered": self.renumbered,
            "repair_actions": list(self.repair_actions),
            "warnings": list(self.warnings),
        }


@dataclass
class ReasoningParseResult:
    steps: list[tuple[int, str]]
    reasoning_text: str
    extracted_final_answer: str
    quality_status: ReasoningParseQuality
    versa_eligible: bool
    diagnostics: ReasoningParseDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [[index, text] for index, text in self.steps],
            "reasoning_text": self.reasoning_text,
            "extracted_final_answer": self.extracted_final_answer,
            "quality_status": self.quality_status.value,
            "versa_eligible": self.versa_eligible,
            "diagnostics": self.diagnostics.to_dict(),
        }


def extract_reasoning_steps(reasoning: str) -> list[tuple[int, str]]:
    """
    將模型輸出的 reasoning 切成 VersaPRM 可逐步評分的 step list。

    Args:
     - reasoning: Agent 輸出的 reasoning 文字。

    Returns:
     - list[tuple[int, str]]: 每個 reasoning step 的編號與文字。
     - []: 沒有可解析內容時回傳空清單。
    """
    return _extract_reasoning_steps_raw(reasoning, allow_paragraph_fallback=False)


def prepare_reasoning_for_verifier(
    reasoning: str,
    *,
    final_answer: str = "",
    structured_steps: list[str] | None = None,
) -> ReasoningParseResult:
    diagnostics = ReasoningParseDiagnostics()
    extracted_answer = ""
    if structured_steps:
        diagnostics.source_format = "structured_list"
        raw_steps: list[tuple[int, str]] = []
        answer_boundary_seen = False
        for index, item in enumerate(structured_steps, start=1):
            before, extracted, found = split_reasoning_and_final_answer(str(item or ""))
            if found:
                diagnostics.final_answer_marker_found = True
                diagnostics.final_answer_removed = True
                diagnostics.final_answer_leak_step_indices.append(index)
                diagnostics.repair_actions.append("strip_final_answer_tail")
                extracted_answer = extracted_answer or extracted
                answer_boundary_seen = True
            if answer_boundary_seen and not before:
                continue
            parsed = _extract_reasoning_steps_raw(before, allow_paragraph_fallback=False)
            if parsed:
                raw_steps.extend(parsed)
            elif normalized_text := " ".join(before.strip().split()):
                raw_steps.append((index, normalized_text))
    else:
        text, extracted_answer, found = split_reasoning_and_final_answer(str(reasoning or ""))
        diagnostics.final_answer_marker_found = found
        diagnostics.final_answer_removed = found
        if found:
            diagnostics.repair_actions.append("strip_final_answer_tail")
        diagnostics.source_format = _source_format(text)
        raw_steps = _extract_reasoning_steps_raw(text, allow_paragraph_fallback=True)

    diagnostics.original_step_count = len(raw_steps)
    diagnostics.explicit_marker_count = len(raw_steps) if diagnostics.source_format in {
        "structured_list", "numbered_text"
    } else 0
    atomized = ReasoningStepAtomizer().atomize(raw_steps)
    canonical = _dedupe_adjacent_steps(atomized.steps)
    diagnostics.compound_step_indices = list(atomized.compound_step_indices)
    diagnostics.atomized_step_indices = list(atomized.atomized_step_indices)
    if atomized.atomized_step_indices:
        diagnostics.repair_actions.append("atomize_compound_step")
    original_indexes = [index for index, _ in raw_steps]
    diagnostics.renumbered = original_indexes != list(range(1, len(raw_steps) + 1))
    if diagnostics.renumbered:
        diagnostics.repair_actions.append("renumber_steps")
    diagnostics.canonical_step_count = len(canonical)

    if final_answer and extracted_answer and _normalized_answer(final_answer) != _normalized_answer(extracted_answer):
        diagnostics.warnings.append("final_answer_conflict")
    if not canonical:
        quality = ReasoningParseQuality.UNRELIABLE
        diagnostics.warnings.append("no_reasoning_steps")
    elif atomized.compound_step_indices:
        quality = ReasoningParseQuality.UNRELIABLE
        diagnostics.warnings.append("unresolved_compound_steps")
    elif diagnostics.repair_actions:
        quality = ReasoningParseQuality.REPAIRED
    else:
        quality = ReasoningParseQuality.VALID
    reasoning_text = "\n".join(f"step {index}. {text}" for index, text in canonical)
    return ReasoningParseResult(
        steps=canonical,
        reasoning_text=reasoning_text,
        extracted_final_answer=extracted_answer,
        quality_status=quality,
        versa_eligible=quality != ReasoningParseQuality.UNRELIABLE,
        diagnostics=diagnostics,
    )


def split_reasoning_and_final_answer(text: str) -> tuple[str, str, bool]:
    source = str(text or "").strip()
    match = _FINAL_ANSWER_TAIL_PATTERN.search(source)
    boxed = re.search(r"(?is)\\boxed\{([^{}]+)\}\s*$", source)
    if boxed and (match is None or boxed.start() < match.start()):
        return source[: boxed.start()].strip(), boxed.group(1).strip(), True
    if not match:
        return source, "", False
    before = _strip_trailing_marker_noise(source[: match.start()].strip())
    after = source[match.end() :].strip()
    after = re.sub(r"^[\s:=\-*`#>]+", "", after).strip()
    after = re.split(r"\n\s*(?:#{1,6}\s+|step\s*\d+[.:])", after, maxsplit=1, flags=re.IGNORECASE)[0]
    return before, after.strip(" \"'`"), True


def _extract_reasoning_steps_raw(
    reasoning: str,
    *,
    allow_paragraph_fallback: bool,
) -> list[tuple[int, str]]:
    text = str(reasoning or "").strip()
    if not text:
        return []

    text = _normalize_reasoning_text(text)
    steps = _extract_by_pattern(text, _EXPLICIT_STEP_PATTERN)
    if steps:
        return steps

    steps = _extract_by_pattern(text, _NUMBERED_STEP_PATTERN, require_sequence=True)
    if steps:
        return steps

    steps = _extract_markdown_heading_steps(text)
    if steps:
        return steps

    return _extract_paragraph_steps(text) if allow_paragraph_fallback else []


def _source_format(text: str) -> str:
    if _EXPLICIT_STEP_PATTERN.search(text) or _NUMBERED_STEP_PATTERN.search(text):
        return "numbered_text"
    if _MARKDOWN_HEADING_PATTERN.search(text):
        return "markdown_heading"
    return "paragraph_fallback"


def _dedupe_adjacent_steps(steps: list[tuple[int, str]]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    previous = ""
    for _, text in steps:
        cleaned = " ".join(str(text or "").split()).strip()
        key = cleaned.casefold()
        if not cleaned or key == previous:
            continue
        result.append((len(result) + 1, cleaned))
        previous = key
    return result


def _normalized_answer(value: str) -> str:
    return " ".join(str(value or "").casefold().split()).strip(" .")


def format_reasoning_steps(reasoning: str) -> str:
    """
    將 reasoning 正規化為 step N. 格式。

    Args:
     - reasoning: 原始 reasoning 文字。

    Returns:
     - str: 正規化後的 reasoning；若解析不到 step，回傳原文。
    """
    result = prepare_reasoning_for_verifier(reasoning)
    if not result.steps:
        return str(reasoning or "").strip()
    return result.reasoning_text


def compress_reasoning(runs: list[Any]) -> str:
    """
    從多次 Stage1 runs 中整理代表性的 reasoning。

    Args:
     - runs: Agent 的 EachAgentReply 清單。

    Returns:
     - str: 若第一筆 reasoning 有 steps，回傳正規化 steps；否則回傳去重後文字。
    """
    reasonings = [run.reasoning.strip() for run in runs if getattr(run, "reasoning", "").strip()]
    if not reasonings:
        return ""

    first_steps = prepare_reasoning_for_verifier(reasonings[0]).steps
    if first_steps:
        return "\n".join(f"step {index}. {text}" for index, text in first_steps)

    counts = Counter(reasonings)
    if len(counts) == 1:
        return reasonings[0]

    selected = []
    seen = set()
    for reasoning in reasonings:
        if reasoning in seen:
            continue
        selected.append(reasoning)
        seen.add(reasoning)
    return "\n".join(f"- {item}" for item in selected)


def _normalize_reasoning_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MARKDOWN_RULE_PATTERN.sub("\n", text)
    return text.strip()


def _extract_by_pattern(
    text: str,
    pattern: Pattern[str],
    *,
    require_sequence: bool = False,
) -> list[tuple[int, str]]:
    markers = _find_markers(text, pattern)
    if require_sequence and not _has_ordered_pair(markers):
        return []
    markers = _dedupe_markers(markers)
    if not markers:
        return []

    steps: list[tuple[int, str]] = []
    for marker_index, marker in enumerate(markers):
        content_end = (
            markers[marker_index + 1].start
            if marker_index + 1 < len(markers)
            else len(text)
        )
        raw_content = text[marker.content_start:content_end]
        if marker_index == len(markers) - 1:
            raw_content = _strip_final_answer_tail(raw_content)
        content = _clean_step_text(raw_content)
        if content:
            steps.append((marker.index, content))
    return steps


def _find_markers(text: str, pattern: Pattern[str]) -> list[_StepMarker]:
    markers: list[_StepMarker] = []
    for match in pattern.finditer(text):
        try:
            index = int(match.group("index"))
        except (TypeError, ValueError):
            continue
        if index <= 0:
            continue
        start = match.start()
        prefix = match.groupdict().get("prefix")
        if prefix:
            start = match.start("prefix")
        markers.append(
            _StepMarker(
                index=index,
                start=start,
                content_start=match.end(),
            )
        )
    return markers


def _dedupe_markers(markers: list[_StepMarker]) -> list[_StepMarker]:
    deduped: list[_StepMarker] = []
    seen_starts: set[int] = set()
    for marker in sorted(markers, key=lambda item: item.start):
        if marker.start in seen_starts:
            continue
        deduped.append(marker)
        seen_starts.add(marker.start)
    return deduped


def _has_ordered_pair(markers: list[_StepMarker]) -> bool:
    indexes = [marker.index for marker in markers]
    return any(next_index == index + 1 for index, next_index in zip(indexes, indexes[1:]))


def _clean_step_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?m)^\s*[-*]\s*$", " ", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip(" -:")


def _extract_markdown_heading_steps(text: str) -> list[tuple[int, str]]:
    markers = list(_MARKDOWN_HEADING_PATTERN.finditer(text))
    if not markers:
        return []

    steps: list[tuple[int, str]] = []
    preface = text[: markers[0].start()].strip(" -\n")
    if preface:
        content = _clean_step_text(preface)
        if content:
            steps.append((1, content))

    for marker_index, marker in enumerate(markers):
        content_start = marker.start()
        content_end = (
            markers[marker_index + 1].start()
            if marker_index + 1 < len(markers)
            else len(text)
        )
        raw_content = text[content_start:content_end]
        if marker_index == len(markers) - 1:
            raw_content = _strip_final_answer_tail(raw_content)
        content = _clean_step_text(raw_content.strip(" -\n"))
        if content:
            steps.append((len(steps) + 1, content))
    return steps


def _extract_paragraph_steps(text: str) -> list[tuple[int, str]]:
    text = _strip_final_answer_tail(text)
    text = _PLAIN_FINAL_ANSWER_LINE_PATTERN.split(text, maxsplit=1)[0].strip()
    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        content = _clean_step_text(paragraphs[0])
        return [(1, content)] if content else []

    steps: list[tuple[int, str]] = []
    buffer: list[str] = []
    for paragraph in paragraphs:
        buffer.append(paragraph)
        joined = "\n\n".join(buffer)
        if len(joined) >= 260 or paragraph.endswith((".", ":", "?", "!")):
            content = _clean_step_text(joined)
            if content:
                steps.append((len(steps) + 1, content))
            buffer = []
    if buffer:
        content = _clean_step_text("\n\n".join(buffer))
        if content:
            steps.append((len(steps) + 1, content))
    return steps


def _strip_final_answer_tail(text: str) -> str:
    match = _FINAL_ANSWER_TAIL_PATTERN.search(text)
    if not match:
        return text
    tail = text[: match.start()].strip()
    return _strip_trailing_marker_noise(tail)


def _strip_trailing_marker_noise(text: str) -> str:
    text = text.rstrip()
    preserved_punctuation = set(".!?)]}$")
    while text:
        char = text[-1]
        if char.isspace() or char in "#>*_`-":
            text = text[:-1].rstrip()
            continue
        if not char.isalnum() and char not in preserved_punctuation:
            text = text[:-1].rstrip()
            continue
        break
    return text.strip()


__all__ = [
    "ReasoningParseDiagnostics",
    "ReasoningParseQuality",
    "ReasoningParseResult",
    "compress_reasoning",
    "extract_reasoning_steps",
    "format_reasoning_steps",
    "prepare_reasoning_for_verifier",
    "split_reasoning_and_final_answer",
]
