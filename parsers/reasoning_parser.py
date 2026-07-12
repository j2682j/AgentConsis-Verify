from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Pattern


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
_FINAL_ANSWER_TAIL_PATTERN = re.compile(
    r"(?is)(?:^|[\s#>\*\-_`])final[_\s-]*answer\s*[:：]?"
)


@dataclass(frozen=True)
class _StepMarker:
    index: int
    start: int
    content_start: int


def extract_reasoning_steps(reasoning: str) -> list[tuple[int, str]]:
    """
    將模型輸出的 reasoning 切成 VersaPRM 可逐步評分的 step list。

    Args:
     - reasoning: Agent 輸出的 reasoning 文字。

    Returns:
     - list[tuple[int, str]]: 每個 reasoning step 的編號與文字。
     - []: 沒有可解析內容時回傳空清單。
    """
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
    return []


def format_reasoning_steps(reasoning: str) -> str:
    """
    將 reasoning 正規化為 step N. 格式。

    Args:
     - reasoning: 原始 reasoning 文字。

    Returns:
     - str: 正規化後的 reasoning；若解析不到 step，回傳原文。
    """
    steps = extract_reasoning_steps(reasoning)
    if not steps:
        return str(reasoning or "").strip()
    return "\n".join(f"step {index}. {text}" for index, text in steps)


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

    first_steps = extract_reasoning_steps(reasonings[0])
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


__all__ = ["compress_reasoning", "extract_reasoning_steps", "format_reasoning_steps"]
