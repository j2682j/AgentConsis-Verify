from __future__ import annotations

import re
from collections import Counter
from typing import Any


def extract_reasoning_steps(reasoning: str) -> list[tuple[int, str]]:
    """
    從 reasoning 文字中抽取 step N. 格式的推理步驟。

    Args:
        - reasoning: Agent 輸出的 reasoning 文字。

    Returns:
        - list[tuple[int, str]]: 每個 reasoning step 的編號與內容。
        - []: 沒有符合 step 格式時回傳空清單。
    """
    text = str(reasoning or "").strip()
    if not text:
        return []

    step_pattern = (
        r"(?ims)(?:^|\n)\s*(?:[-*]\s*)?step\s*(\d+)\s*[\.\):：-]?\s*(.*?)"
        r"(?=(?:\n\s*(?:[-*]\s*)?step\s*\d+\s*[\.\):：-]?\s*)|\Z)"
    )
    steps: list[tuple[int, str]] = []
    for match in re.finditer(step_pattern, text):
        step_no = int(match.group(1))
        content = " ".join(match.group(2).strip().split())
        if content:
            steps.append((step_no, content))
    return steps


def format_reasoning_steps(reasoning: str) -> str:
    """
    將 reasoning step 重新格式化成標準 step N. 格式。

    Args:
        - reasoning: 原始 reasoning 文字。

    Returns:
        - str: 標準化後的 reasoning steps；若無法解析 steps 則回傳原文字。
    """
    steps = extract_reasoning_steps(reasoning)
    if not steps:
        return str(reasoning or "").strip()
    return "\n".join(f"step {index}. {text}" for index, text in steps)


def compress_reasoning(runs: list[Any]) -> str:
    """
    從同答案群的多次 Stage1 runs 中選出代表性 reasoning。

    Args:
        - runs: 同一 Agent 或同一答案群的 EachAgentReply 清單。

    Returns:
        - str: 壓縮後的 reasoning，優先使用可解析 step 的第一份 reasoning。
        - str: 若多份 reasoning 不同，回傳去重後的條列摘要。
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


__all__ = ["compress_reasoning", "extract_reasoning_steps", "format_reasoning_steps"]
