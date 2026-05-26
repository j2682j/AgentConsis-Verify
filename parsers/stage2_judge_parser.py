from __future__ import annotations

import re
from typing import Any

from parsers.json_parse import try_parse_json
from parsers.reasoning_parser import extract_reasoning_steps


class Stage2JudgeParser:
    """
    解析 Stage2 Judge 回覆，抽取每個 reasoning step 的分數並計算平均 judge score。

    Args:
        - 無。

    Returns:
        - Stage2JudgeParser: 可解析 JSON、文字或 markdown table 分數的 parser。
    """

    def parse(
        self,
        raw_reply: str,
        target_reasoning: str,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        解析 judge raw reply，取得 step_scores 與 -1 到 1 之間的平均 judge_score。

        Args:
            - raw_reply: Judge Agent 回傳的原始文字。
            - target_reasoning: 被評分的 reasoning，用於 fallback 建立 step scores。

        Returns:
            - list[dict[str, Any]]: 每個 step 的 score。
            - float: 所有 step scores 的平均分數。
        """
        parsed = try_parse_json(raw_reply)
        step_scores = self._coerce_step_scores(parsed.get("step_scores") if isinstance(parsed, dict) else None)
        if not step_scores:
            step_scores = self._parse_step_scores_from_text(raw_reply)
        if not step_scores:
            fallback_score = self._parse_judge_score(raw_reply)
            reasoning_steps = extract_reasoning_steps(target_reasoning)
            if reasoning_steps:
                step_scores = [{"step": step_no, "score": fallback_score} for step_no, _ in reasoning_steps]
            else:
                step_scores = [{"step": 1, "score": fallback_score}]

        judge_score = sum(item["score"] for item in step_scores) / len(step_scores)
        return step_scores, max(-1.0, min(1.0, judge_score))

    def _parse_judge_score(self, raw_reply: str) -> float:
        """
        從 JSON 或文字中抽取單一 fallback judge score。

        Args:
            - raw_reply: Judge Agent 回傳的原始文字。

        Returns:
            - float: clamp 到 -1 到 1 的分數。
        """
        parsed = try_parse_json(raw_reply)
        value: Any = parsed.get("judge_score") if isinstance(parsed, dict) else None
        if value is None:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_reply or "")
            value = match.group(0) if match else 0
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        return max(-1.0, min(1.0, score))

    def _coerce_step_scores(self, value: Any) -> list[dict[str, Any]]:
        """
        將 JSON step_scores 欄位轉成標準 step-score dict 清單。

        Args:
            - value: JSON 中的 step_scores 欄位。

        Returns:
            - list[dict[str, Any]]: 標準化後的 step scores。
        """
        if not isinstance(value, list):
            return []

        scores: list[dict[str, Any]] = []
        for index, item in enumerate(value, 1):
            if isinstance(item, dict):
                step_value = item.get("step", index)
                score_value = item.get("score", item.get("judge_score", 0))
            else:
                step_value = index
                score_value = item
            try:
                step_no = int(step_value)
            except (TypeError, ValueError):
                step_no = index
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                score = 0.0
            scores.append({"step": step_no, "score": max(-1.0, min(1.0, score))})
        return scores

    def _parse_step_scores_from_text(self, raw_reply: str) -> list[dict[str, Any]]:
        """
        從純文字或 fallback markdown table 中抽取 step scores。

        Args:
            - raw_reply: Judge Agent 回傳的原始文字。

        Returns:
            - list[dict[str, Any]]: 從文字中解析出的 step scores。
        """
        table_scores = self._parse_step_scores_from_markdown_table(raw_reply)
        if table_scores:
            return table_scores

        scores: list[dict[str, Any]] = []
        for match in re.finditer(
            r"(?i)step\s*(\d+)\D{0,20}(-?\d+(?:\.\d+)?)",
            raw_reply or "",
        ):
            scores.append(
                {
                    "step": int(match.group(1)),
                    "score": max(-1.0, min(1.0, float(match.group(2)))),
                }
            )
        return scores

    def _parse_step_scores_from_markdown_table(self, raw_reply: str) -> list[dict[str, Any]]:
        """
        從 markdown table 格式中抽取 step scores。

        Args:
            - raw_reply: Judge Agent 回傳的原始文字。

        Returns:
            - list[dict[str, Any]]: 依 step 編號排序後的 step scores。
        """
        scores: dict[int, float] = {}
        for line in str(raw_reply or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells or not re.fullmatch(r"\d+", cells[0]):
                continue

            step_no = int(cells[0])
            score_value: float | None = None
            candidates = []
            if len(cells) >= 6:
                candidates.append(cells[-2])
            candidates.extend(reversed(cells[1:]))
            for candidate in candidates:
                if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate):
                    score_value = float(candidate)
                    break
            if score_value is None:
                continue
            scores.setdefault(step_no, max(-1.0, min(1.0, score_value)))

        return [{"step": step, "score": score} for step, score in sorted(scores.items())]


__all__ = ["Stage2JudgeParser"]
