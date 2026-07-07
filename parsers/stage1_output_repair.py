from __future__ import annotations

import re
from typing import Any

from .json_parse import try_parse_json
from score.answer_validator import AnswerValidator
from utils.network_utils import normalize_text


class Stage1OutputRepairer:
    """
    Repair Stage1 output structure while preserving the agent's original content.
    """

    def __init__(self, validator: AnswerValidator | None = None) -> None:
        self.validator = validator or AnswerValidator()

    def repair(self, raw_reply: str) -> tuple[dict[str, Any], list[str]]:
        raw_reply = str(raw_reply or "").strip()
        actions: list[str] = []

        parsed = try_parse_json(raw_reply)
        if isinstance(parsed, dict):
            return dict(parsed), actions

        stripped = self._strip_markdown_code_block(raw_reply)
        if stripped != raw_reply:
            actions.append("strip_markdown_code_block")
            parsed = try_parse_json(stripped)
            if isinstance(parsed, dict):
                return dict(parsed), actions

        data: dict[str, Any] = {}
        reasoning_steps = self._extract_reasoning_steps(raw_reply)
        if reasoning_steps:
            data["reasoning_steps"] = reasoning_steps
            data["reasoning"] = "\n".join(reasoning_steps)
            actions.append("extract_reasoning_steps")

        final_answer = self._extract_final_answer(raw_reply)
        if final_answer:
            data["final_answer"] = final_answer
            actions.append("extract_final_answer_label")

        evidence_ids = self._extract_evidence_ids(raw_reply)
        if evidence_ids:
            data["used_evidence_ids"] = evidence_ids
            actions.append("extract_used_evidence_ids")

        tool_request = self._extract_tool_request(raw_reply)
        if tool_request:
            data.update(tool_request)
            actions.append("extract_tool_request_json")

        if not data:
            lines = self._nonempty_lines(raw_reply)
            if lines:
                candidate = self.validator.clean(lines[-1])
                if candidate:
                    data["final_answer"] = candidate
                    data["reasoning"] = "\n".join(lines[:-1])
                    data["reasoning_steps"] = self._extract_reasoning_steps(data["reasoning"])
                    actions.append("fallback_last_line_as_answer")

        return data, actions

    def _strip_markdown_code_block(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_reasoning_steps(self, text: str) -> list[str]:
        matches = list(
            re.finditer(
                r"(step\s*\d+\s*[.:]\s*.*?)(?=\n\s*step\s*\d+\s*[.:]|\n\s*FINAL[_ ]?ANSWER\s*[:=]|$)",
                text or "",
                re.IGNORECASE | re.DOTALL,
            )
        )
        steps = [self._normalize_step(match.group(1), index) for index, match in enumerate(matches, start=1)]
        return [step for step in steps if step]

    def _normalize_step(self, text: str, index: int) -> str:
        text = normalize_text(text)
        match = re.match(r"step\s*(\d+)\s*[.:]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return f"step {index}. {text}".strip()
        number = match.group(1)
        body = normalize_text(match.group(2))
        return f"step {number}. {body}".strip()

    def _extract_final_answer(self, text: str) -> str:
        patterns = [
            r'"final_answer"\s*:\s*"([^"]*)"',
            r"'final_answer'\s*:\s*'([^']*)'",
            r"(?:^|\n)\s*(?:\*\*)?\s*FINAL[_ ]?ANSWER\s*(?:\*\*)?\s*[:=]\s*(.+)",
            r"(?:^|\n)\s*(?:\*\*)?\s*ANSWER\s*(?:\*\*)?\s*[:=]\s*(.+)",
            r"\\boxed\{([^{}]+)\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", re.IGNORECASE)
            if not match:
                continue
            candidate = self._repair_labeled_answer(match.group(1))
            if candidate:
                return candidate
        return ""

    def _repair_labeled_answer(self, text: str) -> str:
        candidate = self.validator.clean(text)
        if not candidate:
            return ""
        candidate = re.sub(r"\s*```.*$", "", candidate, flags=re.DOTALL).strip()
        candidate = re.sub(
            r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|would be)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        boolean_match = re.match(r"^(yes|no)\b", candidate, re.IGNORECASE)
        if boolean_match:
            return boolean_match.group(1).lower()
        if self.validator.is_too_verbose(candidate):
            first_line = candidate.splitlines()[0].strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", first_line)[0].strip()
            for option in (first_line, first_sentence):
                option = self.validator.clean(option)
                if option and not self.validator.is_too_verbose(option):
                    return option
        return candidate

    def _extract_evidence_ids(self, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for match in re.finditer(r"\bE(\d+)\b", text or "", re.IGNORECASE):
            evidence_id = f"E{int(match.group(1))}"
            if evidence_id not in seen:
                seen.add(evidence_id)
                result.append(evidence_id)
        return result

    def _extract_tool_request(self, text: str) -> dict[str, Any]:
        parsed = try_parse_json(text)
        if not isinstance(parsed, dict):
            return {}
        reply_type = str(parsed.get("type", "") or "").strip().lower()
        if reply_type != "tool_request":
            return {}
        return {
            "type": "tool_request",
            "tool_name": str(parsed.get("tool_name", "") or "").strip(),
            "tool_args": parsed.get("tool_args") if isinstance(parsed.get("tool_args"), dict) else {},
            "reasoning_step": str(parsed.get("reasoning_step", "") or "").strip(),
        }

    def _nonempty_lines(self, text: str) -> list[str]:
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]


__all__ = ["Stage1OutputRepairer"]
