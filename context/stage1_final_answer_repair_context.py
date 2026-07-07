from __future__ import annotations

from typing import Any

from .context_builder import ContextBuilder, ContextPacket
from .stage1_context import Stage1ContextBuilder


STAGE1_FINAL_ANSWER_REPAIR_SYSTEM_PROMPT = """You repair one Stage1 agent output.

Use only the original question, prepared evidence, and completed tool results.
Do not request tools. Do not explain outside JSON.
Return one final_answer JSON object only."""


STAGE1_FINAL_ANSWER_REPAIR_USER_PROMPT = """Original_Question:
{question}

Prepared_Evidence:
Solver_Result:
{solver_result}

Attachment_Result:
{attachment_result}

Search_Result:
{search_result}

Completed_Tool_Results:
{tool_trace}

Previous_Invalid_Reply:
{previous_reply}

Repair_Reason:
{repair_reason}

Return exactly this JSON schema:
{{
  "type": "final_answer",
  "reasoning_steps": [
    "step 1. Use the completed evidence or tool results.",
    "step 2. Select the shortest supported answer."
  ],
  "final_answer": "short final answer only",
  "confidence": 0.0,
  "used_evidence_ids": [],
  "answer_type": "number | date | person | organization | location | title | list | short_text | boolean | unknown",
  "tool_request": null
}}

Rules:
- Return final_answer now.
- Do not request another tool.
- Do not output a tool_request JSON.
- If tool results are enough, answer from them.
- If the task requires a short answer, final_answer must not include reasoning.
- If the answer is yes/no, final_answer must be exactly "yes" or "no".
- If evidence is insufficient, still return the best short answer rather than a refusal unless the question explicitly allows refusal.
- Return JSON only."""


class Stage1FinalAnswerRepairContextBuilder(Stage1ContextBuilder):
    """Build a narrow Stage1 repair prompt that only allows a final answer."""

    def structure(self, packets: list[ContextPacket], **kwargs: Any) -> dict[str, Any]:
        structured = super().structure(packets, **kwargs)
        structured["system"] = STAGE1_FINAL_ANSWER_REPAIR_SYSTEM_PROMPT
        structured["tool_trace"] = kwargs.get("tool_trace", self.config.none_text)
        structured["previous_reply"] = kwargs.get("previous_reply", self.config.none_text)
        structured["repair_reason"] = kwargs.get("repair_reason", self.config.none_text)
        return structured

    def compress(self, structured: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        compressed = super().compress(structured, **kwargs)
        compressed["tool_trace"] = (
            self._compress_multiline_text(
                str(structured.get("tool_trace", "")),
                max_lines=40,
                max_chars=6000,
            )
            or self.config.none_text
        )
        compressed["previous_reply"] = (
            self._compress_multiline_text(
                str(structured.get("previous_reply", "")),
                max_lines=20,
                max_chars=3000,
            )
            or self.config.none_text
        )
        compressed["repair_reason"] = (
            self._compress_multiline_text(
                str(structured.get("repair_reason", "")),
                max_lines=4,
                max_chars=600,
            )
            or self.config.none_text
        )
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_FINAL_ANSWER_REPAIR_USER_PROMPT.format(
            question=compressed["question"],
            solver_result=compressed["solver_result"],
            attachment_result=compressed["attachment_result"],
            search_result=compressed["search_result"],
            tool_trace=compressed["tool_trace"],
            previous_reply=compressed["previous_reply"],
            repair_reason=compressed["repair_reason"],
        )
        return [
            {"role": "system", "content": str(compressed["system"])},
            {"role": "user", "content": user_content},
        ]


__all__ = [
    "STAGE1_FINAL_ANSWER_REPAIR_SYSTEM_PROMPT",
    "STAGE1_FINAL_ANSWER_REPAIR_USER_PROMPT",
    "Stage1FinalAnswerRepairContextBuilder",
]
