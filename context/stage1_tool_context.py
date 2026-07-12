from __future__ import annotations

from typing import Any

from .stage1_context import Stage1ContextBuilder


STAGE1_TOOL_SYSTEM_PROMPT = """You are one agent in a multi-agent reasoning network.

Solve the question independently using the provided evidence and optional tools.
You may request at most one tool per turn.
Only request a tool when it is necessary.
Your final reasoning will be evaluated step by step.
Each reasoning step must describe exactly one reasoning action.
Do not combine lookup, conversion, calculation, rounding, and final formatting in one step.
Do not place the final answer inside reasoning_steps.
Return JSON only."""


STAGE1_TOOL_USER_PROMPT = """Question:
{question}

Solver_Result:
{solver_result}

Attachment_Result:
{attachment_result}

Search_Result:
{search_result}

Attachment_Metadata:
{attachment_metadata}

Tool_Trace:
{tool_trace}

Available_Tools:
{available_tools}

Capability_Gap:
{tool_gap}

Tool_Turn_Policy:
{tool_turn_policy}

Instructions:
- If you need one tool, return exactly this JSON shape:
{{"type": "tool_request", "reasoning_step": "step N. why this tool is needed", "tool_name": "search", "tool_args": {{"input": "query", "mode": "text"}}}}
- Replace tool_name and tool_args with the correct available tool when needed.
- Do not invent a tool name that is not listed in Available_Tools.
- If Capability_Gap lists a missing capability, do not repeat an unsupported request.
- Follow Tool_Turn_Policy. When it requests final_answer, do not request another tool.
- If you can answer, return exactly this JSON shape:
{{"type": "final_answer", "reasoning_steps": ["step 1. Identify the required value from evidence.", "step 2. Convert units or normalize values if needed.", "step 3. Perform one calculation or comparison.", "step 4. Apply rounding if required.", "step 5. Format the computed value according to the question."], "final_answer": "short final answer only", "confidence": 0.0, "used_evidence_ids": ["E1"], "answer_type": "number | date | person | organization | location | title | list | short_text | boolean | unknown", "tool_request": null}}
- Return JSON only.
- Use 3 to 6 reasoning_steps; each must start with exactly "step N."
- Each step must be one short sentence and exactly one action.
- If one step needs multiple actions, split it into multiple steps.
- Keep final_answer short and separate from reasoning_steps.
- Never include "Final Answer", "Answer", "boxed", or conclusion text inside reasoning_steps.
- Split lookup, conversion, calculation, rounding, and formatting into separate steps."""


class Stage1ToolContextBuilder(Stage1ContextBuilder):
    """Build Stage1 tool-use messages with accumulated tool trace."""

    def structure(self, packets: list[Any], **kwargs: Any) -> dict[str, Any]:
        structured = super().structure(packets, **kwargs)
        structured["system"] = STAGE1_TOOL_SYSTEM_PROMPT
        structured["tool_trace"] = kwargs.get("tool_trace", self.config.none_text)
        structured["available_tools"] = kwargs.get(
            "available_tools",
            self.config.none_text,
        )
        structured["tool_gap"] = kwargs.get("tool_gap", self.config.none_text)
        structured["tool_turn_policy"] = kwargs.get(
            "tool_turn_policy",
            self.config.none_text,
        )
        structured["attachment_metadata"] = self._format_attachment_metadata(
            kwargs.get("attachment")
        )
        return structured

    def compress(self, structured: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        compressed = super().compress(structured, **kwargs)
        compressed["tool_trace"] = (
            self._compress_multiline_text(
                str(structured.get("tool_trace", "")),
                max_lines=self.config.max_context_lines,
                max_chars=self.config.max_context_chars,
            )
            or self.config.none_text
        )
        compressed["attachment_metadata"] = (
            self._compress_multiline_text(
                str(structured.get("attachment_metadata", "")),
                max_lines=12,
                max_chars=1200,
            )
            or self.config.none_text
        )
        compressed["available_tools"] = (
            self._compress_multiline_text(
                str(structured.get("available_tools", "")),
                max_lines=30,
                max_chars=5000,
            )
            or self.config.none_text
        )
        compressed["tool_gap"] = (
            self._compress_multiline_text(
                str(structured.get("tool_gap", "")),
                max_lines=12,
                max_chars=1800,
            )
            or self.config.none_text
        )
        compressed["tool_turn_policy"] = (
            self._compress_multiline_text(
                str(structured.get("tool_turn_policy", "")),
                max_lines=12,
                max_chars=1600,
            )
            or self.config.none_text
        )
        return self._apply_context_budget(compressed)

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_TOOL_USER_PROMPT.format(
            question=compressed["question"],
            solver_result=compressed["solver_result"],
            attachment_result=compressed["attachment_result"],
            search_result=compressed["search_result"],
            attachment_metadata=compressed["attachment_metadata"],
            tool_trace=compressed["tool_trace"],
            available_tools=compressed["available_tools"],
            tool_gap=compressed["tool_gap"],
            tool_turn_policy=compressed["tool_turn_policy"],
        )
        return [
            {"role": "system", "content": str(compressed["system"])},
            {"role": "user", "content": user_content},
        ]

    def _format_attachment_metadata(self, attachment: Any) -> str:
        if not isinstance(attachment, dict) or not attachment:
            return self.config.none_text

        keys = ("file_path", "path", "file_name", "extension")
        lines = []
        for key in keys:
            value = attachment.get(key)
            if value:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) if lines else self.config.none_text


__all__ = [
    "STAGE1_TOOL_SYSTEM_PROMPT",
    "STAGE1_TOOL_USER_PROMPT",
    "Stage1ToolContextBuilder",
]
