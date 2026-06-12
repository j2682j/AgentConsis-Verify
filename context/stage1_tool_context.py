from __future__ import annotations

from typing import Any

from .stage1_context import Stage1ContextBuilder


STAGE1_TOOL_SYSTEM_PROMPT = """You are one agent in a multi-agent reasoning network.

Solve the question independently using the provided evidence and optional tools.
You may request at most one tool per turn.
Only request a tool when it is necessary.
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

Available tools:
- search: use for external factual lookup. tool_args: {{"input": "query", "mode": "text"}}
- python_calculator: use for arithmetic or deterministic calculation. tool_args: {{"expression": "math expression"}}
- deterministic_solver: use for closed-world deterministic tasks, tables, strings, lists, and unit conversion. tool_args: {{"input": "question"}}
- attachment_reader: use to read a task attachment when attachment metadata is present. tool_args: {{"question": "question", "file_path": "path"}}

Instructions:
- If you need one tool, return exactly this JSON shape:
{{"type": "tool_request", "reasoning_step": "step N. why this tool is needed", "tool_name": "search", "tool_args": {{"input": "query", "mode": "text"}}}}
- Replace tool_name and tool_args with the correct available tool when needed.
- If you can answer, return exactly this JSON shape:
{{"type": "final_answer", "reasoning": "step 1. ...\\nstep 2. ...", "final_answer": "short final answer only"}}
- Reasoning must use explicit numbered steps.
- Do not include markdown or text outside JSON."""


class Stage1ToolContextBuilder(Stage1ContextBuilder):
    """Build Stage1 tool-use messages with accumulated tool trace."""

    def structure(self, packets: list[Any], **kwargs: Any) -> dict[str, Any]:
        structured = super().structure(packets, **kwargs)
        structured["system"] = STAGE1_TOOL_SYSTEM_PROMPT
        structured["tool_trace"] = kwargs.get("tool_trace", self.config.none_text)
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
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_TOOL_USER_PROMPT.format(
            question=compressed["question"],
            solver_result=compressed["solver_result"],
            attachment_result=compressed["attachment_result"],
            search_result=compressed["search_result"],
            attachment_metadata=compressed["attachment_metadata"],
            tool_trace=compressed["tool_trace"],
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
