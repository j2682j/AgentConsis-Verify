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

Tool_Trace:
{tool_trace}

Available tools:
- search: use for external factual lookup. tool_args: {{"input": "query", "mode": "text"}}
- python_calculator: use for arithmetic or deterministic calculation. tool_args: {{"expression": "math expression"}}

Instructions:
- If you need one tool, return exactly this JSON shape:
{{"type": "tool_request", "reasoning_step": "step N. why this tool is needed", "tool_name": "search", "tool_args": {{"input": "query", "mode": "text"}}}}
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
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_TOOL_USER_PROMPT.format(
            question=compressed["question"],
            solver_result=compressed["solver_result"],
            attachment_result=compressed["attachment_result"],
            search_result=compressed["search_result"],
            tool_trace=compressed["tool_trace"],
        )
        return [
            {"role": "system", "content": str(compressed["system"])},
            {"role": "user", "content": user_content},
        ]


__all__ = [
    "STAGE1_TOOL_SYSTEM_PROMPT",
    "STAGE1_TOOL_USER_PROMPT",
    "Stage1ToolContextBuilder",
]
