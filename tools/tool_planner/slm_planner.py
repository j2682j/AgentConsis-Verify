from __future__ import annotations

import json
import os
from typing import Any, Callable

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id

from .schema import ToolCandidate


class SLMToolPlanner:
    """
    Ask an SLM to choose and order tools from a system-provided candidate set.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        plan_callable: Callable[[list[dict[str, str]]], str] | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.model_name = (
            str(model_name)
            if model_name is not None
            else os.getenv("TOOL_PLANNER_MODEL", "qwen3:4b")
        )
        self.llm_client = llm_client
        self.plan_callable = plan_callable
        self.max_tokens = max(64, max_tokens)

    def plan(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None,
        candidates: list[ToolCandidate],
        routing: dict[str, Any] | None = None,
    ) -> str:
        if self.plan_callable is not None:
            return self.plan_callable(
                self._messages(
                    question=question,
                    attachment=attachment,
                    candidates=candidates,
                    routing=routing,
                )
            )
        if not self.model_name:
            return ""

        client = self.llm_client or LLMClient()
        messages = self._messages(
            question=question,
            attachment=attachment,
            candidates=candidates,
            routing=routing,
        )
        model = resolve_model_id(self.model_name)
        if client.provider == "ollama":
            result = client.ollama_native_chat(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                think=False,
                json_format=True,
            )
        else:
            result = client.chat(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                enable_thinking=False,
                response_format={"type": "json_object"},
            )
        return result.content

    def _messages(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None,
        candidates: list[ToolCandidate],
        routing: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        candidate_payload = [candidate.to_dict() for candidate in candidates]
        attachment_payload = {
            key: value
            for key, value in (attachment or {}).items()
            if key in {"file_path", "path", "file_name", "extension"}
        }
        user_payload = {
            "question": question,
            "attachment": attachment_payload,
            "routing_hints": routing or {},
            "candidate_tools": candidate_payload,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a tool planner. Select only from candidate_tools. "
                    "Return JSON only with requires_tools, tool_needs, tool_sequence, handler_plans, and stop_condition. "
                    "Use at most 3 steps. Prefer tool_needs as required capabilities. "
                    "For deterministic_handler plans, include required_handler_role such as simple_math, numeric_arithmetic, unit_conversion, list_operation, table_reasoning, graph_search, coordinate_distance, date_time, string_transform, text_extraction, boggle_dfs, probability_simulation, logic_equivalence, chess_tactics, or unknown. "
                    "Do not invent tool or handler names."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]


__all__ = ["SLMToolPlanner"]
