from __future__ import annotations

import json
import os
from typing import Any, Callable

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id

from .models import AttachmentStrategy
from .parser import AttachmentStrategyParser


class AttachmentStrategyPlanner:
    """
    使用小型語言模型規劃附件題的 handler 解題策略。

    Args:
     - model_name: 用於策略規劃的模型名稱。
     - llm_client: 共用 LLM client。
     - parser: 策略 JSON parser。
     - plan_callable: 測試用替代呼叫。
     - max_tokens: 策略輸出的最大 token 數。

    Returns:
     - AttachmentStrategyPlanner: 回傳 AttachmentStrategy 的規劃器。
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        parser: AttachmentStrategyParser | None = None,
        plan_callable: Callable[[list[dict[str, str]]], str] | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.model_name = (
            str(model_name)
            if model_name is not None
            else os.getenv("TOOL_PLANNER_MODEL", "qwen3:4b")
        )
        self.llm_client = llm_client
        self.parser = parser or AttachmentStrategyParser()
        self.plan_callable = plan_callable
        self.max_tokens = max(96, max_tokens)

    def plan(
        self,
        *,
        question: str,
        attachment_profile: dict[str, Any],
        allowed_handlers: list[dict[str, Any]] | None = None,
    ) -> tuple[AttachmentStrategy, str]:
        messages = self._messages(
            question=question,
            attachment_profile=attachment_profile,
            allowed_handlers=allowed_handlers or [],
        )
        if self.plan_callable is not None:
            raw = self.plan_callable(messages)
            return self.parser.parse(raw), raw

        client = self.llm_client or LLMClient()
        model = resolve_model_id(self.model_name)
        if client.provider == "ollama":
            result = client.ollama_native_chat(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                think=False,
                json_format=True,
                keep_alive=0,
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
        raw = result.content
        return self.parser.parse(raw), raw

    def _messages(
        self,
        *,
        question: str,
        attachment_profile: dict[str, Any],
        allowed_handlers: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        payload = {
            "question": question,
            "attachment_profile": attachment_profile,
            "allowed_handlers": allowed_handlers,
        }
        return [
            {
                "role": "system",
                "content": (
                    "Plan how to solve an attachment-based task. "
                    "Return JSON only. Do not answer the task. "
                    "Choose at most one exact handler_name from allowed_handlers. "
                    "Use only inputs listed in attachment_profile.available_inputs. "
                    "If no listed handler can perform the operation, leave required_handler empty. "
                    "Set needs_search only when the parsed attachment cannot provide required information."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": payload,
                        "schema": {
                            "information_need": "specific information required from the attachment",
                            "required_handler": "one allowed handler name or role, otherwise empty",
                            "required_inputs": ["inputs that must already be available"],
                            "expected_answer": "natural language answer requirement",
                            "needs_search": False,
                            "missing_inputs": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]


__all__ = ["AttachmentStrategyPlanner"]
