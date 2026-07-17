from __future__ import annotations

import json
import os
from typing import Any, Callable

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id

from .models import AttachmentStrategy
from .parser import AttachmentStrategyParser


class AttachmentStrategyReviewer:
    """
    在 handler 失敗或資訊不足時，讓模型修正附件策略一次。

    Args:
     - model_name: 用於策略修正的模型名稱。
     - llm_client: 共用 LLM client。
     - parser: 策略 JSON parser。
     - review_callable: 測試用替代呼叫。
     - max_tokens: review 輸出的最大 token 數。

    Returns:
     - AttachmentStrategyReviewer: 回傳修正策略與候選答案。
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        parser: AttachmentStrategyParser | None = None,
        review_callable: Callable[[list[dict[str, str]]], str] | None = None,
        max_tokens: int = 320,
    ) -> None:
        self.model_name = (
            str(model_name)
            if model_name is not None
            else os.getenv("TOOL_PLANNER_MODEL", "qwen3:4b")
        )
        self.llm_client = llm_client
        self.parser = parser or AttachmentStrategyParser()
        self.review_callable = review_callable
        self.max_tokens = max(128, max_tokens)

    def review(
        self,
        *,
        question: str,
        strategy: AttachmentStrategy,
        handler_results: list[dict[str, Any]],
        attachment_profile: dict[str, Any] | None = None,
        allowed_handlers: list[dict[str, Any]] | None = None,
    ) -> tuple[AttachmentStrategy | None, str, str]:
        messages = self._messages(
            question=question,
            strategy=strategy,
            handler_results=handler_results,
            attachment_profile=attachment_profile or {},
            allowed_handlers=allowed_handlers or [],
        )
        if self.review_callable is not None:
            raw = self.review_callable(messages)
            return self._parse_review(raw)

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
        return self._parse_review(result.content)

    def _parse_review(self, raw: str) -> tuple[AttachmentStrategy | None, str, str]:
        from parsers.json_parse import try_parse_json

        parsed = try_parse_json(raw)
        if not isinstance(parsed, dict):
            return None, "", raw
        revised = parsed.get("revised_strategy")
        strategy = self.parser.from_dict(revised) if isinstance(revised, dict) else None
        return strategy, "", raw

    def _messages(
        self,
        *,
        question: str,
        strategy: AttachmentStrategy,
        handler_results: list[dict[str, Any]],
        attachment_profile: dict[str, Any],
        allowed_handlers: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        payload = {
            "question": question,
            "previous_strategy": strategy.to_dict(),
            "handler_results": handler_results,
            "attachment_profile": attachment_profile,
            "allowed_handlers": allowed_handlers,
        }
        return [
            {
                "role": "system",
                "content": (
                    "Revise one attachment strategy after handler feedback. "
                    "Return JSON only. Use only exact handler_name values listed in allowed_handlers. "
                    "Do not answer the task. Select at most one next handler."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": payload,
                        "schema": {
                            "revised_strategy": {
                                "information_need": "",
                                "required_handler": "",
                                "required_inputs": [],
                                "expected_answer": "",
                                "needs_search": False,
                                "missing_inputs": [],
                            },
                            "reason": "",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]


__all__ = ["AttachmentStrategyReviewer"]
