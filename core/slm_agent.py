from __future__ import annotations

import json
import os
from typing import Any, Iterator

from exceptions import AgentsException

from .llm_client import LLMChatResult, LLMClient
from .model_registry import resolve_model_id


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def estimate_text_tokens(text: str) -> int:
    """
    在 provider 未回傳 usage 時估算文字 token 數量。

    Args:
        - text: 要估算的文字。

    Returns:
        - int: 估算 token 數量；空字串回傳 0。
    """
    normalized = str(text or "")
    if not normalized:
        return 0
    return max(1, int(len(normalized) / 4))


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    """
    在 provider 未回傳 usage 時估算 chat prompt token 數量。

    Args:
        - messages: OpenAI-compatible chat messages。

    Returns:
        - int: 估算的 prompt token 數量。
    """
    total = 0
    for message in messages or []:
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
        total += 4
    return total


class SLM_Agent:
    """
    保存 Agent 層生成設定，並透過共用 LLMClient 呼叫語言模型。

    Args:
        - api_key: OpenAI-compatible API key。
        - base_url: OpenAI-compatible server base URL。
        - temperature: 預設生成溫度。
        - max_tokens: 預設最大 completion token 數量。
        - timeout: API timeout 秒數。
        - model_name: 系統使用的模型別名。
        - llm_client: 可注入的共用 provider client。
        - kwargs: reasoning_effort 等額外 completion 參數。

    Returns:
        - SLM_Agent: 提供 invoke、invoke_with_usage、stream_invoke 的 Agent。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.5,
        max_tokens: int | None = None,
        timeout: int | None = None,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = str(model_name or "")
        self.model = resolve_model_id(self.model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        explicit_enable_thinking = kwargs.pop("enable_thinking", None)
        self.enable_thinking = (
            _env_bool("LLM_ENABLE_THINKING", False)
            if explicit_enable_thinking is None
            else bool(explicit_enable_thinking)
        )
        self.reasoning_effort = kwargs.pop(
            "reasoning_effort",
            "low" if "gpt-oss" in self.model_name.lower() else None,
        )
        self.kwargs = kwargs
        self.llm_client = llm_client or LLMClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _completion_options(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(self.kwargs)
        options.update(overrides or {})
        options.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            options.setdefault("max_tokens", self.max_tokens)
        options.setdefault("enable_thinking", self.enable_thinking)
        if self.reasoning_effort:
            options.setdefault("reasoning_effort", self.reasoning_effort)
        return options

    @staticmethod
    def _result_content(result: LLMChatResult) -> str:
        if result.tool_calls:
            tool_call = result.tool_calls[0]
            tool_name = str(tool_call.get("name", "") or "").strip()
            reasoning_step = result.reasoning or (
                f"step 1. Request the {tool_name or 'requested'} tool."
            )
            return json.dumps(
                {
                    "type": "tool_request",
                    "reasoning_step": reasoning_step,
                    "tool_name": tool_name,
                    "tool_args": tool_call.get("arguments", {}) or {},
                },
                ensure_ascii=False,
            )
        return result.content

    def _chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMChatResult:
        verbose = _env_bool("VERBOSE_LLM_CALLS", False)
        if verbose:
            print(f"[LLM] calling model={self.model}")
        try:
            options = self._completion_options(kwargs)
            if getattr(self.llm_client, "provider", "") == "ollama":
                result = self._ollama_native_chat(messages, options)
            else:
                options.pop("keep_alive", None)
                options.pop("unload_after_call", None)
                result = self.llm_client.chat(
                    model=self.model,
                    messages=messages,
                    **options,
                )
            if verbose:
                print(f"[LLM] model={self.model} response received")
            return result
        except Exception as exc:
            raise AgentsException(
                f"SLM chat 呼叫失敗: {type(exc).__name__}: {exc}"
            ) from exc

    def _ollama_native_chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> LLMChatResult:
        temperature = options.pop("temperature", self.temperature)
        max_tokens = options.pop("max_tokens", self.max_tokens)
        enable_thinking = bool(options.pop("enable_thinking", self.enable_thinking))
        options.pop("reasoning_effort", None)
        keep_alive = options.pop("keep_alive", None)
        if bool(options.pop("unload_after_call", False)):
            keep_alive = 0
        return self.llm_client.ollama_native_chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            think=enable_thinking,
            keep_alive=keep_alive,
            **options,
        )

    def unload(self) -> dict[str, Any]:
        if getattr(self.llm_client, "provider", "") != "ollama":
            return {
                "model": self.model,
                "provider": getattr(self.llm_client, "provider", ""),
                "unload_method": "none",
                "unloaded": False,
                "warning": "provider_is_not_ollama",
            }
        try:
            self.llm_client.ollama_native_chat(
                model=self.model,
                messages=[{"role": "user", "content": ""}],
                temperature=0,
                max_tokens=1,
                think=False,
                keep_alive=0,
            )
            return {
                "model": self.model,
                "provider": "ollama",
                "unload_method": "keep_alive_0",
                "unloaded": True,
                "warning": "",
            }
        except Exception as exc:
            return {
                "model": self.model,
                "provider": "ollama",
                "unload_method": "keep_alive_0",
                "unloaded": False,
                "warning": f"{type(exc).__name__}: {exc}",
            }

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Any:
        """
        呼叫模型並回傳 provider 原始 response。

        Args:
            - messages: OpenAI-compatible chat messages。
            - temperature: 此次呼叫覆蓋的生成溫度。

        Returns:
            - Any: Provider 原始 chat completion response。
        """
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        return self._chat(messages, **overrides).raw_response

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        呼叫模型並回傳上層可解析的 assistant 文字。

        Args:
            - messages: OpenAI-compatible chat messages。
            - kwargs: 此次呼叫覆蓋的 completion 參數。

        Returns:
            - str: 模型文字或正規化後的 tool_request JSON。
        """
        return self._result_content(self._chat(messages, **kwargs))

    def invoke_with_usage(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        """
        呼叫模型並同時回傳 prompt 與 completion token usage。

        Args:
            - messages: OpenAI-compatible chat messages。
            - kwargs: 此次呼叫覆蓋的 completion 參數。

        Returns:
            - tuple[str, int, int]: 文字、prompt tokens、completion tokens。
        """
        result = self._chat(messages, **kwargs)
        content = self._result_content(result)
        prompt_tokens = result.prompt_tokens or estimate_chat_tokens(messages)
        completion_tokens = (
            result.completion_tokens or estimate_text_tokens(content)
        )
        return content, prompt_tokens, completion_tokens

    def stream_invoke(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        串流呼叫模型並逐段回傳文字。

        Args:
            - messages: OpenAI-compatible chat messages。
            - kwargs: 此次呼叫覆蓋的 completion 參數。

        Returns:
            - Iterator[str]: 模型回傳的文字片段。
        """
        try:
            yield from self.llm_client.stream(
                model=self.model,
                messages=messages,
                **self._completion_options(kwargs),
            )
        except Exception as exc:
            raise AgentsException(
                f"SLM stream 呼叫失敗: {type(exc).__name__}: {exc}"
            ) from exc


__all__ = [
    "SLM_Agent",
    "estimate_chat_tokens",
    "estimate_text_tokens",
]
