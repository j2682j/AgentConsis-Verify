from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.request import Request, urlopen

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMChatResult:
    """
    保存 provider-neutral 的語言模型回覆。

    Args:
        - content: 可供上層使用的文字；content 為空時回退至 reasoning。
        - reasoning: Provider 額外回傳的推理文字。
        - tool_calls: 正規化後的工具呼叫資料。
        - prompt_tokens: Prompt token 數量。
        - completion_tokens: Completion token 數量。
        - raw_response: Provider 原始 response。

    Returns:
        - LLMChatResult: Provider-neutral chat completion 結果。
    """

    content: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Any = None


class LLMClient:
    """
    統一呼叫 OpenAI-compatible LLM provider，例如 vLLM 與 Ollama。

    Args:
        - provider: Provider 名稱；預設讀取 LLM_PROVIDER。
        - base_url: OpenAI-compatible API base URL。
        - api_key: API key。
        - timeout: API timeout 秒數。
        - client: 測試或依賴注入使用的 OpenAI-compatible client。

    Returns:
        - LLMClient: 可執行一般與串流 chat completion 的共用 client。
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        client: Any | None = None,
        native_urlopen: Any | None = None,
    ) -> None:
        self.provider = (
            provider or os.getenv("LLM_PROVIDER", "ollama")
        ).strip().lower()
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
        )
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OLLAMA_API_KEY")
        )
        self.timeout = timeout or int(
            os.getenv("LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "180"))
        )
        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.native_urlopen = native_urlopen or urlopen

    def _ollama_native_url(self) -> str:
        base_url = (
            os.getenv("OLLAMA_HOST")
            or self.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url.rstrip('/')}/api/chat"

    def ollama_native_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool = False,
        json_format: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> LLMChatResult:
        """
        透過 Ollama native `/api/chat` 呼叫模型。

        Args:
            - model: Ollama 模型名稱。
            - messages: Ollama chat messages。
            - temperature: 生成溫度。
            - max_tokens: 最大生成 token 數，對應 Ollama num_predict。
            - think: 是否啟用模型 thinking 模式。
            - json_format: 是否要求 JSON，或直接提供 Ollama JSON Schema。
            - kwargs: 額外的 Ollama options。

        Returns:
            - LLMChatResult: 正規化後的文字、reasoning 與 usage。
        """
        if self.provider != "ollama":
            raise ValueError("ollama_native_chat() 僅適用於 Ollama provider。")

        options = dict(kwargs)
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "think": bool(think),
            "options": options,
        }
        if json_format:
            payload["format"] = (
                json_format if isinstance(json_format, dict) else "json"
            )

        request = Request(
            self._ollama_native_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.native_urlopen(request, timeout=self.timeout) as response:
            raw_data = response.read().decode("utf-8")
        data = json.loads(raw_data)
        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        reasoning = str(
            message.get("thinking")
            or message.get("reasoning")
            or ""
        ).strip()
        return LLMChatResult(
            content=content or reasoning,
            reasoning=reasoning,
            tool_calls=self._normalize_tool_calls(message),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            raw_response=data,
        )

    @staticmethod
    def _value(source: Any, name: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @classmethod
    def _normalize_tool_calls(cls, message: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for tool_call in cls._value(message, "tool_calls", []) or []:
            function = cls._value(tool_call, "function")
            name = str(cls._value(function, "name", "") or "").strip()
            raw_arguments = cls._value(function, "arguments", {})
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                try:
                    parsed = json.loads(str(raw_arguments or "{}"))
                    arguments = parsed if isinstance(parsed, dict) else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    arguments = {}
            normalized.append(
                {
                    "id": str(cls._value(tool_call, "id", "") or ""),
                    "name": name,
                    "arguments": arguments,
                }
            )
        return normalized

    def _prepare_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        enable_thinking: bool | None,
    ) -> list[dict[str, str]]:
        prepared = [dict(message) for message in messages]
        disable_thinking = (
            enable_thinking is False
            or (
                enable_thinking is None
                and not _env_bool("LLM_ENABLE_THINKING", False)
            )
        )
        if (
            self.provider == "ollama"
            and disable_thinking
            and "qwen3" in model.lower()
        ):
            for message in reversed(prepared):
                if message.get("role") != "user":
                    continue
                content = str(message.get("content", "") or "")
                if "/no_think" not in content:
                    message["content"] = f"{content.rstrip()}\n\n/no_think"
                break
        return prepared

    def _build_options(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
        enable_thinking: bool | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        options = dict(kwargs)
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if enable_thinking is not None and self.provider == "vllm":
            extra_body = dict(options.get("extra_body") or {})
            chat_template_kwargs = dict(
                extra_body.get("chat_template_kwargs") or {}
            )
            chat_template_kwargs.setdefault(
                "enable_thinking",
                bool(enable_thinking),
            )
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            options["extra_body"] = extra_body
        return options

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMChatResult:
        """
        呼叫 OpenAI-compatible chat completions endpoint。

        Args:
            - model: Provider 實際提供的模型名稱。
            - messages: OpenAI chat messages。
            - temperature: 生成溫度。
            - max_tokens: 最大 completion token 數量。
            - enable_thinking: 是否啟用模型 thinking 模式。
            - stream: 此方法僅接受 False；串流請使用 stream()。
            - kwargs: 其他 OpenAI-compatible completion 參數。

        Returns:
            - LLMChatResult: 文字、reasoning、工具呼叫、usage 與原始回覆。
        """
        if stream:
            raise ValueError("LLMClient.chat() 不支援 stream=True，請使用 stream()。")
        prepared_messages = self._prepare_messages(
            messages,
            model=model,
            enable_thinking=enable_thinking,
        )
        options = self._build_options(
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            kwargs=kwargs,
        )
        response = self.client.chat.completions.create(
            model=model,
            messages=prepared_messages,
            stream=False,
            **options,
        )
        message = response.choices[0].message
        message_content = str(self._value(message, "content", "") or "").strip()
        reasoning = str(
            self._value(message, "reasoning", "")
            or self._value(message, "reasoning_content", "")
            or ""
        ).strip()
        usage = self._value(response, "usage")
        return LLMChatResult(
            content=message_content or reasoning,
            reasoning=reasoning,
            tool_calls=self._normalize_tool_calls(message),
            prompt_tokens=int(self._value(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(
                self._value(usage, "completion_tokens", 0) or 0
            ),
            raw_response=response,
        )

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        串流呼叫模型並逐段回傳可顯示文字。

        Args:
            - model: Provider 實際提供的模型名稱。
            - messages: OpenAI chat messages。
            - temperature: 生成溫度。
            - max_tokens: 最大 completion token 數量。
            - enable_thinking: 是否啟用模型 thinking 模式。
            - kwargs: 其他 OpenAI-compatible completion 參數。

        Returns:
            - Iterator[str]: Assistant content 或 reasoning 的文字片段。
        """
        prepared_messages = self._prepare_messages(
            messages,
            model=model,
            enable_thinking=enable_thinking,
        )
        options = self._build_options(
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            kwargs=kwargs,
        )
        response = self.client.chat.completions.create(
            model=model,
            messages=prepared_messages,
            stream=True,
            **options,
        )
        for chunk in response:
            choices = self._value(chunk, "choices", []) or []
            if not choices:
                continue
            delta = self._value(choices[0], "delta")
            text = str(
                self._value(delta, "content", "")
                or self._value(delta, "reasoning", "")
                or self._value(delta, "reasoning_content", "")
                or ""
            )
            if text:
                yield text


__all__ = ["LLMChatResult", "LLMClient"]
