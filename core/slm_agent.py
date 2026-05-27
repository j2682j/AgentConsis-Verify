"""本地 OpenAI-compatible SLM/Ollama 呼叫封裝。"""

import os
from typing import Iterator, Optional

from openai import OpenAI

from exceptions import AgentsException


MODEL_ID_MAP = {
    "nemotron-mini:4b": os.getenv("Nemotron_MODEL_ID"),
    "minicpm3:4b": os.getenv("Minicpm_MODEL_ID"),
    "qwen3:4b": os.getenv("Qwen_MODEL_ID"),
    "gemma3:4b": os.getenv("Gemma_MODEL_ID"),
    "gpt-oss:20b": os.getenv("GPT_OSS_MODEL_ID"),
}


def estimate_text_tokens(text: str) -> int:
    """
    以簡單字元長度估算文字 token 數，作為本地模型未回傳 usage 時的 fallback。

    Args:
        - text: 要估算 token 的文字。

    Returns:
        - int: 估算 token 數；空字串回傳 0。
    """
    normalized = str(text or "")
    if not normalized:
        return 0
    return max(1, int(len(normalized) / 4))


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    """
    估算 OpenAI-compatible chat messages 的 prompt token 數。

    Args:
        - messages: chat messages 清單，每個元素包含 role 與 content。

    Returns:
        - int: 估算出的 prompt token 數。
    """
    total = 0
    for message in messages or []:
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
        total += 4
    return total


class SLM_Agent:
    """
    封裝本地 OpenAI-compatible SLM/Ollama 模型呼叫，負責建立 client、
    送出 chat messages、取得文字回覆，並在需要時回傳 token usage 估算值。

    Args:
        - api_key: OpenAI-compatible API key，未提供時使用 OLLAMA_API_KEY。
        - base_url: OpenAI-compatible server base URL，未提供時使用 OLLAMA_BASE_URL。
        - temperature: 模型生成多樣性 
        - max_tokens: 單次模型回覆的最大 token 數。
        - timeout: API 呼叫 timeout 秒數，未提供時使用 OLLAMA_TIMEOUT。
        - model_name: 專案內部模型別名，例如 nemotron-mini:4b、phi4-mini:3.8b、qwen3:4b、gemma3:4b。
        - kwargs: 傳給 Agent 的其他額外設定。

    Returns:
        - SLM_Agent: 可透過 invoke、invoke_with_usage、stream_invoke 呼叫 SLM 的 Agent 物件。
        - invoke(): 回傳模型文字內容。
        - invoke_with_usage(): 回傳 tuple，格式為 (content, prompt_tokens, completion_tokens)。
        - stream_invoke(): 逐段回傳模型串流文字。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.5, # 0.0 為最保守，1.0 為較大創意發揮空間；默認 0.5 提供適度多樣性
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        model_env_key_map = {
            "nemotron-mini:4b": "Nemotron_MODEL_ID",
            "phi4-mini:3.8b": "Phi_MODEL_ID",
            "qwen3:4b": "Qwen_MODEL_ID",
            "gemma3:4b": "Gemma_MODEL_ID",
            "minicpm3_4b:latest": "Minicpm_MODEL_ID",
            "gpt-oss:20b": "GPT_OSS_MODEL_ID",
        }

        env_key = model_env_key_map.get(model_name)
        self.model = os.getenv(env_key) if env_key else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.kwargs = kwargs

        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        """
        建立 OpenAI-compatible client。

        Args:
            - 無。

        Returns:
            - OpenAI: 使用目前 api_key、base_url 與 timeout 的 client。
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ):
        """
        呼叫 chat completions API 並回傳完整 response 物件。

        Args:
            - messages: 要傳給模型的 chat messages。
            - temperature: 此次呼叫覆蓋用的生成溫度。

        Returns:
            - Any: OpenAI-compatible chat completion response。
        """
        verbose = os.getenv("VERBOSE_LLM_CALLS", "0").lower() in {"1", "true", "yes", "on"}
        if verbose:
            print(f"[LLM] calling model={self.model}")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            if verbose:
                print(f"[LLM] model={self.model} response received")
            return response
        except Exception as e:
            print(f"[ERROR] LLM API failed: {e}")
            raise AgentsException("SLM API 呼叫失敗")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        呼叫模型並直接回傳 assistant message content。

        Args:
            - messages: 要傳給模型的 chat messages。
            - kwargs: 覆蓋 temperature、max_tokens 或其他 chat completion 參數。

        Returns:
            - str: 模型回覆文字。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            )
            return response.choices[0].message.content
        except Exception:
            raise AgentsException("SLM chat 呼叫失敗")

    def invoke_with_usage(self, messages: list[dict[str, str]], **kwargs) -> tuple[str, int, int]:
        """
        呼叫模型並回傳文字內容與 token usage；若 server 未提供 usage 則使用估算值。

        Args:
            - messages: 要傳給模型的 chat messages。
            - kwargs: 覆蓋 temperature、max_tokens 或其他 chat completion 參數。

        Returns:
            - str: 模型回覆文字。
            - int: prompt token 數。
            - int: completion token 數。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            )
            content = response.choices[0].message.content
            prompt_tokens = (
                response.usage.prompt_tokens
                if response.usage and hasattr(response.usage, "prompt_tokens")
                else 0
            )
            completion_tokens = (
                response.usage.completion_tokens
                if response.usage and hasattr(response.usage, "completion_tokens")
                else 0
            )
            if prompt_tokens <= 0:
                prompt_tokens = estimate_chat_tokens(messages)
            if completion_tokens <= 0:
                completion_tokens = estimate_text_tokens(content)
            return content, prompt_tokens, completion_tokens
        except Exception:
            raise AgentsException("SLM 呼叫或 usage 解析失敗")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        以 iterator 形式回傳模型串流輸出。

        Args:
            - messages: 要傳給模型的 chat messages。
            - kwargs: 覆蓋 temperature 等呼叫參數。

        Returns:
            - Iterator[str]: 模型串流輸出的文字片段。
        """
        temperature = kwargs.get("temperature")
        yield from self.think(messages, temperature)
