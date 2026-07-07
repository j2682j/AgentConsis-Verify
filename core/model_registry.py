from __future__ import annotations

import os


MODEL_ENV_KEYS = {
    "nemotron-3-nano:4b": "Nemotron_MODEL_ID",
    "qwen3:4b": "Qwen_MODEL_ID",
    "gemma3:4b": "Gemma_MODEL_ID",
}


def resolve_model_id(model_name: str | None) -> str:
    """
    將系統內的模型別名解析成 provider 實際提供的模型名稱。

    Args:
        - model_name: AgentConfig 或其他呼叫端使用的模型別名。

    Returns:
        - str: 環境變數覆蓋後的模型 ID；未設定時保留原名稱。
    """
    alias = str(model_name or "").strip()
    if not alias:
        return ""
    env_key = MODEL_ENV_KEYS.get(alias)
    if not env_key:
        return alias
    return str(os.getenv(env_key, "") or alias).strip()


__all__ = ["MODEL_ENV_KEYS", "resolve_model_id"]
