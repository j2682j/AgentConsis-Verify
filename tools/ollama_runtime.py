from __future__ import annotations

import shutil
import subprocess


def stop_ollama_model(model_name: str, *, timeout: int = 30) -> str:
    """
    Stop an Ollama model if the Ollama CLI is available.

    Args:
        - model_name: Ollama model name to unload.
        - timeout: Maximum seconds to wait for the stop command.

    Returns:
        - str: Empty string on success, otherwise a warning message.
    """
    model = str(model_name or "").strip()
    if not model:
        return "missing Ollama model name"
    if shutil.which("ollama") is None:
        return "ollama CLI not found"

    try:
        completed = subprocess.run(
            ["ollama", "stop", model],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return f"ollama stop failed: {type(exc).__name__}: {exc}"

    if completed.returncode != 0:
        stderr = str(completed.stderr or "").strip()
        stdout = str(completed.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={completed.returncode}"
        return f"ollama stop failed for {model}: {detail}"
    return ""


def stop_ollama_models(model_names: list[str] | set[str] | tuple[str, ...]) -> list[dict[str, str]]:
    """
    Stop multiple Ollama models once each.

    Args:
        - model_names: Ollama model names to unload.

    Returns:
        - list[dict[str, str]]: Stop status records for metadata/reporting.
    """
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for raw_model in model_names:
        model = str(raw_model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        warning = stop_ollama_model(model)
        records.append(
            {
                "model": model,
                "stopped": "true" if not warning else "false",
                "warning": warning,
            }
        )
    return records


__all__ = ["stop_ollama_model", "stop_ollama_models"]
