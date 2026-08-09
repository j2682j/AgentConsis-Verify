"""Per-task VRAM snapshots, so a GPU fault can be read back afterwards.

level1_final_11 died at task 4 of 53 with `CUDA error: unknown error`, and the
run then burned three more hours producing empty results. The telemetry that
survived showed the fault was not instantaneous: on the task immediately
before it, VersaPRM ran at 473 ms/step against a 24 ms median for comparable
tasks -- a five-fold collapse, which is what host-memory spilling looks like
when VRAM runs out. Nothing in the trace recorded how much VRAM was actually
in use, so that stayed a hypothesis.

Two numbers are kept because they answer different questions. `device_*` is
the whole card, which includes Ollama's llama-server; `torch_*` is only this
process. If the device figure climbs across tasks while the torch figure does
not, whatever is accumulating is outside torch.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - torch is a hard dependency in practice
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_MB = 1024 * 1024


def available() -> bool:
    """True when there is a CUDA device to measure."""

    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def snapshot() -> dict[str, Any]:
    """Current VRAM use, or an empty dict when there is no CUDA device."""

    if not available():
        return {}
    try:
        free, total = torch.cuda.mem_get_info()
        return {
            "device_used_mb": round((total - free) / _MB),
            "device_free_mb": round(free / _MB),
            "device_total_mb": round(total / _MB),
            "torch_allocated_mb": round(torch.cuda.memory_allocated() / _MB),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / _MB),
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / _MB),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def begin_task() -> dict[str, Any]:
    """Snapshot and reset the process peak so the next reading is per-task."""

    before = snapshot()
    if available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    return before


def summarize(before: dict[str, Any], after: dict[str, Any]) -> str:
    """One compact line for the run log, empty when there is nothing to say."""

    if not before or not after or "error" in after:
        return ""
    used = after.get("device_used_mb", 0)
    total = after.get("device_total_mb", 0)
    delta = used - before.get("device_used_mb", 0)
    return (
        f"vram device={used}/{total}MB ({delta:+d}MB) "
        f"torch_peak={after.get('torch_peak_reserved_mb', 0)}MB"
    )


__all__ = ["available", "begin_task", "snapshot", "summarize"]
