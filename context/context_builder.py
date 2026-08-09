from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


@dataclass
class ContextPacket:
    """A content unit used to build agent messages."""

    packet_type: str
    content: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0
    timestamp: dt.datetime = field(default_factory=dt.datetime.now)

    def __post_init__(self) -> None:
        self.content = "" if self.content is None else str(self.content)
        if self.token_count <= 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """Settings for context selection and compression."""

    max_tokens: int = 8000
    reserve_ratio: float = 0.15
    min_relevance: float = 0.0
    # These bounds were raised to 240/24000 for level1_final_09 and reverted:
    # the extra evidence made the 4B Agents worse, not better. Grouping that
    # run's 477 Agent runs by how much their context grew against
    # level1_final_08 gives a dose-response, so this is not run-to-run noise:
    #
    #   context within 1.2x   366 runs   30.6% -> 29.8% correct
    #   grown 1.2-2x           67 runs   16.4% -> 11.9%
    #   grown over 2x          44 runs   29.5% -> 15.9%
    #
    # Two tasks show the mechanism. On 5a0c1adf every Agent answered 'Claus'
    # from a 4,964-character context and none did from 19,672; on 840bfca7 two
    # Agents produced the exact award number from about 5,000 characters and
    # all three produced different wrong ones from 14,000. Fitting the window is
    # not the constraint -- the largest context ever built was 35,998
    # characters against windows of 32k and up. Locating the answer inside it
    # is.
    #
    # Note for anyone retuning these: compression cuts lines first and
    # characters second, so the line bound decides the outcome. A retrieval
    # record runs 6 lines and 443 characters at the median and retrieval hands
    # over 16 of them -- 96 lines against 80, but only 7,088 characters against
    # 12,000. Moving the character bound alone changes nothing.
    max_context_lines: int = 80
    max_context_chars: int = 12000
    max_solver_chars: int = 2000
    none_text: str = "None"
    enable_compression: bool = True

    def get_available_tokens(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """Base class for building chat messages from context packets."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    def build(self, **kwargs: Any) -> list[dict[str, str]]:
        messages, _diagnostics = self.build_with_diagnostics(**kwargs)
        return messages

    def build_with_diagnostics(self, **kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
        packets = self.gather(**kwargs)
        selected = self.select(packets, **kwargs)
        structured = self.structure(selected, **kwargs)
        compressed = self.compress(structured, **kwargs)
        diagnostics = compressed.get("_context_budget", {})
        return self.render(compressed, **kwargs), dict(diagnostics or {})

    def gather(self, **kwargs: Any) -> list[ContextPacket]:
        raise NotImplementedError

    def select(self, packets: list[ContextPacket], **kwargs: Any) -> list[ContextPacket]:
        raise NotImplementedError

    def structure(self, packets: list[ContextPacket], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def compress(self, structured: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, compressed: dict[str, Any], **kwargs: Any) -> list[dict[str, str]]:
        raise NotImplementedError

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().split())

    def _compress_multiline_text(
        self,
        text: Any,
        *,
        max_lines: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        raw = "" if text is None else str(text).strip()
        if not raw or raw == self.config.none_text:
            return ""

        max_lines = max_lines or self.config.max_context_lines
        max_chars = max_chars or self.config.max_context_chars
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        compressed = "\n".join(lines[:max_lines]).strip()
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars].rstrip() + " ..."
        return compressed


def count_tokens(text: Any) -> int:
    value = "" if text is None else str(text)
    if not value:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(value))
        except Exception:
            pass
    return max(1, len(value) // 4)


ContextBuildConfig = ContextConfig


__all__ = [
    "ContextBuildConfig",
    "ContextBuilder",
    "ContextConfig",
    "ContextPacket",
    "count_tokens",
]
