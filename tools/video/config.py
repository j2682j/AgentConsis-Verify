from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEvidenceConfig:
    """
    Runtime configuration for remote video evidence preparation.

    Args:
        - vision_model: Ollama vision model used for frame analysis.
        - max_frames: Maximum sampled frames sent to the vision model.
        - frame_interval_sec: Base interval used for ffmpeg frame extraction.
        - max_video_seconds: Optional upper bound for downloaded video length.

    Returns:
        - VideoEvidenceConfig: Immutable video evidence runtime settings.
    """

    vision_model: str = "qwen3-vl:4b"
    max_frames: int = 32
    frame_interval_sec: float = 2.0
    max_video_seconds: int = 900
    timeout_sec: int = 240
    max_tokens: int = 384
    cookiefile: str = ""

    @classmethod
    def from_env(cls) -> "VideoEvidenceConfig":
        return cls(
            vision_model=os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b"),
            max_frames=int(os.getenv("VIDEO_EVIDENCE_MAX_FRAMES", "32")),
            frame_interval_sec=float(os.getenv("VIDEO_EVIDENCE_FRAME_INTERVAL_SEC", "2")),
            max_video_seconds=int(os.getenv("VIDEO_EVIDENCE_MAX_VIDEO_SECONDS", "900")),
            timeout_sec=int(os.getenv("VIDEO_EVIDENCE_TIMEOUT", "240")),
            max_tokens=int(os.getenv("VIDEO_EVIDENCE_MAX_TOKENS", "384")),
            cookiefile=(
                os.getenv("VIDEO_EVIDENCE_COOKIEFILE")
                or os.getenv("YTDLP_COOKIEFILE")
                or ""
            ),
        )


__all__ = ["VideoEvidenceConfig"]
