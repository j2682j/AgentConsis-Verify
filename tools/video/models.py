from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoDownloadResult:
    ok: bool
    url: str
    video_path: Path | None = None
    title: str = ""
    video_id: str = ""
    duration: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["video_path"] = str(self.video_path) if self.video_path else ""
        return payload


@dataclass(frozen=True)
class FrameItem:
    frame_id: str
    timestamp_sec: float
    image_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "image_path": str(self.image_path),
        }


@dataclass(frozen=True)
class FrameAnalysisResult:
    frame_id: str
    timestamp_sec: float
    ok: bool
    answer_value: str = ""
    count: int | None = None
    evidence: str = ""
    confidence: float = 0.0
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoEvidenceResult:
    ok: bool
    url: str
    title: str = ""
    output_text: str = ""
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    frame_results: list[FrameAnalysisResult] = field(default_factory=list)
    answer_candidates: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_results"] = [item.to_dict() for item in self.frame_results]
        return payload


__all__ = [
    "FrameAnalysisResult",
    "FrameItem",
    "VideoDownloadResult",
    "VideoEvidenceResult",
]
