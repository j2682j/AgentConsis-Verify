from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import Tool, ToolParameter
from .tool_result import ToolExecutionResult, failure_result
from .video import (
    FrameExtractor,
    FrameSampler,
    VideoDownloader,
    VideoEvidenceBuilder,
    VideoEvidenceConfig,
    VisionFrameAnalyzer,
)


class VideoEvidenceTool(Tool):
    """
    Prepare visual evidence from a remote video URL.

    Args:
        - config: Optional runtime configuration for video extraction and vision analysis.

    Returns:
        - VideoEvidenceTool: ToolManager-compatible video evidence tool.
    """

    VIDEO_URL_RE = re.compile(
        r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[^\s)>\"]+",
        re.IGNORECASE,
    )
    REMOTE_VIDEO_RE = re.compile(r"https?://[^\s)>\"]+\.(?:mp4|mov|mkv|webm)(?:\?[^\s)]*)?", re.IGNORECASE)
    YOUTUBE_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,20}$")

    def __init__(self, config: VideoEvidenceConfig | None = None) -> None:
        super().__init__(
            name="video_evidence",
            description=(
                "Extract visual evidence from YouTube or remote video URLs by sampling frames "
                "and analyzing them with an Ollama vision model."
            ),
            capabilities={
                "video",
                "youtube_url",
                "video.visual",
                "video.frame_analysis",
                "vision",
                "evidence_text",
            },
            deterministic=False,
        )
        self.config = config or VideoEvidenceConfig.from_env()
        self.downloader = VideoDownloader(self.config)
        self.extractor = FrameExtractor(self.config)
        self.sampler = FrameSampler(max_frames=self.config.max_frames)
        self.analyzer = VisionFrameAnalyzer(self.config)
        self.builder = VideoEvidenceBuilder()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("url", "string", "YouTube or remote video URL.", True),
            ToolParameter("question", "string", "Original question.", True),
            ToolParameter("answer_role", "string", "Expected answer role.", False),
            ToolParameter("max_frames", "integer", "Maximum frames to analyze.", False),
        ]

    def run(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Download, sample, analyze, and aggregate visual video evidence.

        Args:
            - parameters: Tool arguments including url and question.

        Returns:
            - dict[str, Any]: ToolExecutionResult-compatible payload.
        """
        url = self._normalize_url(
            parameters.get("url")
            or parameters.get("input")
            or parameters.get("href")
            or parameters.get("query")
            or ""
        )
        question = str(parameters.get("question") or "").strip()
        answer_role = str(parameters.get("answer_role") or "").strip()
        if not url:
            return failure_result(
                "video_evidence",
                status="unsupported",
                error_code="missing_video_url",
                error_message="video_evidence requires a YouTube or remote video URL",
                retry_hint="Use a full remote video URL or use attachment_reader for local media.",
            )
        if not self._is_remote_url(url):
            return failure_result(
                "video_evidence",
                status="unsupported",
                error_code="invalid_video_url",
                error_message=f"{url!r} is not a valid remote video URL",
                retry_hint="Use a full http(s) YouTube/video URL.",
            )

        try:
            max_frames = int(parameters.get("max_frames") or self.config.max_frames)
        except (TypeError, ValueError):
            max_frames = self.config.max_frames

        with tempfile.TemporaryDirectory(prefix="scp_video_evidence_") as tmp:
            root = Path(tmp)
            download = self.downloader.download(url, root / "download")
            if not download.ok or not download.video_path:
                return failure_result(
                    "video_evidence",
                    status="retryable_failure",
                    error_code="video_download_failed",
                    error_message=download.error or "video download failed",
                    retryable=True,
                    retry_hint="Fallback to transcript or web search evidence.",
                    raw_result=download.to_dict(),
                )
            try:
                frames = self.extractor.extract(download.video_path, root / "frames")
            except Exception as exc:
                return failure_result(
                    "video_evidence",
                    status="retryable_failure",
                    error_code="frame_extraction_failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    retry_hint="Ensure ffmpeg is installed or fallback to transcript evidence.",
                    raw_result=download.to_dict(),
                )
            sampled = FrameSampler(max_frames=max_frames).sample(frames)
            frame_results = [
                self.analyzer.analyze(
                    frame=frame,
                    question=question,
                    answer_role=answer_role,
                )
                for frame in sampled
            ]
            built = self.builder.build(
                url=url,
                title=download.title,
                frame_results=frame_results,
            )

        if not built.ok:
            return failure_result(
                "video_evidence",
                status="partial",
                error_code="no_visual_evidence",
                error_message=built.error or "no useful visual evidence found",
                retryable=True,
                retry_hint="Fallback to video_transcript or web search.",
                raw_result=built.to_dict(),
            )

        return ToolExecutionResult(
            ok=True,
            tool_name="video_evidence",
            status="success",
            output_text=built.output_text,
            raw_result=built.to_dict(),
            evidence_valid=True,
        ).to_dict()

    @classmethod
    def extract_url(cls, text: str) -> str:
        source = str(text or "")
        match = cls.VIDEO_URL_RE.search(source) or cls.REMOTE_VIDEO_RE.search(source)
        if match:
            return match.group(0).rstrip(".,)")
        bare = source.strip()
        if cls.YOUTUBE_BARE_ID_RE.fullmatch(bare):
            return f"https://www.youtube.com/watch?v={bare}"
        return ""

    @classmethod
    def _normalize_url(cls, value: Any) -> str:
        text = str(value or "").strip()
        return cls.extract_url(text) or text

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        parsed = urlparse(str(value or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


__all__ = ["VideoEvidenceTool"]
