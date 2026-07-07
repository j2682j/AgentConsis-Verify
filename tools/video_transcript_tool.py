from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import Tool, ToolParameter
from .tool_result import failure_result


class VideoTranscriptTool(Tool):
    """Extract transcript evidence from a video URL."""

    YOUTUBE_ID_RE = re.compile(
        r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)"
        r"([A-Za-z0-9_-]{6,})"
    )
    YOUTUBE_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,20}$")

    def __init__(self) -> None:
        super().__init__(
            name="video_transcript",
            description=(
                "Extract captions or ASR transcript from YouTube/video URLs. "
                "Use this for questions that require listening to or reading video content."
            ),
            capabilities={
                "video.transcript",
                "video.caption",
                "video.audio_asr",
                "web.video",
            },
            attachment_types=set(),
            deterministic=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="YouTube or video URL to transcribe.",
                required=True,
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Preferred language code, for example en.",
                required=False,
            ),
            ToolParameter(
                name="max_chars",
                type="integer",
                description="Maximum transcript characters to return.",
                required=False,
            ),
            ToolParameter(
                name="allow_asr",
                type="boolean",
                description="Allow audio download and Whisper ASR if captions are unavailable.",
                required=False,
            ),
        ]

    def run(self, parameters: dict[str, Any]) -> str | dict[str, Any]:
        url = str(
            parameters.get("url")
            or parameters.get("input")
            or parameters.get("href")
            or parameters.get("query")
            or ""
        ).strip()
        if not url:
            return self._failure(
                error_code="missing_video_url",
                error_message="video_transcript requires a YouTube or remote video URL",
            )

        if self._looks_like_local_media_path(url):
            return self._failure(
                status="unsupported",
                error_code="local_media_requires_attachment_reader",
                error_message=(
                    "video_transcript received a local media file path; use attachment_reader "
                    "for local audio/video attachments"
                ),
                retry_hint="Call attachment_reader with file_path instead of video_transcript.",
            )

        if not self._is_remote_url(url):
            bare_id = self._youtube_bare_id(url)
            if bare_id:
                url = f"https://www.youtube.com/watch?v={bare_id}"
            else:
                return self._failure(
                    status="unsupported",
                    error_code="invalid_video_url",
                    error_message=f"{url!r} is not a valid remote video URL",
                    retry_hint="Use a full http(s) YouTube/video URL, or attachment_reader for local files.",
                )

        language = str(parameters.get("language") or "en").strip() or "en"
        max_chars = int(parameters.get("max_chars") or 24000)
        max_chars = max(2000, min(max_chars, 60000))
        allow_asr = self._bool(parameters.get("allow_asr"), default=True)

        errors: list[str] = []
        youtube_id = self._youtube_id(url)
        if youtube_id:
            try:
                return self._fetch_youtube_captions(
                    video_id=youtube_id,
                    url=url,
                    language=language,
                    max_chars=max_chars,
                )
            except Exception as exc:
                errors.append(f"caption: {type(exc).__name__}: {exc}")

        if allow_asr:
            try:
                return self._transcribe_audio(
                    url=url,
                    language=language,
                    max_chars=max_chars,
                    previous_errors=errors,
                )
            except Exception as exc:
                errors.append(f"asr: {type(exc).__name__}: {exc}")

        return self._failure(
            status="unavailable",
            error_code="video_transcript_unavailable",
            error_message="video transcript unavailable; " + "; ".join(errors),
            retry_hint="Use search or attachment_reader if another source is available.",
        )

    @classmethod
    def _youtube_id(cls, url: str) -> str:
        match = cls.YOUTUBE_ID_RE.search(url)
        if match:
            return match.group(1)
        return ""

    @classmethod
    def _youtube_bare_id(cls, value: str) -> str:
        candidate = str(value or "").strip()
        return candidate if cls.YOUTUBE_BARE_ID_RE.fullmatch(candidate) else ""

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        parsed = urlparse(str(value or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _looks_like_local_media_path(value: str) -> bool:
        candidate = str(value or "").strip()
        if not candidate:
            return False
        if VideoTranscriptTool._is_remote_url(candidate):
            return False
        suffix = Path(candidate).suffix.lower()
        return suffix in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}

    @staticmethod
    def _failure(
        *,
        status: str = "retryable_failure",
        error_code: str,
        error_message: str,
        retry_hint: str = "Use a valid remote video URL or select another tool.",
    ) -> dict[str, Any]:
        return failure_result(
            "video_transcript",
            status=status,
            error_code=error_code,
            error_message=error_message,
            retryable=False,
            retry_hint=retry_hint,
        )

    @staticmethod
    def _bool(value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    class _QuietYtdlpLogger:
        def debug(self, message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            pass

        def error(self, message: str) -> None:
            pass

    @staticmethod
    def _timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @classmethod
    def _format_lines(cls, items: list[tuple[float, str]], *, max_chars: int) -> str:
        lines: list[str] = []
        length = 0
        for start, text in items:
            cleaned = " ".join(str(text or "").split())
            if not cleaned:
                continue
            line = f"[{cls._timestamp(start)}] {cleaned}"
            if length + len(line) + 1 > max_chars:
                lines.append("[truncated]")
                break
            lines.append(line)
            length += len(line) + 1
        return "\n".join(lines)

    def _fetch_youtube_captions(
        self,
        *,
        video_id: str,
        url: str,
        language: str,
        max_chars: int,
    ) -> str:
        from youtube_transcript_api import YouTubeTranscriptApi

        languages = tuple(
            dict.fromkeys([language, "en", "en-US", "en-GB"])
        )
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=languages,
            preserve_formatting=True,
        )
        items: list[tuple[float, str]] = []
        for entry in transcript:
            start = float(getattr(entry, "start", 0.0))
            text = str(getattr(entry, "text", "") or "")
            items.append((start, text))
        body = self._format_lines(items, max_chars=max_chars)
        if not body:
            raise RuntimeError("caption transcript was empty")
        return (
            "Video transcript source: youtube captions\n"
            f"URL: {url}\n"
            f"YouTube video id: {video_id}\n"
            f"Language requested: {language}\n\n"
            f"{body}"
        )

    def _transcribe_audio(
        self,
        *,
        url: str,
        language: str,
        max_chars: int,
        previous_errors: list[str],
    ) -> str:
        import yt_dlp
        from faster_whisper import WhisperModel

        model_size = os.getenv("VIDEO_TRANSCRIPT_WHISPER_MODEL", "base")
        device = os.getenv("VIDEO_TRANSCRIPT_WHISPER_DEVICE", "auto")
        compute_type = os.getenv("VIDEO_TRANSCRIPT_WHISPER_COMPUTE_TYPE", "int8")

        with tempfile.TemporaryDirectory(prefix="scp_video_transcript_") as tmp:
            outtmpl = str(Path(tmp) / "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                "logger": self._QuietYtdlpLogger(),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            audio_files = sorted(Path(tmp).glob("*.wav"))
            if not audio_files:
                raise RuntimeError("yt-dlp did not produce a wav file")

            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            segments, info_obj = model.transcribe(
                str(audio_files[0]),
                language=language or None,
                vad_filter=True,
                beam_size=5,
            )
            items = [(float(segment.start), segment.text) for segment in segments]
            body = self._format_lines(items, max_chars=max_chars)
            if not body:
                raise RuntimeError("ASR transcript was empty")

        title = str((info or {}).get("title") or "").strip()
        header = [
            "Video transcript source: faster-whisper ASR",
            f"URL: {url}",
        ]
        if title:
            header.append(f"Title: {title}")
        header.extend(
            [
                f"Language requested: {language}",
                f"Detected language: {getattr(info_obj, 'language', '')}",
                f"Whisper model: {model_size}; device: {device}; compute_type: {compute_type}",
            ]
        )
        if previous_errors:
            header.append("Previous transcript attempts: " + "; ".join(previous_errors))
        return "\n".join(header) + "\n\n" + body


__all__ = ["VideoTranscriptTool"]
