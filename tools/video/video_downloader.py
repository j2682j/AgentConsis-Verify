from __future__ import annotations

from pathlib import Path

from .config import VideoEvidenceConfig
from .models import VideoDownloadResult


class VideoDownloader:
    """
    Download a remote video into a task-local temporary directory.

    Args:
        - config: Video evidence runtime configuration.

    Returns:
        - VideoDownloader: Downloader backed by yt-dlp.
    """

    def __init__(self, config: VideoEvidenceConfig) -> None:
        self.config = config

    def download(self, url: str, output_dir: Path) -> VideoDownloadResult:
        """
        Download a video URL using a low-resolution format.

        Args:
            - url: Remote video URL.
            - output_dir: Directory for downloaded media.

        Returns:
            - VideoDownloadResult: Download metadata and local path.
        """
        try:
            import yt_dlp
        except ImportError as exc:
            return VideoDownloadResult(
                ok=False,
                url=url,
                error=f"yt-dlp is not installed: {exc}",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(output_dir / "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": self.config.timeout_sec,
            "merge_output_format": "mp4",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            return VideoDownloadResult(
                ok=False,
                url=url,
                error=f"{type(exc).__name__}: {exc}",
            )

        duration = float((info or {}).get("duration") or 0.0)
        if duration and duration > self.config.max_video_seconds:
            return VideoDownloadResult(
                ok=False,
                url=url,
                title=str((info or {}).get("title") or ""),
                video_id=str((info or {}).get("id") or ""),
                duration=duration,
                error=f"video duration {duration:.1f}s exceeds limit {self.config.max_video_seconds}s",
            )

        candidates = sorted(output_dir.glob("*"))
        video_files = [
            path for path in candidates if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
        ]
        if not video_files:
            return VideoDownloadResult(
                ok=False,
                url=url,
                title=str((info or {}).get("title") or ""),
                video_id=str((info or {}).get("id") or ""),
                duration=duration,
                error="yt-dlp did not produce a supported video file",
            )
        return VideoDownloadResult(
            ok=True,
            url=url,
            video_path=video_files[0],
            title=str((info or {}).get("title") or ""),
            video_id=str((info or {}).get("id") or ""),
            duration=duration,
        )


__all__ = ["VideoDownloader"]
