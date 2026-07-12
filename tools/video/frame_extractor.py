from __future__ import annotations

import subprocess
from pathlib import Path

from .config import VideoEvidenceConfig
from .models import FrameItem


class FrameExtractor:
    """
    Extract timestamped frames from a local video file.

    Args:
        - config: Video evidence runtime configuration.

    Returns:
        - FrameExtractor: ffmpeg-backed frame extractor.
    """

    def __init__(self, config: VideoEvidenceConfig) -> None:
        self.config = config

    def extract(self, video_path: Path, output_dir: Path) -> list[FrameItem]:
        """
        Extract JPEG frames at a fixed interval.

        Args:
            - video_path: Local video path.
            - output_dir: Directory for extracted frames.

        Returns:
            - list[FrameItem]: Extracted frame metadata.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        interval = max(float(self.config.frame_interval_sec), 0.5)
        pattern = str(output_dir / "frame_%05d.jpg")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval}",
            "-q:v",
            "3",
            pattern,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        frames: list[FrameItem] = []
        for index, path in enumerate(sorted(output_dir.glob("frame_*.jpg")), start=1):
            frames.append(
                FrameItem(
                    frame_id=path.stem,
                    timestamp_sec=(index - 1) * interval,
                    image_path=path,
                )
            )
        return frames


__all__ = ["FrameExtractor"]
