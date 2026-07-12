from __future__ import annotations

from .models import FrameItem


class FrameSampler:
    """
    Select a bounded, uniformly spread frame subset.

    Args:
        - max_frames: Maximum frames to retain.

    Returns:
        - FrameSampler: Deterministic frame sampler.
    """

    def __init__(self, *, max_frames: int) -> None:
        self.max_frames = max(1, int(max_frames))

    def sample(self, frames: list[FrameItem]) -> list[FrameItem]:
        """
        Sample frames without changing temporal order.

        Args:
            - frames: Extracted frame list.

        Returns:
            - list[FrameItem]: Sampled frames.
        """
        if len(frames) <= self.max_frames:
            return list(frames)
        if self.max_frames == 1:
            return [frames[len(frames) // 2]]
        last_index = len(frames) - 1
        indexes = {
            round(i * last_index / (self.max_frames - 1))
            for i in range(self.max_frames)
        }
        return [frames[index] for index in sorted(indexes)]


__all__ = ["FrameSampler"]
