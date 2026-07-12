from __future__ import annotations

from .models import FrameAnalysisResult, VideoEvidenceResult


class VideoEvidenceBuilder:
    """
    Convert frame-level observations into compact Stage1 evidence.

    Args:
        - max_items: Maximum evidence lines exposed to Stage1.

    Returns:
        - VideoEvidenceBuilder: Aggregator for visual evidence.
    """

    def __init__(self, *, max_items: int = 5) -> None:
        self.max_items = max(1, max_items)

    def build(
        self,
        *,
        url: str,
        title: str,
        frame_results: list[FrameAnalysisResult],
    ) -> VideoEvidenceResult:
        """
        Aggregate analyzed frames into answer candidates and evidence text.

        Args:
            - url: Source video URL.
            - title: Video title if available.
            - frame_results: Per-frame visual observations.

        Returns:
            - VideoEvidenceResult: Prompt-ready video evidence package.
        """
        successful = [item for item in frame_results if item.ok and (item.evidence or item.answer_value)]
        ranked = sorted(
            successful,
            key=lambda item: (
                item.count if item.count is not None else -1,
                item.confidence,
                -item.timestamp_sec,
            ),
            reverse=True,
        )
        evidence_items: list[dict[str, object]] = []
        for index, item in enumerate(ranked[: self.max_items], start=1):
            timestamp = self._timestamp(item.timestamp_sec)
            value_text = f" Candidate answer: {item.answer_value}." if item.answer_value else ""
            evidence_items.append(
                {
                    "id": f"V{index}",
                    "source_title": title or "Video frame evidence",
                    "timestamp": timestamp,
                    "evidence": f"At {timestamp}, {item.evidence}{value_text}".strip(),
                    "answer_value": item.answer_value,
                    "confidence": item.confidence,
                    "frame_id": item.frame_id,
                }
            )

        answer_candidates: list[dict[str, object]] = []
        if ranked:
            best = ranked[0]
            answer_candidates.append(
                {
                    "answer": best.answer_value or (str(best.count) if best.count is not None else ""),
                    "timestamp": self._timestamp(best.timestamp_sec),
                    "frame_id": best.frame_id,
                    "confidence": best.confidence,
                    "method": "max_frame_observation",
                }
            )

        output_text = self._render(url=url, title=title, evidence_items=evidence_items)
        return VideoEvidenceResult(
            ok=bool(output_text),
            url=url,
            title=title,
            output_text=output_text,
            evidence_items=evidence_items,
            frame_results=frame_results,
            answer_candidates=answer_candidates,
            error="" if output_text else "no useful visual frame evidence",
        )

    def _render(self, *, url: str, title: str, evidence_items: list[dict[str, object]]) -> str:
        if not evidence_items:
            return ""
        lines = [
            "Video Evidence:",
            f"Source Title: {title or 'Remote video'}",
            f"URL: {url}",
        ]
        for item in evidence_items:
            lines.extend(
                [
                    f"[{item['id']}]",
                    f"Source Title: {item['source_title']}",
                    f"Evidence: {item['evidence']}",
                ]
            )
        return "\n".join(lines).strip()

    @staticmethod
    def _timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


__all__ = ["VideoEvidenceBuilder"]
