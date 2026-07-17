from __future__ import annotations

import re
from typing import Any

from .schema import ToolCandidate


class ToolCandidateRouter:
    """
    Build the candidate tool set available to the hybrid planner.
    """

    def route(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
        deterministic_handler_available: bool = True,
    ) -> list[ToolCandidate]:
        routing = routing or {}
        attachment = attachment or {}
        needs_video = self._has_video_url(question)
        needs_visual_video = self._needs_visual_video(question)
        needs_transcript_video = self._needs_transcript_video(question)
        deterministic_gap = (
            routing.get("deterministic_tool_gap")
            if isinstance(routing.get("deterministic_tool_gap"), dict)
            else {}
        )
        gap_missing = {
            str(item or "").strip()
            for item in deterministic_gap.get("missing_inputs", []) or []
            if str(item or "").strip()
        }
        candidates: list[ToolCandidate] = []

        if needs_video:
            if needs_visual_video or not needs_transcript_video:
                candidates.append(
                    ToolCandidate(
                        tool_name="video_evidence",
                        capability="Extract visual frame evidence from YouTube or remote video URLs.",
                        priority_hint="Run first when the question asks about visible video content, objects, counts, or camera evidence.",
                        required=True,
                    )
                )
            if needs_transcript_video:
                candidates.append(
                    ToolCandidate(
                        tool_name="video_transcript",
                        capability="Extract transcript evidence from YouTube or remote video URLs.",
                        priority_hint="Run when the question asks about speech, captions, subtitles, or transcript content.",
                        required=True,
                    )
                )

        needs_attachment_for_gap = bool(
            attachment
            and gap_missing
            & {
                "table_rows",
                "source_text",
                "grid",
                "candidate_words",
                "edges",
                "date_values",
                "numbers",
                "list_items",
                "quoted_or_inline_text",
                "two_coordinate_pairs",
            }
        )

        if attachment:
            candidates.append(
                ToolCandidate(
                    tool_name="attachment_reader",
                    capability="Read attachments and extract text, tables, media metadata, or structured context.",
                    priority_hint=(
                        deterministic_gap.get("next_action_hint")
                        if needs_attachment_for_gap
                        else "Run first when attachment content is needed."
                    ),
                    required=bool(routing.get("use_attachment", True) or needs_attachment_for_gap),
                )
            )

        needs_search_for_gap = bool(
            gap_missing
            & {
                "source_text",
                "date_values",
                "numbers",
                "matching_text",
                "connected_path",
            }
        )

        search_allowed = routing.get("search_allowed") is not False
        if (routing.get("use_search") and search_allowed) or needs_search_for_gap:
            candidates.append(
                ToolCandidate(
                    tool_name="search",
                    capability="Retrieve external factual evidence from the web.",
                    priority_hint=(
                        deterministic_gap.get("next_action_hint")
                        if needs_search_for_gap
                        else "Use for open-world facts not contained in the question or attachment."
                    ),
                    required=needs_search_for_gap and not attachment,
                )
            )

        if deterministic_handler_available:
            candidates.append(
                ToolCandidate(
                    tool_name="deterministic_handler",
                    capability=(
                        "Run exact computation over tables, graphs, coordinates, dates, units, strings, "
                        "or numeric expressions after required data is available."
                    ),
                    priority_hint=(
                        "Use after missing inputs are recovered: "
                        + ", ".join(sorted(gap_missing))
                        if gap_missing
                        else "Use after attachment/search data is available when exact computation is needed."
                    ),
                )
            )

        return candidates

    def _has_video_url(self, question: str) -> bool:
        return bool(
            re.search(
                r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[^\s)>\"]+",
                str(question or ""),
                flags=re.IGNORECASE,
            )
            or re.search(
                r"https?://[^\s)>\"]+\.(?:mp4|mov|mkv|webm)(?:\?[^\s)]*)?",
                str(question or ""),
                flags=re.IGNORECASE,
            )
        )

    def _needs_visual_video(self, question: str) -> bool:
        lowered = str(question or "").lower()
        return any(
            term in lowered
            for term in (
                "camera",
                "visible",
                "shown",
                "seen",
                "watch",
                "frame",
                "simultaneously",
                "appears",
                "appear",
                "highest number",
            )
        )

    def _needs_transcript_video(self, question: str) -> bool:
        lowered = str(question or "").lower()
        return any(
            term in lowered
            for term in ("transcript", "caption", "subtitles", "said", "spoken", "says")
        )


__all__ = ["ToolCandidateRouter"]
