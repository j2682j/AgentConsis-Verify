from __future__ import annotations

from typing import Any

from .schema import ToolCandidate, ToolNeed, ToolPlan, ToolPlanStep


class FallbackToolPlanner:
    """
    Build a conservative system plan when SLM planning is unavailable or invalid.
    """

    def plan(
        self,
        *,
        candidates: list[ToolCandidate],
        routing: dict[str, Any] | None = None,
        deterministic_handler_requested: bool = False,
    ) -> ToolPlan:
        candidate_names = {candidate.tool_name for candidate in candidates}
        routing = routing or {}
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
        steps: list[ToolPlanStep] = []
        needs: list[ToolNeed] = []

        if "video_evidence" in candidate_names:
            needs.append(
                ToolNeed(
                    need_type="video_visual",
                    required_capabilities=["youtube_url", "video.visual", "video.frame_analysis"],
                    input_refs=["question.youtube_url"],
                    reason="Question contains video input that should be converted to visual frame evidence.",
                )
            )
            steps.append(
                ToolPlanStep(
                    tool_name="video_evidence",
                    purpose="extract visual frame evidence from video URL",
                    expected_output="timestamped visual evidence",
                )
            )

        if "video_transcript" in candidate_names:
            needs.append(
                ToolNeed(
                    need_type="video_transcript",
                    required_capabilities=["youtube_url", "transcript"],
                    input_refs=["question.youtube_url"],
                    reason="Question contains video input that should be converted to transcript evidence.",
                )
            )
            steps.append(
                ToolPlanStep(
                    tool_name="video_transcript",
                    purpose="extract transcript evidence from video URL",
                    expected_output="timestamped transcript evidence",
                )
            )

        needs_attachment_for_gap = bool(
            gap_missing
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
        if "attachment_reader" in candidate_names and (
            routing.get("use_attachment", True) or needs_attachment_for_gap
        ):
            steps.append(
                ToolPlanStep(
                    tool_name="attachment_reader",
                    purpose=(
                        "recover deterministic handler missing inputs"
                        if needs_attachment_for_gap
                        else "extract attachment evidence"
                    ),
                    expected_output=(
                        ", ".join(sorted(gap_missing))
                        if needs_attachment_for_gap
                        else "attachment context"
                    ),
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
        if "search" in candidate_names and (routing.get("use_search") or needs_search_for_gap):
            steps.append(
                ToolPlanStep(
                    tool_name="search",
                    purpose=(
                        "recover deterministic handler missing inputs"
                        if needs_search_for_gap
                        else "retrieve external factual evidence"
                    ),
                    expected_output=(
                        ", ".join(sorted(gap_missing))
                        if needs_search_for_gap
                        else "search evidence"
                    ),
                )
            )
        if (
            "deterministic_handler" in candidate_names
            and (
                deterministic_handler_requested
                or routing.get("use_deterministic_solver")
                or routing.get("use_python_solver")
            )
        ):
            dependencies = [
                step.tool_name
                for step in steps
                if step.tool_name in {"attachment_reader", "search"}
            ]
            steps.append(
                ToolPlanStep(
                    tool_name="deterministic_handler",
                    purpose="run exact deterministic computation if possible",
                    depends_on=dependencies,
                    expected_output="exact deterministic evidence",
                )
            )

        return ToolPlan(
            requires_tools=bool(steps),
            tool_needs=needs,
            tool_sequence=steps[:3],
            stop_condition="required evidence has been prepared",
            planner_source="fallback",
        )


__all__ = ["FallbackToolPlanner"]
