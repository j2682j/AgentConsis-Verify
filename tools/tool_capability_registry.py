from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from utils.network_utils import normalize_text

from .tool_contract import ToolCapabilitySpec


class ToolCapabilityRegistry:
    """
    根據 capability need 與目前可用輸入匹配工具。

    Args:
     - specs: 可覆蓋預設工具能力宣告的 spec 清單。

    Returns:
     - ToolCapabilityRegistry: Hybrid Tool Planner 使用的 capability matcher。

    """

    VIDEO_URL_RE = re.compile(
        r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[^\s)>\"]+",
        re.IGNORECASE,
    )
    REMOTE_VIDEO_RE = re.compile(r"https?://[^\s)>\"]+\.(?:mp4|mov|mkv|webm)(?:\?[^\s)]*)?", re.IGNORECASE)
    MEDIA_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}
    TABLE_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".json", ".xml", ".zip"}
    VISUAL_VIDEO_TERMS = {
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
    }
    TRANSCRIPT_VIDEO_TERMS = {"transcript", "caption", "subtitles", "said", "spoken", "says"}

    def __init__(self, specs: Iterable[ToolCapabilitySpec] | None = None) -> None:
        self.specs: dict[str, ToolCapabilitySpec] = {}
        for spec in self.default_specs():
            self.register(spec)
        for spec in specs or []:
            self.register(spec)

    @classmethod
    def default_specs(cls) -> list[ToolCapabilitySpec]:
        return [
            ToolCapabilitySpec(
                tool_name="video_evidence",
                capabilities=["video", "youtube_url", "video.visual", "video.frame_analysis", "vision", "evidence_text"],
                input_types=["question", "question.youtube_url", "youtube_url", "remote_video_url"],
                output_types=["visual_evidence", "timestamped_frames", "evidence_text"],
                required_inputs=["youtube_url"],
                optional_inputs=["question", "answer_role", "max_frames"],
                priority=98,
            ),
            ToolCapabilitySpec(
                tool_name="video_transcript",
                capabilities=["video", "youtube_url", "transcript", "video.transcript", "speech_to_text"],
                input_types=["question", "question.youtube_url", "youtube_url", "remote_video_url"],
                output_types=["transcript", "timestamped_text", "evidence_text"],
                required_inputs=["youtube_url"],
                optional_inputs=["language", "max_chars", "allow_asr"],
                priority=95,
            ),
            ToolCapabilitySpec(
                tool_name="attachment_reader",
                capabilities=[
                    "attachment",
                    "text_extraction",
                    "pdf",
                    "image",
                    "ocr",
                    "table",
                    "audio_file",
                    "video_file",
                    "document",
                ],
                input_types=["attachment", "attachment.file_path", "local_file"],
                output_types=["extracted_text", "metadata", "evidence_text"],
                required_inputs=["file_path"],
                priority=85,
            ),
            ToolCapabilitySpec(
                tool_name="deterministic_handler",
                capabilities=[
                    "deterministic",
                    "calculator",
                    "table_reasoning",
                    "graph_search",
                    "unit_conversion",
                    "exact_solver",
                    "puzzle",
                ],
                input_types=["question", "attachment_context", "search_context"],
                output_types=["exact_answer", "deterministic_evidence"],
                required_inputs=["question"],
                priority=75,
            ),
            ToolCapabilitySpec(
                tool_name="search",
                capabilities=["web_search", "factual_search", "external_evidence", "web_page"],
                input_types=["question", "query"],
                output_types=["web_evidence", "source_chunks"],
                required_inputs=["question"],
                priority=60,
            ),
        ]

    def register(self, spec: ToolCapabilitySpec) -> None:
        if spec.tool_name:
            self.specs[spec.tool_name] = spec

    def infer_needs(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        從題目、附件與 routing hints 產生薄的 capability needs。

        Args:
         - question: 原始問題。
         - attachment: GAIA attachment metadata。
         - routing: SystemRoutingContract 的提示。

        Returns:
         - list[ToolNeed]: 可交給 registry matching 的工具需求。

        """
        from .tool_planner.schema import ToolNeed

        text = normalize_text(question)
        routing = routing or {}
        attachment = attachment or {}
        needs: list[ToolNeed] = []

        if self.extract_video_url(text):
            if self._needs_visual_video(text):
                needs.append(
                    ToolNeed(
                        need_type="video_visual",
                        required_capabilities=["youtube_url", "video.visual", "video.frame_analysis"],
                        input_refs=["question.youtube_url"],
                        reason="Question asks about visible video content that requires frame evidence.",
                    )
                )
            if self._needs_transcript_video(text):
                needs.append(
                    ToolNeed(
                        need_type="video_transcript",
                        required_capabilities=["youtube_url", "transcript"],
                        input_refs=["question.youtube_url"],
                        reason="Question asks for speech, captions, or transcript evidence.",
                    )
                )
            if not self._needs_visual_video(text) and not self._needs_transcript_video(text):
                needs.append(
                    ToolNeed(
                        need_type="video_visual",
                        required_capabilities=["youtube_url", "video.visual", "video.frame_analysis"],
                        input_refs=["question.youtube_url"],
                        reason="Question contains a remote video URL and asks for video evidence.",
                    )
                )
            needs.append(
                ToolNeed(
                    need_type="video_fallback_search",
                    required_capabilities=["web_search", "external_evidence"],
                    input_refs=["question.youtube_url"],
                    reason="Search may provide metadata fallback for remote video tasks.",
                )
            )

        extension = self._attachment_extension(attachment)
        if attachment:
            capabilities = ["attachment", "text_extraction"]
            if extension in self.MEDIA_EXTENSIONS:
                capabilities = ["attachment", "video_file" if extension in {".mp4", ".mov", ".mkv", ".webm"} else "audio_file", "text_extraction"]
            elif extension in self.TABLE_EXTENSIONS:
                capabilities = ["attachment", "table", "text_extraction"]
            elif extension in self.IMAGE_EXTENSIONS:
                capabilities = ["attachment", "image", "ocr"]
            elif extension in self.DOCUMENT_EXTENSIONS:
                capabilities = ["attachment", "document", "text_extraction"]
            needs.append(
                ToolNeed(
                    need_type="attachment",
                    required_capabilities=capabilities,
                    input_refs=["attachment.file_path"],
                    reason="Question includes an attachment that must be converted to evidence.",
                )
            )

        if routing.get("use_deterministic_solver") or routing.get("use_python_solver"):
            needs.append(
                ToolNeed(
                    need_type="deterministic",
                    required_capabilities=["deterministic", "exact_solver"],
                    input_refs=["question", "attachment_context", "search_context"],
                    reason="Routing indicates exact deterministic computation may be required.",
                )
            )

        if routing.get("use_search") and routing.get("search_allowed") is not False:
            needs.append(
                ToolNeed(
                    need_type="search",
                    required_capabilities=["web_search", "external_evidence"],
                    input_refs=["question"],
                    reason="Routing indicates external factual evidence may be required.",
                )
            )

        return self._dedupe_needs(needs)

    def match_steps(
        self,
        *,
        needs: Iterable[Any],
        candidate_tool_names: Iterable[str] | None = None,
        available_inputs: Iterable[str] | None = None,
    ) -> list[Any]:
        """
        將 capability needs 轉成可執行工具步驟。

        Args:
         - needs: Planner 或系統推得的 capability needs。
         - candidate_tool_names: 目前 planner 允許使用的工具名稱。
         - available_inputs: 本任務目前可用的輸入型態。

        Returns:
         - list[ToolPlanStep]: 依 priority 排序且去重的工具步驟。

        """
        from .tool_planner.schema import ToolPlanStep

        allowed = {str(name) for name in candidate_tool_names or [] if str(name)}
        available = {self._key(item) for item in available_inputs or [] if str(item)}
        matched: list[tuple[int, ToolPlanStep]] = []
        seen: set[str] = set()
        for need in needs:
            spec = self._best_spec_for_need(need, allowed_tools=allowed, available_inputs=available)
            if spec is None or spec.tool_name in seen:
                continue
            seen.add(spec.tool_name)
            matched.append(
                (
                    -spec.priority,
                    ToolPlanStep(
                        tool_name=spec.tool_name,
                        purpose=need.reason or f"satisfy {need.need_type} capability need",
                        depends_on=[],
                        expected_output=", ".join(spec.output_types[:3]),
                    ),
                )
            )
        matched.sort(key=lambda item: item[0])
        return [step for _, step in matched]

    def available_inputs(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
    ) -> list[str]:
        inputs = ["question"]
        if self.extract_video_url(question):
            inputs.extend(["question.youtube_url", "youtube_url", "remote_video_url"])
        if attachment:
            inputs.extend(["attachment", "attachment.file_path", "local_file"])
        return inputs

    @classmethod
    def extract_video_url(cls, question: str) -> str:
        text = str(question or "")
        match = cls.VIDEO_URL_RE.search(text) or cls.REMOTE_VIDEO_RE.search(text)
        return match.group(0).rstrip(".,)") if match else ""

    def _best_spec_for_need(
        self,
        need: Any,
        *,
        allowed_tools: set[str],
        available_inputs: set[str],
    ) -> ToolCapabilitySpec | None:
        required = {self._key(item) for item in need.required_capabilities if self._key(item)}
        input_refs = {self._key(item) for item in need.input_refs if self._key(item)}
        best: tuple[int, ToolCapabilitySpec] | None = None
        for spec in self.specs.values():
            if allowed_tools and spec.tool_name not in allowed_tools:
                continue
            capabilities = {self._key(item) for item in spec.capabilities}
            if required and not required.issubset(capabilities):
                continue
            spec_inputs = {self._key(item) for item in spec.input_types}
            if input_refs and available_inputs and not (input_refs & spec_inputs or spec_inputs & available_inputs):
                continue
            score = spec.priority + len(required & capabilities) * 10
            if best is None or score > best[0]:
                best = (score, spec)
        return best[1] if best else None

    def _dedupe_needs(self, needs: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for need in needs:
            key = f"{need.need_type}:{','.join(sorted(self._key(item) for item in need.required_capabilities))}"
            if key in seen:
                continue
            seen.add(key)
            result.append(need)
        return result

    def _attachment_extension(self, attachment: dict[str, Any]) -> str:
        extension = str(attachment.get("extension", "") or "").strip().lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if extension:
            return extension
        file_path = str(attachment.get("file_path", "") or attachment.get("path", "") or "")
        return Path(file_path).suffix.lower()

    def _key(self, value: Any) -> str:
        return normalize_text(str(value or "")).casefold().strip()

    def _needs_visual_video(self, text: str) -> bool:
        lowered = normalize_text(text).casefold()
        return any(term in lowered for term in self.VISUAL_VIDEO_TERMS)

    def _needs_transcript_video(self, text: str) -> bool:
        lowered = normalize_text(text).casefold()
        return any(term in lowered for term in self.TRANSCRIPT_VIDEO_TERMS)


__all__ = ["ToolCapabilityRegistry"]
