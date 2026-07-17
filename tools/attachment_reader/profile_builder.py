from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import AttachmentProfile, ParsedAttachmentPayload, TextBlock


class AttachmentProfileBuilder:
    """將附件解析結果整理成不含完整內容的能力與結構摘要。"""

    def __init__(self, *, preview_chars: int = 1800) -> None:
        self.preview_chars = max(400, int(preview_chars))

    def build(
        self,
        *,
        file_path: Path,
        extension: str,
        read_ok: bool,
        reader: str,
        content: str,
        warnings: list[str] | None = None,
        reader_metadata: dict[str, Any] | None = None,
        parsed_payload: ParsedAttachmentPayload | None = None,
    ) -> AttachmentProfile:
        normalized_extension = self._normalize_extension(extension or file_path.suffix)
        normalized_content = str(content or "").strip()
        warning_list = [str(item) for item in (warnings or []) if str(item).strip()]
        payload = parsed_payload or ParsedAttachmentPayload()
        if not payload.has_content() and normalized_content:
            payload.text_blocks.append(TextBlock(text=normalized_content))
        content_types = payload.content_types()
        available_inputs = ["question"]
        if bool(file_path.name) and file_path.exists():
            available_inputs.extend(["attachment", "file_path"])
        available_inputs.extend(payload.available_inputs())
        return AttachmentProfile(
            file_name=file_path.name,
            file_type=normalized_extension,
            parse_status=self._parse_status(
                read_ok=read_ok,
                has_content=bool(normalized_content),
                reader=reader,
                warnings=warning_list,
            ),
            content_types=content_types,
            structure_summary={
                **payload.structure_summary(),
                **self._selected_reader_metadata(reader_metadata or {}),
            },
            content_preview=self._preview(normalized_content),
            available_inputs=available_inputs,
            warnings=warning_list,
        )

    def _parse_status(
        self,
        *,
        read_ok: bool,
        has_content: bool,
        reader: str,
        warnings: list[str],
    ) -> str:
        if reader == "unsupported_reader":
            return "unsupported"
        if not read_ok or not has_content:
            return "failed"
        if warnings:
            return "partial"
        return "success"

    def _selected_reader_metadata(self, reader_metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in reader_metadata.items()
            if key in {"page_count", "sheet_names", "row_count", "column_count", "duration"}
        }

    def _preview(self, content: str) -> str:
        compact = re.sub(r"[ \t]+", " ", str(content or "")).strip()
        if len(compact) <= self.preview_chars:
            return compact
        return compact[: self.preview_chars].rstrip() + "\n[preview truncated]"

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = str(extension or "").strip().lower()
        if normalized and not normalized.startswith("."):
            normalized = "." + normalized
        return normalized


__all__ = ["AttachmentProfileBuilder"]
