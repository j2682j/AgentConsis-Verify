from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .formatters import format_attachment_context, truncate_text
from .models import AttachmentReadResult, AttachmentReaderConfig, ParsedAttachmentPayload
from .payload_builder import AttachmentPayloadBuilder
from .profile_builder import AttachmentProfileBuilder
from .readers.archive_reader import ArchiveAttachmentReader
from .readers.code_reader import CodeAttachmentReader
from .readers.excel_reader import ExcelAttachmentReader
from .readers.media_reader import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, MediaAttachmentReader
from .readers.office_reader import OfficeAttachmentReader
from .readers.text_reader import TEXT_EXTENSIONS, TextAttachmentReader


class AttachmentEvidenceBuilder:
    """AttachmentEvidenceBuilder 類別，封裝此模組的資料結構與服務邏輯。"""

    TEXT_EXTENSIONS = TEXT_EXTENSIONS
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
    AUDIO_EXTENSIONS = AUDIO_EXTENSIONS

    def __init__(
        self,
        *,
        max_text_chars: int = 12000,
        max_table_rows: int = 80,
        max_pdf_pages: int = 20,
        python_timeout: int = 20,
        vision_model: str | None = None,
        vision_timeout: int | None = None,
        audio_model_size: str | None = None,
        audio_device: str | None = None,
        audio_compute_type: str | None = None,
        max_zip_members: int = 30,
        max_zip_file_bytes: int = 8 * 1024 * 1024,
        max_zip_total_bytes: int = 40 * 1024 * 1024,
        max_zip_depth: int = 1,
        profile_builder: AttachmentProfileBuilder | None = None,
    ) -> None:
        """初始化物件與必要狀態。"""
        self.config = AttachmentReaderConfig(
            max_text_chars=max_text_chars,
            max_table_rows=max_table_rows,
            max_pdf_pages=max_pdf_pages,
            python_timeout=python_timeout,
            vision_model=vision_model or os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b"),
            vision_timeout=vision_timeout
            or int(os.getenv("OLLAMA_VISION_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "180"))),
            audio_model_size=audio_model_size or os.getenv("FASTER_WHISPER_MODEL_SIZE", "base"),
            audio_device=audio_device or os.getenv("FASTER_WHISPER_DEVICE", "cuda"),
            audio_compute_type=audio_compute_type
            or os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16"),
            max_zip_members=max_zip_members,
            max_zip_file_bytes=max_zip_file_bytes,
            max_zip_total_bytes=max_zip_total_bytes,
            max_zip_depth=max_zip_depth,
        )
        self.text_reader = TextAttachmentReader(self.config)
        self.office_reader = OfficeAttachmentReader(self.config)
        self.excel_reader = ExcelAttachmentReader(self.config)
        self.code_reader = CodeAttachmentReader(self.config)
        self.media_reader = MediaAttachmentReader(self.config)
        self.archive_reader = ArchiveAttachmentReader(self.config, self)
        self.profile_builder = profile_builder or AttachmentProfileBuilder()
        self.payload_builder = AttachmentPayloadBuilder(max_table_rows=max_table_rows)

    def build(self, question: str, attachment: dict[str, Any] | None) -> dict[str, Any]:
        """build 的主要實作。"""
        if not attachment:
            return self._empty()

        file_path = Path(str(attachment.get("file_path") or attachment.get("path") or ""))
        file_name = str(attachment.get("file_name", "") or file_path.name)
        extension = str(attachment.get("extension", "") or file_path.suffix).lower()
        warnings: list[str] = []

        if not file_path.exists() or not file_path.is_file():
            warnings.append("attachment file does not exist or is not a file")
            return self._result(
                context=format_attachment_context(
                    file_name=file_name,
                    file_path=file_path,
                    extension=extension,
                    content="",
                    warnings=warnings,
                ),
                used=False,
                file_path=file_path,
                extension=extension,
                warnings=warnings,
                content="",
                read_ok=False,
            )

        result = self.read_file(question=question, file_path=file_path, extension=extension)
        content = truncate_text(result.content, self.config.max_text_chars)
        parsed_payload = result.parsed_payload or self.payload_builder.build(
            file_path=file_path,
            extension=extension,
            content=content,
            reader=result.reader,
            reader_metadata=result.metadata,
        )
        structured_error = str(
            parsed_payload.native_metadata.get("structured_parse_error") or ""
        ).strip()
        if structured_error:
            result.warnings.append(f"structured payload unavailable: {structured_error}")
        context = format_attachment_context(
            file_name=file_name,
            file_path=file_path,
            extension=extension,
            content=content,
            warnings=result.warnings,
        )
        return self._result(
            context=context,
            used=result.ok,
            file_path=file_path,
            extension=extension,
            warnings=result.warnings,
            reader=result.reader,
            content=content,
            read_ok=result.ok,
            reader_metadata=result.metadata,
            parsed_payload=parsed_payload,
        )

    def read_file(
        self,
        *,
        question: str,
        file_path: Path,
        extension: str | None = None,
        depth: int = 0,
    ) -> AttachmentReadResult:
        """read_file 的主要實作。"""
        extension = (extension or file_path.suffix).lower()
        try:
            if extension in TEXT_EXTENSIONS:
                return AttachmentReadResult(True, "text_reader", self.text_reader.read(file_path, extension))
            if extension == ".docx":
                return AttachmentReadResult(True, "docx_reader", self.office_reader.read_docx(file_path))
            if extension == ".pptx":
                return AttachmentReadResult(True, "pptx_reader", self.office_reader.read_pptx(file_path))
            if extension == ".xlsx":
                return AttachmentReadResult(True, "excel_reader", self.excel_reader.read_xlsx(question, file_path))
            if extension == ".xls":
                return AttachmentReadResult(
                    True, "pandas_xls_reader", self.excel_reader.read_xls(question, file_path)
                )
            if extension == ".pdf":
                return AttachmentReadResult(True, "pdf_reader", self.office_reader.read_pdf(file_path))
            if extension == ".py":
                return AttachmentReadResult(True, "python_reader", self.code_reader.read_python(file_path))
            if extension == ".zip":
                return AttachmentReadResult(
                    True,
                    "zip_reader",
                    self.archive_reader.read_zip(question, file_path, depth=depth),
                )
            if extension in IMAGE_EXTENSIONS:
                return AttachmentReadResult(
                    True, "ollama_vision_reader", self.media_reader.read_image(question, file_path)
                )
            if extension in AUDIO_EXTENSIONS:
                return AttachmentReadResult(
                    True,
                    "faster_whisper_audio_reader",
                    self.media_reader.analyze_audio(question, file_path),
                )

            return AttachmentReadResult(
                False,
                "unsupported_reader",
                f"Unsupported attachment type: {extension or '(none)'}",
                ["unsupported attachment extension"],
            )
        except Exception as exc:
            return AttachmentReadResult(
                False,
                "error_reader",
                "",
                [f"attachment read failed: {type(exc).__name__}: {exc}"],
            )

    def format_nested_file_result(
        self,
        *,
        question: str,
        file_path: Path,
        display_name: str,
        depth: int,
    ) -> str:
        """format_nested_file_result 的主要實作。"""
        extension = file_path.suffix.lower()
        result = self.read_file(
            question=question,
            file_path=file_path,
            extension=extension,
            depth=depth,
        )
        header = f"Nested reader for {display_name} ({extension or '(none)'})"
        content = result.content
        if result.warnings:
            content = "Warnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings)
        return f"{header}\n{truncate_text(content, self.config.max_text_chars)}"

    def _empty(self) -> dict[str, Any]:
        """_empty 的內部輔助實作。"""
        return {
            "context": "",
            "profile": {},
            "parsed_payload": {},
            "used": False,
            "tool_usage": [],
            "metadata": {},
        }

    def _result(
        self,
        *,
        context: str,
        used: bool,
        file_path: Path,
        extension: str,
        warnings: list[str],
        reader: str = "attachment_reader",
        content: str = "",
        read_ok: bool = False,
        reader_metadata: dict[str, Any] | None = None,
        parsed_payload: ParsedAttachmentPayload | None = None,
    ) -> dict[str, Any]:
        """_result 的內部輔助實作。"""
        failed = any("failed" in warning for warning in warnings)
        output_text = context if used else "\n".join(warnings)
        payload = parsed_payload or ParsedAttachmentPayload(
            provenance={
                "source": "attachment_reader",
                "file_path": str(file_path),
                "file_type": extension,
                "reader": reader,
            }
        )
        profile = self.profile_builder.build(
            file_path=file_path,
            extension=extension,
            read_ok=read_ok,
            reader=reader,
            content=content,
            warnings=warnings,
            reader_metadata=reader_metadata or {},
            parsed_payload=payload,
        )
        payload.provenance.update(
            {
                "source": "attachment_reader",
                "file_path": str(file_path),
                "file_type": extension,
                "reader": reader,
                "parse_status": profile.parse_status,
            }
        )
        parsed_payload_dict = {
            **payload.to_dict(),
            "content": content,
            "reader": reader,
            "reader_metadata": dict(reader_metadata or {}),
        }
        return {
            "context": context if used else "",
            "profile": profile.to_dict(),
            "parsed_payload": parsed_payload_dict,
            "used": used,
            "tool_usage": [
                {
                    "ok": used and not failed,
                    "tool_name": "attachment_reader",
                    "output_text": output_text,
                    "raw_result": {
                        "file_path": str(file_path),
                        "file_type": extension,
                        "reader": reader,
                        "warnings": warnings,
                        "profile": profile.to_dict(),
                    },
                    "error": "; ".join(warnings) if warnings and (failed or not used) else None,
                }
            ],
            "metadata": {
                "file_path": str(file_path),
                "file_type": extension,
                "reader": reader,
                "warnings": warnings,
                "profile": profile.to_dict(),
            },
        }
