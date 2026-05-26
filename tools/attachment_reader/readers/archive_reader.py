from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import AttachmentReaderConfig

if TYPE_CHECKING:
    from ..evidence_builder import AttachmentEvidenceBuilder


class ArchiveAttachmentReader:
    """ArchiveAttachmentReader 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(
        self,
        config: AttachmentReaderConfig,
        dispatcher: AttachmentEvidenceBuilder,
    ) -> None:
        """初始化物件與必要狀態。"""
        self.config = config
        self.dispatcher = dispatcher

    def read_zip(self, question: str, file_path: Path, depth: int = 0) -> str:
        """read_zip 的主要實作。"""
        if depth > self.config.max_zip_depth:
            return f"ZIP archive skipped because nested zip depth exceeded {self.config.max_zip_depth}."

        with zipfile.ZipFile(file_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            listed_names = [info.filename for info in infos[:100]]
            selected_infos, warnings = self._select_safe_zip_members(infos)

            sections = [
                "ZIP archive:",
                f"- file_count: {len(infos)}",
                f"- processed_file_count: {len(selected_infos)}",
                "ZIP contents:",
                *[f"- {name}" for name in listed_names],
            ]
            if len(infos) > len(listed_names):
                sections.append(f"- [showing first {len(listed_names)} entries]")
            if warnings:
                sections.append("ZIP warnings:")
                sections.extend(f"- {warning}" for warning in warnings)

            if not selected_infos:
                return "\n".join(sections)

            with tempfile.TemporaryDirectory(prefix="gaia_attachment_zip_") as tmp_dir:
                tmp_root = Path(tmp_dir).resolve()
                extracted_sections: list[str] = []
                for info in selected_infos:
                    try:
                        extracted_path = self._extract_zip_member_safely(
                            archive=archive,
                            info=info,
                            tmp_root=tmp_root,
                        )
                        extracted_content = self.dispatcher.format_nested_file_result(
                            question=question,
                            file_path=extracted_path,
                            display_name=info.filename,
                            depth=depth + 1,
                        )
                    except Exception as exc:
                        extracted_content = f"[ERROR] {type(exc).__name__}: {exc}"

                    extracted_sections.append(
                        f"Extracted file: {info.filename}\n"
                        f"- size_bytes: {info.file_size}\n"
                        f"{extracted_content}".strip()
                    )

            if extracted_sections:
                sections.append("Extracted file evidence:")
                sections.append("\n\n".join(extracted_sections))
            return "\n".join(sections)

    def _select_safe_zip_members(
        self, infos: list[zipfile.ZipInfo]
    ) -> tuple[list[zipfile.ZipInfo], list[str]]:
        """_select_safe_zip_members 的內部輔助實作。"""
        selected: list[zipfile.ZipInfo] = []
        warnings: list[str] = []
        total_size = 0

        for info in infos:
            if len(selected) >= self.config.max_zip_members:
                warnings.append(
                    f"skipped remaining files after max_zip_members={self.config.max_zip_members}"
                )
                break

            normalized_name = str(info.filename or "").replace("\\", "/")
            if self._is_unsafe_zip_name(normalized_name):
                warnings.append(f"skipped unsafe zip path: {info.filename}")
                continue
            if info.file_size > self.config.max_zip_file_bytes:
                warnings.append(
                    f"skipped {info.filename}: file_size={info.file_size} exceeds "
                    f"max_zip_file_bytes={self.config.max_zip_file_bytes}"
                )
                continue
            if total_size + info.file_size > self.config.max_zip_total_bytes:
                warnings.append(
                    f"skipped {info.filename}: total extracted bytes would exceed "
                    f"max_zip_total_bytes={self.config.max_zip_total_bytes}"
                )
                continue

            selected.append(info)
            total_size += info.file_size

        return selected, warnings

    def _is_unsafe_zip_name(self, name: str) -> bool:
        """_is_unsafe_zip_name 的內部輔助實作。"""
        normalized = name.replace("\\", "/")
        return (
            not normalized
            or normalized.startswith("/")
            or ":" in normalized.split("/", 1)[0]
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        )

    def _extract_zip_member_safely(
        self,
        *,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        tmp_root: Path,
    ) -> Path:
        """_extract_zip_member_safely 的內部輔助實作。"""
        normalized_name = str(info.filename or "").replace("\\", "/")
        if self._is_unsafe_zip_name(normalized_name):
            raise ValueError(f"unsafe zip path: {info.filename}")

        target_path = (tmp_root / normalized_name).resolve()
        try:
            target_path.relative_to(tmp_root)
        except ValueError as exc:
            raise ValueError(f"zip path escapes extraction root: {info.filename}") from exc

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target_path.open("wb") as target:
            copied = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > self.config.max_zip_file_bytes:
                    raise ValueError(
                        f"zip member exceeded max_zip_file_bytes={self.config.max_zip_file_bytes}: "
                        f"{info.filename}"
                    )
                target.write(chunk)
        return target_path

