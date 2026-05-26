from __future__ import annotations

from pathlib import Path

from ..models import AttachmentReaderConfig


class OfficeAttachmentReader:
    """OfficeAttachmentReader 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """初始化物件與必要狀態。"""
        self.config = config

    def read_docx(self, file_path: Path) -> str:
        """read_docx 的主要實作。"""
        try:
            from docx import Document
        except ImportError:
            return "DOCX attachment detected, but python-docx is not installed."

        doc = Document(str(file_path))
        lines: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                lines.append(text)
        for table_idx, table in enumerate(doc.tables, start=1):
            lines.append(f"[table {table_idx}]")
            for row in table.rows[: self.config.max_table_rows]:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(lines)

    def read_pptx(self, file_path: Path) -> str:
        """read_pptx 的主要實作。"""
        try:
            from pptx import Presentation
        except ImportError:
            return "PPTX attachment detected, but python-pptx is not installed."

        presentation = Presentation(str(file_path))
        lines: list[str] = []
        for slide_idx, slide in enumerate(presentation.slides, start=1):
            slide_lines: list[str] = []
            for shape in slide.shapes:
                text = getattr(shape, "text", "")
                if text and text.strip():
                    slide_lines.append(" ".join(text.split()))
            if slide_lines:
                lines.append(f"[slide {slide_idx}] " + "\n".join(slide_lines))
        return "\n".join(lines)

    def read_pdf(self, file_path: Path) -> str:
        """read_pdf 的主要實作。"""
        reader_cls = None
        try:
            from pypdf import PdfReader

            reader_cls = PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader

                reader_cls = PdfReader
            except ImportError:
                return "PDF attachment detected, but pypdf/PyPDF2 is not installed."

        reader = reader_cls(str(file_path))
        lines: list[str] = []
        for idx, page in enumerate(reader.pages[: self.config.max_pdf_pages], start=1):
            text = page.extract_text() or ""
            if text.strip():
                lines.append(f"[page {idx}]\n{text.strip()}")
        if len(reader.pages) > self.config.max_pdf_pages:
            lines.append(f"[truncated after {self.config.max_pdf_pages} pages]")
        return "\n\n".join(lines)

