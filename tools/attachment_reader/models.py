from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttachmentReaderConfig:
    """
    保存 attachment reader 的讀取限制與模型設定。

    Args:
        - max_text_chars: attachment context 最大文字長度。
        - max_table_rows: 表格最多讀取列數。
        - max_pdf_pages: PDF 最多讀取頁數。
        - python_timeout: Python 程式附檔分析 timeout 秒數。
        - vision_model: 圖片分析使用的 vision model。
        - vision_timeout: 圖片分析 timeout 秒數。
        - audio_model_size: 語音轉錄模型大小。
        - audio_device: 語音轉錄裝置。
        - audio_compute_type: 語音轉錄 compute type。
        - max_zip_members: ZIP 最多讀取檔案數。
        - max_zip_file_bytes: ZIP 單檔最大大小。
        - max_zip_total_bytes: ZIP 總讀取大小上限。
        - max_zip_depth: ZIP 遞迴讀取深度上限。

    Returns:
        - AttachmentReaderConfig: attachment reader 設定物件。
    """

    max_text_chars: int = 12000
    max_table_rows: int = 80
    max_pdf_pages: int = 20
    python_timeout: int = 20
    vision_model: str = "qwen3-vl:4b"
    vision_timeout: int = 180
    audio_model_size: str = "base"
    audio_device: str = "cuda"
    audio_compute_type: str = "float16"
    max_zip_members: int = 30
    max_zip_file_bytes: int = 8 * 1024 * 1024
    max_zip_total_bytes: int = 40 * 1024 * 1024
    max_zip_depth: int = 1


@dataclass
class AttachmentReadResult:
    """
    保存單一 attachment reader 的讀取結果。

    Args:
        - ok: 是否成功讀取。
        - reader: 使用的 reader 名稱。
        - content: 讀取出的文字內容。
        - warnings: 讀取過程中的警告訊息。
        - metadata: reader 產生的額外 metadata。

    Returns:
        - AttachmentReadResult: attachment 讀取結果。
    """

    ok: bool
    reader: str
    content: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parsed_payload: ParsedAttachmentPayload | None = None


@dataclass
class TextBlock:
    text: str
    section: str = ""
    page: int | None = None
    block_type: str = "paragraph"


@dataclass
class TablePayload:
    name: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    cell_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    truncated: bool = False


@dataclass
class ListPayload:
    title: str = ""
    items: list[str] = field(default_factory=list)


@dataclass
class CoordinateRecord:
    identifier: str
    x: float
    y: float
    z: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationRecord:
    source: str
    relation: str
    target: str


@dataclass
class VisualBlock:
    text: str = ""
    region: str = ""
    object_type: str = "vision"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedAttachmentPayload:
    """保存附件解析後可驗證、可回溯的結構化內容。"""

    schema_version: str = "1.0"
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TablePayload] = field(default_factory=list)
    lists: list[ListPayload] = field(default_factory=list)
    coordinates: list[CoordinateRecord] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)
    visual_blocks: list[VisualBlock] = field(default_factory=list)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    native_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_content(self) -> bool:
        return any(
            (
                self.text_blocks,
                self.tables,
                self.lists,
                self.coordinates,
                self.relations,
                self.visual_blocks,
                self.semantic_facts,
            )
        )

    def content_types(self) -> list[str]:
        values: list[str] = []
        attachment_kind = str(self.native_metadata.get("attachment_kind") or "")
        if self.text_blocks:
            values.append("text")
        if self.tables:
            values.append("table")
        if self.lists:
            values.append("list")
        if self.coordinates:
            values.append("coordinates")
        if self.relations:
            values.append("relations")
        if self.visual_blocks:
            values.append("image")
            if any(block.text.strip() for block in self.visual_blocks):
                values.append("ocr_text")
        if self.semantic_facts:
            values.append("semantic_facts")
        if attachment_kind in {"audio", "video", "archive", "code"}:
            values.append(attachment_kind)
        if attachment_kind in {"audio", "video"} and self.text_blocks:
            values.append("transcript")
        return values

    def available_inputs(self) -> list[str]:
        values: list[str] = []
        if self.has_content():
            values.extend(["attachment_context", "source_text"])
        if self.tables:
            values.append("table")
            if any(table.rows for table in self.tables):
                values.append("rows")
            if any(table.columns for table in self.tables):
                values.append("columns")
            if any(table.cell_metadata for table in self.tables):
                values.append("cell_metadata")
        if self.lists and any(item.items for item in self.lists):
            values.append("list_items")
        if self.coordinates:
            values.append("coordinates")
        if self.relations:
            values.extend(["relations", "edges"])
        if self.visual_blocks:
            values.append("image")
            if any(block.text.strip() for block in self.visual_blocks):
                values.append("ocr_text")
            visual_keys = {
                key
                for block in self.visual_blocks
                for key, value in block.attributes.items()
                if value not in (None, "", [], {})
            }
            for key in ("numbers", "grid", "candidate_words", "objects", "colors"):
                if key in visual_keys:
                    values.append(key)
        if self.semantic_facts:
            values.append("semantic_facts")
        attachment_kind = str(self.native_metadata.get("attachment_kind") or "")
        if attachment_kind in {"audio", "video"} and self.text_blocks:
            values.append("transcript")
        if attachment_kind == "archive":
            values.append("nested_files")
        if attachment_kind == "video" and self.visual_blocks:
            values.append("frames")
        return list(dict.fromkeys(values))

    def structure_summary(self) -> dict[str, Any]:
        return {
            "text_block_count": len(self.text_blocks),
            "table_count": len(self.tables),
            "row_count": sum(len(table.rows) for table in self.tables),
            "styled_cell_count": sum(len(table.cell_metadata) for table in self.tables),
            "list_count": len(self.lists),
            "coordinate_count": len(self.coordinates),
            "relation_count": len(self.relations),
            "visual_block_count": len(self.visual_blocks),
            "semantic_fact_count": len(self.semantic_facts),
        }


@dataclass
class AttachmentProfile:
    """描述附件完成通用解析後可供規劃與工具使用的能力摘要。"""

    file_name: str = ""
    file_type: str = ""
    parse_status: str = "failed"
    content_types: list[str] = field(default_factory=list)
    structure_summary: dict[str, Any] = field(default_factory=dict)
    content_preview: str = ""
    available_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AttachmentProfile",
    "AttachmentReadResult",
    "AttachmentReaderConfig",
    "CoordinateRecord",
    "ListPayload",
    "ParsedAttachmentPayload",
    "RelationRecord",
    "TablePayload",
    "TextBlock",
    "VisualBlock",
]
