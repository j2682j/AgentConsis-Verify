from __future__ import annotations

from dataclasses import dataclass, field
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
    vision_model: str = "qwen3-vl:8b"
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
