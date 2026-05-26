from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    """
    將文字裁切到指定長度，並在裁切時附加標記。

    Args:
        - text: 原始文字。
        - max_chars: 最大保留字元數。

    Returns:
        - str: 裁切後文字。
    """
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def compact_single_line(value: Any, default: str = "(blank)") -> str:
    """
    將任意值壓縮成單行文字。

    Args:
        - value: 要轉成單行的值。
        - default: 空值時使用的預設文字。

    Returns:
        - str: 單行文字。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    return " ".join(text.split())


def format_rows(title: str, rows: list[list[str]], *, truncated: bool, max_rows: int) -> str:
    """
    將表格列資料格式化成純文字區塊。

    Args:
        - title: 表格區塊標題。
        - rows: 表格列資料。
        - truncated: 是否已裁切列數。
        - max_rows: 最大顯示列數。

    Returns:
        - str: 格式化後表格文字。
    """
    lines = [title]
    for row in rows:
        lines.append(" | ".join(row))
    if truncated:
        lines.append(f"[showing first {max_rows} rows]")
    return "\n".join(lines)


def format_attachment_context(
    *,
    file_name: str,
    file_path: Any,
    extension: str,
    content: str,
    warnings: list[str],
) -> str:
    """
    將 attachment 讀取內容格式化成 Stage1 evidence context。

    Args:
        - file_name: 附檔名稱。
        - file_path: 附檔路徑。
        - extension: 附檔副檔名。
        - content: reader 抽取出的內容。
        - warnings: 讀取過程中的警告。

    Returns:
        - str: 可放入 prompt 的 attachment context。
    """
    extracted = str(content or "").strip()
    if not extracted:
        extracted = "None"
    return f"Extracted content:\n{extracted}"
