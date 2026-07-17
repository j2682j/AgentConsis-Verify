from __future__ import annotations

from typing import Any, Mapping

from utils.network_utils import normalize_text

from .collection_record import CollectionRecord


class RecordTextSerializer:
    """
    將結構化記錄轉成保留欄位關係的自然語言 passage。

    Args:
     - None.

    Returns:
     - RecordTextSerializer: E5、FAISS 與 Labeler 共用的序列化元件。
    """

    def serialize(
        self,
        record: CollectionRecord | Mapping[str, Any],
        *,
        content_override: str | None = None,
    ) -> str:
        """依固定欄位順序輸出記錄，並略過空欄位。"""
        data = record.fields if isinstance(record, CollectionRecord) else dict(record)
        content = data.get("content", "") if content_override is None else content_override
        authors = data.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        extra_fields = data.get("extra_fields", {})
        if isinstance(extra_fields, (list, tuple)):
            extra_fields = dict(extra_fields)
        lines = [
            self._line("Record Type", data.get("record_type", "")),
            self._line("Title", data.get("title", "")),
            self._line("Authors", "; ".join(str(item) for item in authors if item)),
            self._line("Date", data.get("date", "")),
            self._line("Source", data.get("source", "")),
            self._line("Content Link", data.get("content_url", "")),
            self._line("Language", data.get("language", "")),
            self._line("Country", data.get("country", "")),
        ]
        if isinstance(extra_fields, Mapping):
            lines.extend(
                self._line(str(name), value)
                for name, value in extra_fields.items()
            )
        lines.append(self._line("Content", content))
        return "\n".join(line for line in lines if line)

    def _line(self, label: str, value: Any) -> str:
        cleaned = normalize_text(str(value or ""))
        return f"{label}: {cleaned}" if cleaned else ""


__all__ = ["RecordTextSerializer"]
