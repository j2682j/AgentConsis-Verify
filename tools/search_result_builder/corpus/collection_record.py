from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CollectionRecord:
    """
    表示集合頁面中的一筆不可任意拆散的結構化記錄。

    Args:
     - record_type: 記錄類型，例如 publication、article 或 database_row。
     - title: 論文、文章或資料列的主要名稱。
     - authors: 與此記錄直接關聯的作者集合。
     - date: 出版年份、日期或資料庫中的時間欄位。
     - source: 期刊、出版者、資料庫或來源名稱。
     - content_url: 指向記錄正文或詳細頁面的連結。
     - language: 記錄明示的語言。
     - country: 記錄明示的國家或地區。
     - content: 摘要、描述或與記錄直接關聯的正文。
     - parent_url: 發現此記錄的集合頁面 URL。
     - extra_fields: 無法映射到標準欄位但仍需保留的欄名和值。

    Returns:
     - CollectionRecord: 可正規化、序列化並建立 passage embedding 的記錄。
    """

    record_type: str
    title: str = ""
    authors: tuple[str, ...] = ()
    date: str = ""
    source: str = ""
    content_url: str = ""
    language: str = ""
    country: str = ""
    content: str = ""
    parent_url: str = ""
    extra_fields: tuple[tuple[str, str], ...] = ()
    extraction_method: str = ""
    record_id: str = ""
    retrieved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """將記錄轉成可寫入 JSONL 的資料。"""
        payload = asdict(self)
        payload["authors"] = list(self.authors)
        payload["extra_fields"] = dict(self.extra_fields)
        return payload

    @property
    def fields(self) -> dict[str, Any]:
        """回傳下游檢索與診斷需要的結構化欄位。"""
        return {
            "record_type": self.record_type,
            "title": self.title,
            "authors": list(self.authors),
            "date": self.date,
            "source": self.source,
            "content_url": self.content_url,
            "language": self.language,
            "country": self.country,
            "content": self.content,
            "parent_url": self.parent_url,
            "extra_fields": dict(self.extra_fields),
        }


@dataclass(frozen=True)
class CollectionExtractionResult:
    """保存集合記錄抽取結果與最小診斷資訊。"""

    records: list[CollectionRecord] = field(default_factory=list)
    methods: tuple[str, ...] = ()


__all__ = ["CollectionExtractionResult", "CollectionRecord"]
