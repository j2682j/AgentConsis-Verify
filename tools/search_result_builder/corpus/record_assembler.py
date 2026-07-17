from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable
from urllib.parse import urljoin

from utils.network_utils import normalize_text

from .collection_record import CollectionRecord


class RecordAssembler:
    """
    正規化集合記錄並在不破壞欄位關係的前提下合併重複項目。

    Args:
     - None.

    Returns:
     - RecordAssembler: 結構化記錄的正規化與去重元件。
    """

    _AUTHOR_SPLIT_RE = re.compile(r"\s*(?:;|\||\band\b|\s+&\s+)\s*", re.I)

    def assemble(
        self,
        records: Iterable[CollectionRecord],
        *,
        parent_url: str,
    ) -> list[CollectionRecord]:
        """正規化、驗證並合併同一集合頁中的記錄。"""
        assembled: list[CollectionRecord] = []
        positions: dict[str, int] = {}
        for record in records:
            normalized = self.normalize(record, parent_url=parent_url)
            if not self.is_valid(normalized):
                continue
            key = self.identity_key(normalized)
            if key in positions:
                index = positions[key]
                assembled[index] = self.merge(assembled[index], normalized)
                continue
            positions[key] = len(assembled)
            assembled.append(normalized)
        return assembled

    def normalize(
        self,
        record: CollectionRecord,
        *,
        parent_url: str,
    ) -> CollectionRecord:
        """清理文字、作者、網址與額外欄位。"""
        base_url = normalize_text(record.parent_url) or normalize_text(parent_url)
        authors: list[str] = []
        seen_authors: set[str] = set()
        for value in record.authors:
            for author in self._AUTHOR_SPLIT_RE.split(normalize_text(value)):
                author = re.sub(r"^by\s+", "", author, flags=re.I).strip()
                key = author.casefold()
                if author and key not in seen_authors:
                    authors.append(author)
                    seen_authors.add(key)
        extras: list[tuple[str, str]] = []
        seen_fields: set[str] = set()
        for name, value in record.extra_fields:
            clean_name = normalize_text(name)
            clean_value = normalize_text(value)
            key = clean_name.casefold()
            if not clean_name or not clean_value or key in seen_fields:
                continue
            extras.append((clean_name, clean_value))
            seen_fields.add(key)
        return replace(
            record,
            record_type=normalize_text(record.record_type).casefold() or "database_row",
            title=normalize_text(record.title),
            authors=tuple(authors),
            date=normalize_text(record.date),
            source=normalize_text(record.source),
            content_url=self._absolute_url(base_url, record.content_url),
            language=normalize_text(record.language),
            country=normalize_text(record.country),
            content=normalize_text(record.content),
            parent_url=base_url,
            extra_fields=tuple(extras),
            extraction_method=normalize_text(record.extraction_method),
        )

    def is_valid(self, record: CollectionRecord) -> bool:
        """以最小 schema 契約確認記錄具有識別欄位與關聯內容。"""
        identity = bool(record.title or record.content_url)
        relation = bool(
            record.authors
            or record.date
            or record.source
            or (
                record.content_url
                and record.content_url.casefold() != record.parent_url.casefold()
            )
            or record.language
            or record.country
            or record.content
            or record.extra_fields
        )
        return identity and relation

    def identity_key(self, record: CollectionRecord) -> str:
        """建立不會混合相同標題不同年份記錄的穩定識別鍵。"""
        parts = [record.record_type, record.title, record.date, record.content_url]
        if not record.content_url:
            parts.extend(record.authors)
            parts.append(record.source)
        key = "|".join(normalize_text(part).casefold() for part in parts)
        if key.strip("|"):
            return key
        return "|".join(
            [record.parent_url.casefold(), record.content.casefold()[:240]]
        )

    def merge(
        self,
        left: CollectionRecord,
        right: CollectionRecord,
    ) -> CollectionRecord:
        """只在識別鍵相同時補齊缺少欄位。"""
        authors = tuple(dict.fromkeys([*left.authors, *right.authors]))
        extras = tuple(dict([*left.extra_fields, *right.extra_fields]).items())
        return replace(
            left,
            title=left.title or right.title,
            authors=authors,
            date=left.date or right.date,
            source=left.source or right.source,
            content_url=left.content_url or right.content_url,
            language=left.language or right.language,
            country=left.country or right.country,
            content=self._longer(left.content, right.content),
            extra_fields=extras,
            extraction_method=left.extraction_method or right.extraction_method,
        )

    def _absolute_url(self, parent_url: str, value: str) -> str:
        url = normalize_text(value)
        if not url:
            return ""
        return urljoin(parent_url, url)

    def _longer(self, left: str, right: str) -> str:
        return right if len(right) > len(left) else left


__all__ = ["RecordAssembler"]
