from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable
from urllib.parse import urljoin

from utils.network_utils import normalize_text

from .collection_record import CollectionExtractionResult, CollectionRecord
from .record_assembler import RecordAssembler


class CollectionRecordExtractor:
    """
    從 JSON-LD、HTML 表格與重複 DOM 區塊抽取結構化集合記錄。

    Args:
     - max_records: 單一頁面最多保留的集合記錄數。
     - assembler: 記錄正規化與去重元件。

    Returns:
     - CollectionRecordExtractor: 集合頁結構辨識與記錄抽取元件。
    """

    _SUPPORTED_TYPES = {
        "article": "article",
        "newsarticle": "article",
        "blogposting": "article",
        "report": "article",
        "scholarlyarticle": "publication",
        "creativework": "publication",
        "thesis": "publication",
        "dataset": "database_row",
    }
    _FIELD_ALIASES = {
        "title": {
            "title", "article title", "paper title", "publication title",
            "headline", "work", "name",
        },
        "authors": {"author", "authors", "creator", "creators", "writer", "writers"},
        "date": {
            "date", "year", "published", "publication date", "date published",
            "release date", "publication year",
        },
        "source": {"source", "journal", "venue", "publisher", "database", "publication"},
        "content_url": {"url", "link", "content link", "full text", "doi", "website"},
        "language": {"language", "languages", "lang"},
        "country": {"country", "nation", "region", "territory"},
        "content": {"content", "abstract", "summary", "description", "excerpt"},
    }

    def __init__(
        self,
        *,
        max_records: int = 120,
        assembler: RecordAssembler | None = None,
    ) -> None:
        self.max_records = max(1, max_records)
        self.assembler = assembler or RecordAssembler()

    def extract(
        self,
        html: str,
        *,
        parent_url: str = "",
        source_title: str = "",
        source_kind: str = "web",
    ) -> CollectionExtractionResult:
        """抽取頁面中的記錄，無結構化集合時回傳空集合。"""
        raw = str(html or "")
        if "<" not in raw or ">" not in raw:
            return CollectionExtractionResult()
        soup = self._load_soup(raw)
        if soup is None:
            return CollectionExtractionResult()
        records: list[CollectionRecord] = []
        methods: list[str] = []

        json_ld_records = self._extract_json_ld(
            soup,
            parent_url=parent_url,
            source_title=source_title,
        )
        if json_ld_records:
            records.extend(json_ld_records)
            methods.append("json_ld")

        table_records = self._extract_tables(
            soup,
            parent_url=parent_url,
            source_title=source_title,
        )
        if table_records:
            records.extend(table_records)
            methods.append("html_table")

        repeated_records = self._extract_repeated_blocks(
            soup,
            parent_url=parent_url,
            source_title=source_title,
            source_kind=source_kind,
        )
        if repeated_records:
            records.extend(repeated_records)
            methods.append("repeated_dom")

        assembled = self.assembler.assemble(
            records[: self.max_records * 3],
            parent_url=parent_url,
        )[: self.max_records]
        return CollectionExtractionResult(
            records=assembled,
            methods=tuple(methods),
        )

    def _load_soup(self, html: str):
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return None
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        for selector in "nav, header, footer, aside, form, script:not([type='application/ld+json']), style, noscript".split(", "):
            for node in soup.select(selector):
                node.decompose()
        return soup

    def _extract_json_ld(
        self,
        soup: Any,
        *,
        parent_url: str,
        source_title: str,
    ) -> list[CollectionRecord]:
        records: list[CollectionRecord] = []
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text() or ""
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            for node in self._json_nodes(payload):
                record_type = self._schema_record_type(node.get("@type"))
                if not record_type:
                    continue
                title = self._first_text(node, "headline", "name")
                authors = tuple(self._person_names(node.get("author") or node.get("creator")))
                date = self._first_text(node, "datePublished", "dateCreated", "dateModified")
                source = self._named_value(node.get("publisher") or node.get("isPartOf"))
                content_url = self._first_text(node, "url", "mainEntityOfPage", "sameAs")
                content = self._first_text(node, "abstract", "description", "articleBody")
                language = self._named_value(node.get("inLanguage"))
                country = self._country_value(
                    node.get("contentLocation")
                    or node.get("spatialCoverage")
                    or node.get("countryOfOrigin")
                )
                records.append(
                    CollectionRecord(
                        record_type=record_type,
                        title=title,
                        authors=authors,
                        date=date,
                        source=source or source_title,
                        content_url=urljoin(parent_url, content_url),
                        language=language,
                        country=country,
                        content=content,
                        parent_url=parent_url,
                        extraction_method="json_ld",
                    )
                )
        return records

    def _json_nodes(self, value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, list):
            for item in value:
                yield from self._json_nodes(item)
            return
        if not isinstance(value, dict):
            return
        if "@type" in value:
            yield value
        graph = value.get("@graph")
        if graph is not None:
            yield from self._json_nodes(graph)
        for key in ("itemListElement", "hasPart", "item"):
            nested = value.get(key)
            if nested is not None:
                yield from self._json_nodes(nested)

    def _schema_record_type(self, value: Any) -> str:
        values = value if isinstance(value, list) else [value]
        for item in values:
            key = normalize_text(str(item or "")).casefold()
            key = key.rsplit("/", 1)[-1]
            if key in self._SUPPORTED_TYPES:
                return self._SUPPORTED_TYPES[key]
        return ""

    def _extract_tables(
        self,
        soup: Any,
        *,
        parent_url: str,
        source_title: str,
    ) -> list[CollectionRecord]:
        records: list[CollectionRecord] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [self._cell_text(cell) for cell in rows[0].find_all(["th", "td"])]
            if not headers:
                continue
            mapped_headers = [self._field_name(header) for header in headers]
            for row in rows[1:]:
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue
                values = [self._cell_text(cell) for cell in cells]
                fields: dict[str, Any] = {}
                extras: list[tuple[str, str]] = []
                links: list[str] = []
                for index, value in enumerate(values):
                    header = headers[index] if index < len(headers) else f"Column {index + 1}"
                    field_name = mapped_headers[index] if index < len(mapped_headers) else ""
                    cell = cells[index]
                    link = cell.find("a", href=True)
                    if link is not None:
                        links.append(urljoin(parent_url, str(link.get("href") or "")))
                    if field_name:
                        if field_name == "authors":
                            fields[field_name] = tuple(self._split_authors(value))
                        elif field_name not in fields or not fields[field_name]:
                            fields[field_name] = value
                    elif value:
                        extras.append((header, value))
                title_index = next(
                    (index for index, name in enumerate(mapped_headers) if name == "title"),
                    -1,
                )
                if title_index >= 0 and title_index < len(cells):
                    title_link = cells[title_index].find("a", href=True)
                    if title_link is not None:
                        fields["content_url"] = urljoin(
                            parent_url,
                            str(title_link.get("href") or ""),
                        )
                fields.setdefault("content_url", links[0] if links else "")
                if not fields.get("title"):
                    fields["title"] = next((value for value in values if value), "")
                records.append(
                    CollectionRecord(
                        record_type=self._table_record_type(mapped_headers),
                        title=str(fields.get("title") or ""),
                        authors=tuple(fields.get("authors") or ()),
                        date=str(fields.get("date") or ""),
                        source=str(fields.get("source") or source_title),
                        content_url=str(fields.get("content_url") or ""),
                        language=str(fields.get("language") or ""),
                        country=str(fields.get("country") or ""),
                        content=str(fields.get("content") or ""),
                        parent_url=parent_url,
                        extra_fields=tuple(extras),
                        extraction_method="html_table",
                    )
                )
        return records

    def _extract_repeated_blocks(
        self,
        soup: Any,
        *,
        parent_url: str,
        source_title: str,
        source_kind: str,
    ) -> list[CollectionRecord]:
        blocks: list[Any] = list(soup.find_all("article"))
        seen_nodes = {id(block) for block in blocks}
        for parent in soup.find_all(["ul", "ol", "div", "section"]):
            children = [child for child in parent.find_all(recursive=False) if getattr(child, "name", None)]
            if len(children) < 2:
                continue
            signatures = [self._block_signature(child) for child in children]
            common = Counter(signatures).most_common(1)
            if not common or common[0][1] < 2:
                continue
            signature = common[0][0]
            if not signature:
                continue
            for child, child_signature in zip(children, signatures, strict=False):
                if child_signature != signature or id(child) in seen_nodes:
                    continue
                text = normalize_text(child.get_text(" ", strip=True))
                if not 20 <= len(text) <= 4000 or child.find("a", href=True) is None:
                    continue
                blocks.append(child)
                seen_nodes.add(id(child))

        records: list[CollectionRecord] = []
        for block in blocks:
            record = self._block_record(
                block,
                parent_url=parent_url,
                source_title=source_title,
                source_kind=source_kind,
            )
            if record is not None:
                records.append(record)
        return records

    def _block_record(
        self,
        block: Any,
        *,
        parent_url: str,
        source_title: str,
        source_kind: str,
    ) -> CollectionRecord | None:
        heading = block.find(["h1", "h2", "h3", "h4", "h5"])
        title_link = heading.find("a", href=True) if heading is not None else None
        if title_link is None:
            title_link = block.find("a", href=True)
        title = self._node_text(heading) or self._node_text(title_link)
        if not title:
            return None
        content_url = urljoin(parent_url, str(title_link.get("href") or "")) if title_link else ""
        authors = tuple(
            self._unique_texts(
                block.select(
                    "[rel~='author'], [itemprop='author'], .author, .authors, "
                    "[class*='author'], [class*='byline']"
                )
            )
        )
        date_node = block.select_one(
            "time, [itemprop='datePublished'], [class*='date'], [class*='year']"
        )
        date = ""
        if date_node is not None:
            date = normalize_text(str(date_node.get("datetime") or "")) or self._node_text(date_node)
        source_node = block.select_one(
            "[itemprop='publisher'], [class*='journal'], [class*='venue'], [class*='source']"
        )
        language_node = block.select_one(
            "[itemprop='inLanguage'], [class*='language'], [data-language]"
        )
        country_node = block.select_one(
            "[class*='country'], [data-country], [itemprop='addressCountry']"
        )
        language = self._attribute_or_text(language_node, "data-language")
        country = self._attribute_or_text(country_node, "data-country")
        if not language:
            language = normalize_text(str(block.get("lang") or ""))
        if not country:
            country = self._semantic_image_label(block, marker="country")
        if not language:
            language = self._semantic_image_label(block, marker="language")
        content_nodes = block.select(
            "[itemprop='description'], [class*='abstract'], [class*='summary'], "
            "[class*='description'], p"
        )
        content = " ".join(self._unique_texts(content_nodes)[:3])
        if not content:
            full_text = normalize_text(block.get_text(" ", strip=True))
            content = full_text.removeprefix(title).strip(" -:|")
        visual_labels = self._visual_labels(block)
        record_type = "publication" if authors else "article"
        if source_kind in {"database", "dataset"}:
            record_type = "database_row"
        return CollectionRecord(
            record_type=record_type,
            title=title,
            authors=authors,
            date=date,
            source=self._node_text(source_node) or source_title,
            content_url=content_url,
            language=language,
            country=country,
            content=content,
            parent_url=parent_url,
            extra_fields=(
                (("Visual Labels", "; ".join(visual_labels)),)
                if visual_labels
                else ()
            ),
            extraction_method="repeated_dom",
        )

    def _block_signature(self, node: Any) -> str:
        name = str(getattr(node, "name", "") or "")
        classes = sorted(str(item) for item in list(node.get("class") or []))[:3]
        has_heading = bool(node.find(["h1", "h2", "h3", "h4", "h5"]))
        has_time = bool(node.find("time"))
        has_link = bool(node.find("a", href=True))
        return "|".join([name, ".".join(classes), str(has_heading), str(has_time), str(has_link)])

    def _table_record_type(self, fields: list[str]) -> str:
        if "authors" in fields:
            return "publication"
        if "title" in fields and any(name in fields for name in ("date", "language", "country")):
            return "article"
        return "database_row"

    def _field_name(self, header: str) -> str:
        key = re.sub(r"[^a-z0-9]+", " ", normalize_text(header).casefold()).strip()
        for field_name, aliases in self._FIELD_ALIASES.items():
            if key in aliases:
                return field_name
        return ""

    def _cell_text(self, cell: Any) -> str:
        values = [normalize_text(cell.get_text(" ", strip=True))]
        values.extend(
            normalize_text(str(image.get("alt") or image.get("title") or image.get("aria-label") or ""))
            for image in cell.find_all("img")
        )
        return " ".join(dict.fromkeys(value for value in values if value))

    def _first_text(self, node: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = node.get(key)
            text = self._named_value(value)
            if text:
                return text
        return ""

    def _named_value(self, value: Any) -> str:
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, dict):
            for key in ("name", "headline", "@id", "url"):
                text = self._named_value(value.get(key))
                if text:
                    return text
        if isinstance(value, list):
            values = [self._named_value(item) for item in value]
            return "; ".join(item for item in values if item)
        return ""

    def _person_names(self, value: Any) -> list[str]:
        values = value if isinstance(value, list) else [value]
        result: list[str] = []
        for item in values:
            name = self._named_value(item)
            if name:
                result.append(name)
        return result

    def _country_value(self, value: Any) -> str:
        if isinstance(value, dict):
            address = value.get("address")
            if isinstance(address, dict):
                country = self._named_value(address.get("addressCountry"))
                if country:
                    return country
        return self._named_value(value)

    def _split_authors(self, value: str) -> list[str]:
        return [
            normalize_text(item)
            for item in re.split(r"\s*(?:;|\||\band\b|\s+&\s+)\s*", value, flags=re.I)
            if normalize_text(item)
        ]

    def _unique_texts(self, nodes: Iterable[Any]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            text = self._node_text(node)
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    def _node_text(self, node: Any) -> str:
        if node is None:
            return ""
        return normalize_text(node.get_text(" ", strip=True))

    def _attribute_or_text(self, node: Any, attribute: str) -> str:
        if node is None:
            return ""
        return normalize_text(str(node.get(attribute) or "")) or self._node_text(node)

    def _semantic_image_label(self, block: Any, *, marker: str) -> str:
        for image in block.find_all("img"):
            context = " ".join(
                [
                    str(image.get("class") or ""),
                    str(image.get("data-type") or ""),
                    str(image.get("aria-label") or ""),
                ]
            ).casefold()
            if marker not in context:
                continue
            label = normalize_text(
                str(image.get("alt") or image.get("title") or image.get("aria-label") or "")
            )
            if label:
                return label
        return ""

    def _visual_labels(self, block: Any) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for image in block.find_all("img"):
            label = normalize_text(
                str(image.get("alt") or image.get("title") or image.get("aria-label") or "")
            )
            key = label.casefold()
            if label and key not in seen:
                labels.append(label)
                seen.add(key)
        return labels


__all__ = ["CollectionRecordExtractor"]
