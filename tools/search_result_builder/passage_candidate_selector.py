from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class PassageSelection:
    """
    保存送入 Labeler 前的 passage 選擇結果。

    Args:
     - document: 原始 corpus passage。
     - retrieval_score: Passage 在 dense retrieval 中的最高相似度。
     - fusion_score: 多路排名經 RRF 合併後的分數。
     - selection_sources: Passage 被選入候選池的來源。
     - expanded_from: 相鄰擴展的來源 passage ID。

    Returns:
     - PassageSelection: 可轉回 retrieval control 輸入的 passage。
    """

    document: dict[str, Any]
    retrieval_score: float
    fusion_score: float
    selection_sources: tuple[str, ...]
    expanded_from: str = ""


class PassageCandidateSelector:
    """
    使用 dense、BM25 與相鄰 passage 擴展建立高召回候選池。

    Args:
     - rrf_constant: Reciprocal Rank Fusion 的排名平滑常數。
     - lexical_top_k: BM25 最多提供的 passage 數量。
     - max_per_domain: 最終候選池每個 domain 的 passage 上限。
     - max_neighbor_items: 最多加入的相鄰 passages 數量。

    Returns:
     - PassageCandidateSelector: 不需額外生成模型的 passage selector。
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_.-]*")
    _PASSAGE_ID_RE = re.compile(r"^(.*-)(\d+)$")

    def __init__(
        self,
        *,
        rrf_constant: int = 60,
        lexical_top_k: int = 30,
        max_per_domain: int = 3,
        max_neighbor_items: int = 4,
    ) -> None:
        self.rrf_constant = max(1, rrf_constant)
        self.lexical_top_k = max(1, lexical_top_k)
        self.max_per_domain = max(1, max_per_domain)
        self.max_neighbor_items = max(0, max_neighbor_items)

    def select(
        self,
        *,
        passage_map: dict[str, dict[str, Any]],
        ranked_dense_lists: dict[str, list[tuple[str, float]]],
        lexical_query: str,
        max_items: int,
    ) -> list[PassageSelection]:
        """
        合併多路 passage 排名並加入少量相鄰內容。

        Args:
         - passage_map: Corpus passage ID 到內容的映射。
         - ranked_dense_lists: 各 dense query view 的排序結果。
         - lexical_query: BM25 使用的查詢文字。
         - max_items: 最終送入 Labeler 的 passage 上限。

        Returns:
         - list[PassageSelection]: 依融合排名排列的 passage candidates。
        """
        limit = max(1, max_items)
        rankings: dict[str, list[str]] = {
            name: [document_id for document_id, _ in values]
            for name, values in ranked_dense_lists.items()
        }
        lexical_rank = self._bm25_rank(
            query=lexical_query,
            passage_map=passage_map,
            top_k=self.lexical_top_k,
        )
        if lexical_rank:
            rankings["bm25"] = lexical_rank

        dense_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        fusion_scores: Counter[str] = Counter()
        for source, values in ranked_dense_lists.items():
            for rank, (document_id, score) in enumerate(values, start=1):
                dense_scores[document_id] = max(
                    dense_scores.get(document_id, float("-inf")),
                    float(score),
                )
                sources.setdefault(document_id, set()).add(source)
                fusion_scores[document_id] += self._rrf(rank)
        for source, document_ids in rankings.items():
            if source in ranked_dense_lists:
                continue
            for rank, document_id in enumerate(document_ids, start=1):
                sources.setdefault(document_id, set()).add(source)
                fusion_scores[document_id] += self._rrf(rank)

        ordered_ids = sorted(
            fusion_scores,
            key=lambda document_id: (
                -fusion_scores[document_id],
                document_id,
            ),
        )
        selected: list[PassageSelection] = []
        selected_ids: set[str] = set()
        domain_counts: Counter[str] = Counter()
        neighbor_count = 0

        for document_id in ordered_ids:
            if len(selected) >= limit:
                break
            primary = self._selection(
                document_id=document_id,
                passage_map=passage_map,
                dense_scores=dense_scores,
                fusion_scores=fusion_scores,
                sources=sources,
            )
            if primary is None or not self._accept_domain(primary, domain_counts):
                continue
            selected.append(primary)
            selected_ids.add(document_id)

            if neighbor_count >= self.max_neighbor_items:
                continue
            if self._is_structured_row(primary.document):
                continue
            for neighbor_id in self._neighbor_ids(document_id):
                if len(selected) >= limit or neighbor_count >= self.max_neighbor_items:
                    break
                if neighbor_id in selected_ids or neighbor_id not in passage_map:
                    continue
                neighbor = PassageSelection(
                    document=dict(passage_map[neighbor_id]),
                    retrieval_score=max(0.0, dense_scores.get(neighbor_id, 0.0)),
                    fusion_score=0.0,
                    selection_sources=("adjacent",),
                    expanded_from=document_id,
                )
                if not self._accept_domain(neighbor, domain_counts):
                    continue
                selected.append(neighbor)
                selected_ids.add(neighbor_id)
                neighbor_count += 1
        return selected

    def _selection(
        self,
        *,
        document_id: str,
        passage_map: dict[str, dict[str, Any]],
        dense_scores: dict[str, float],
        fusion_scores: Counter[str],
        sources: dict[str, set[str]],
    ) -> PassageSelection | None:
        document = passage_map.get(document_id)
        if document is None:
            return None
        return PassageSelection(
            document=dict(document),
            retrieval_score=max(0.0, dense_scores.get(document_id, 0.0)),
            fusion_score=float(fusion_scores.get(document_id, 0.0)),
            selection_sources=tuple(sorted(sources.get(document_id, set()))),
        )

    def _bm25_rank(
        self,
        *,
        query: str,
        passage_map: dict[str, dict[str, Any]],
        top_k: int,
    ) -> list[str]:
        query_terms = list(dict.fromkeys(self._tokens(query)))
        if not query_terms or not passage_map:
            return []
        document_ids = list(passage_map)
        documents = [
            self._tokens(
                " ".join(
                    [
                        str(passage_map[document_id].get("title", "") or ""),
                        str(passage_map[document_id].get("text", "") or ""),
                    ]
                )
            )
            for document_id in document_ids
        ]
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))
        document_count = len(documents)
        average_length = sum(len(tokens) for tokens in documents) / max(1, document_count)
        scores: list[tuple[str, float]] = []
        for document_id, tokens in zip(document_ids, documents, strict=False):
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency <= 0:
                    continue
                frequency_in_documents = document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * length / max(1.0, average_length)
                )
                score += inverse_document_frequency * frequency * 2.5 / denominator
            if score > 0:
                scores.append((document_id, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [document_id for document_id, _ in scores[:top_k]]

    def _tokens(self, text: str) -> list[str]:
        return [
            token.casefold()
            for token in self._TOKEN_RE.findall(normalize_text(text))
            if len(token) > 1
        ]

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_constant + max(1, rank))

    def _neighbor_ids(self, document_id: str) -> Iterable[str]:
        match = self._PASSAGE_ID_RE.match(document_id)
        if match is None:
            return []
        prefix, suffix = match.groups()
        width = len(suffix)
        index = int(suffix)
        result = []
        if index > 0:
            result.append(f"{prefix}{index - 1:0{width}d}")
        result.append(f"{prefix}{index + 1:0{width}d}")
        return result

    def _is_structured_row(self, document: dict[str, Any]) -> bool:
        record_type = normalize_text(document.get("record_type", "")).casefold()
        if record_type and record_type != "passage":
            return True
        text = normalize_text(document.get("text", "")).casefold()
        return "columns:" in text and "row:" in text

    def _accept_domain(
        self,
        selection: PassageSelection,
        counts: Counter[str],
    ) -> bool:
        record_type = normalize_text(selection.document.get("record_type", "")).casefold()
        if record_type and record_type != "passage":
            return True
        domain = self._domain(selection.document.get("url", ""))
        if not domain:
            return True
        if counts[domain] >= self.max_per_domain:
            return False
        counts[domain] += 1
        return True

    def _domain(self, value: Any) -> str:
        try:
            return urlparse(str(value or "")).netloc.casefold()
        except Exception:
            return ""


__all__ = ["PassageCandidateSelector", "PassageSelection"]
