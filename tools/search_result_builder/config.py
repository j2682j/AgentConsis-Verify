from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchSignals:
    """
    保存搜尋流程需要的輕量問題訊號。

    目前這個資料由 embedding salience query generator 從原始問題抽出的
    重要 span 建立，不再經過額外的問題分析階段。

    Args:
        - answer_type: 保留欄位，預設 unknown，不在 search 主流程中推斷答案型別。
        - target_terms: embedding salience 選出的重要文字 span。
        - constraints: 保留欄位，目前通常為空。
        - source_hints: 保留欄位，目前通常為空。
        - needs_multi_hop: 保留欄位，目前由 retrieval controller 判斷是否需要下一跳。

    Returns:
        - SearchSignals: 搜尋控制用的輕量問題訊號。
    """

    answer_type: str = "unknown"
    target_terms: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    needs_multi_hop: bool = False


@dataclass
class SearchSourceCandidate:
    """
    保存 search tool 回傳的一筆 source。

    Args:
        - source_id: source id。
        - query_id: 對應的 query id。
        - title: 搜尋結果標題。
        - url: 搜尋結果 URL。
        - domain: URL domain。
        - snippet: 搜尋結果摘要。
        - raw_content: full-page fetch 後取得的網頁內容。
        - rank: 搜尋結果排名。
        - fetched: 是否已抓取完整頁面。
        - blocked: 是否被 source filter 擋下。
        - block_reason: 被擋下的原因。
        - should_fetch_full_page: 是否建議抓取完整頁面。
        - filter_reasons: filter / fetch 的判斷紀錄。

    Returns:
        - SearchSourceCandidate: 搜尋來源資料。
    """

    source_id: str
    query_id: str
    title: str
    url: str
    domain: str = ""
    snippet: str = ""
    raw_content: str = ""
    rank: int = 0
    fetched: bool = False
    blocked: bool = False
    block_reason: str = ""
    should_fetch_full_page: bool = False
    filter_reasons: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """
    保存 source analysis 後可交給 Agent 的 evidence chunk。

    Args:
        - evidence_id: evidence id。
        - source_id: 對應的 source id。
        - query_id: 對應的 query id。
        - text: evidence 文字。
        - title: source title。
        - url: source URL。
        - matched_terms: EfficientRAG labeler 保留的 useful tokens。
        - matched_spans: useful tokens 對齊回原文後的 spans/context metadata。
        - retrieval_score: Retriever / FAISS 的相似度分數。
        - sequence_tag: EfficientRAG labeler 的 CONTINUE / TERMINATE 類標籤。
        - selection_reason: evidence conversion 選中此 chunk 的原因。
        - helpfulness_score: Helpfulness Expert 分數。
        - evidence_quality: next-hop controller 使用的 evidence 品質分數。
        - conversion_score: evidence conversion 排序用分數。
        - cleaning_reasons: helpfulness / labeler / dedup 的處理紀錄。

    Returns:
        - EvidenceItem: prompt-ready evidence。
    """

    evidence_id: str
    source_id: str
    query_id: str
    text: str
    title: str = ""
    url: str = ""
    matched_terms: list[str] = field(default_factory=list)
    matched_spans: list[dict[str, Any]] = field(default_factory=list)
    retrieval_score: float = 0.0
    sequence_tag: str = ""
    selection_reason: str = ""
    helpfulness_score: float = 0.0
    evidence_quality: float = 0.0
    conversion_score: float = 0.0
    cleaning_reasons: list[str] = field(default_factory=list)


__all__ = [
    "EvidenceItem",
    "SearchSignals",
    "SearchSourceCandidate",
]
