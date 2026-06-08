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
class SearchQueryPlan:
    """
    描述一次要執行的搜尋 query。

    Args:
        - query_id: query id。
        - query: 實際送進搜尋工具的字串。
        - purpose: query 的用途描述。
        - priority: query 優先順序。
        - source_hints: 偏好的來源提示，目前通常為空。
        - expected_answer_type: 保留欄位，預設 unknown。
        - requires_full_page: 是否建議後續 source analysis 抓完整頁面。

    Returns:
        - SearchQueryPlan: 搜尋 query 計畫。
    """

    query_id: str
    query: str
    purpose: str
    priority: int = 0
    source_hints: list[str] = field(default_factory=list)
    expected_answer_type: str = "unknown"
    requires_full_page: bool = False


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
        - helpfulness_score: Helpfulness Expert 分數。
        - evidence_quality: next-hop controller 使用的 evidence 品質分數。
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
    helpfulness_score: float = 0.0
    evidence_quality: float = 0.0
    cleaning_reasons: list[str] = field(default_factory=list)


@dataclass
class CandidateAnswer:
    """
    保存 search flow 可能抽出的候選答案。

    目前 Agent prompt 不直接使用 candidate answer；此結構主要保留給
    next-hop query 或後續實驗使用。

    Args:
        - answer: 候選答案文字。
        - answer_type: 候選答案型別。
        - support_count: 支撐 evidence 數量。
        - confidence: 候選答案信心分數。
        - evidence_ids: 支撐該候選的 evidence ids。
        - source_ids: 支撐該候選的 source ids。

    Returns:
        - CandidateAnswer: 候選答案資料。
    """

    answer: str
    answer_type: str = "entity"
    support_count: int = 0
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceOutput:
    """
    保存 EvidenceSearcher 的完整輸出。

    Args:
        - question: 原始問題。
        - queries: 已執行的 query plans。
        - sources: source analysis 後保留的 sources。
        - evidence_items: useful evidence chunks。
        - summary: 給 Agent 使用的 prompt-ready evidence context。
        - candidates: 候選答案，目前通常不輸出到 Agent prompt。
        - search_signals: 搜尋控制訊號，目前來自 embedding salient spans。
        - candidate_diagnostics: source analysis 診斷資訊。
        - tool_usage: search tool 使用紀錄。
        - blocked_sources: 被 hard filter 擋下的 sources。

    Returns:
        - EvidenceOutput: search_result_builder 的結構化輸出。
    """

    question: str
    queries: list[SearchQueryPlan]
    sources: list[SearchSourceCandidate]
    evidence_items: list[EvidenceItem]
    summary: str
    candidates: list[CandidateAnswer] = field(default_factory=list)
    search_signals: SearchSignals | None = None
    candidate_diagnostics: dict[str, Any] = field(default_factory=dict)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    blocked_sources: list[SearchSourceCandidate] = field(default_factory=list)


__all__ = [
    "CandidateAnswer",
    "EvidenceItem",
    "EvidenceOutput",
    "SearchQueryPlan",
    "SearchSignals",
    "SearchSourceCandidate",
]
