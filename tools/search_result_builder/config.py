from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionAnalysis:
    """
    儲存搜尋流程需要的輕量問題分析。

    Args:
        - answer_type: 預估答案型別。
        - target_terms: 問題中的重要實體、名詞或關鍵詞。
        - constraints: 日期、年份、數字等限制條件。
        - source_hints: 可選的來源提示。
        - needs_multi_hop: 是否可能需要多跳搜尋。

    Returns:
        - QuestionAnalysis: 問題分析資料。
    """

    answer_type: str = "entity"
    target_terms: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    needs_multi_hop: bool = False


@dataclass
class SearchQueryPlan:
    """
    儲存一次要送進搜尋引擎的 query plan。

    Args:
        - query_id: query id。
        - query: 搜尋字串。
        - purpose: query 用途。
        - priority: query 優先序。
        - source_hints: 可選的來源提示。
        - expected_answer_type: 預期答案型別。
        - requires_full_page: 是否建議 source analysis 抓全文。

    Returns:
        - SearchQueryPlan: 單一搜尋 query 計畫。
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
    儲存 search tool 回傳的一個 source。

    Args:
        - source_id: source id。
        - query_id: 來源 query id。
        - title: 搜尋結果標題。
        - url: 搜尋結果 URL。
        - domain: URL domain。
        - snippet: 搜尋結果摘要或短內容。
        - raw_content: full-page fetch 後的全文內容。
        - rank: 原始搜尋結果排序。
        - fetched: 是否已抓全文。
        - blocked: 是否被 hard filter 擋掉。
        - block_reason: 被擋原因。
        - leak_score: leak hard-filter 診斷值。
        - duplicate_score: duplicate hard-filter 診斷值。
        - question_echo_score: question echo hard-filter 診斷值。
        - should_fetch_full_page: 是否要抓全文。
        - filter_reasons: filter / fetch 診斷資訊。

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
    leak_score: float = 0.0
    duplicate_score: float = 0.0
    question_echo_score: float = 0.0
    should_fetch_full_page: bool = False
    filter_reasons: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """
    儲存已通過 source analysis 的 useful evidence chunk。

    Args:
        - evidence_id: evidence id。
        - source_id: 來源 source id。
        - query_id: 來源 query id。
        - text: evidence 文字。
        - title: source title。
        - url: source URL。
        - matched_terms: EfficientRAG labeler 保留的 useful tokens。
        - helpfulness_score: Helpfulness Expert 分數。
        - evidence_quality: 給 next-hop controller 使用的 evidence 品質分數，目前等同 helpfulness_score。
        - cleaning_reasons: helpfulness / labeler / dedup 診斷資訊。

    Returns:
        - EvidenceItem: 可放入 Agent prompt 的 evidence。
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
    儲存可選的候選答案。

    目前新 search flow 不主動產生 candidate，但保留此結構給 next-hop query
    此結構用來記錄搜尋流程抽出的候選答案。

    Args:
        - answer: 候選答案文字。
        - answer_type: 候選答案型別。
        - support_count: 支撐次數。
        - confidence: 候選答案信心。
        - evidence_ids: 支撐 evidence ids。
        - source_ids: 支撐 source ids。

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
    儲存 EvidenceSearcher 的完整輸出。

    Args:
        - question: 原始問題。
        - queries: 執行過的 query plans。
        - sources: 通過 hard filter 的 sources。
        - evidence_items: useful evidence chunks。
        - candidates: 可選候選答案，目前通常為空。
        - summary: 給 Agent 的 prompt-ready evidence context。
        - question_analysis: 問題分析結果。
        - candidate_diagnostics: 搜尋與 source analysis 診斷資訊。
        - tool_usage: search tool 使用紀錄。
        - blocked_sources: 被 hard filter 擋掉的 sources。

    Returns:
        - EvidenceOutput: search_result_builder 主輸出。
    """

    question: str
    queries: list[SearchQueryPlan]
    sources: list[SearchSourceCandidate]
    evidence_items: list[EvidenceItem]
    summary: str
    candidates: list[CandidateAnswer] = field(default_factory=list)
    question_analysis: QuestionAnalysis | None = None
    candidate_diagnostics: dict[str, Any] = field(default_factory=dict)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    blocked_sources: list[SearchSourceCandidate] = field(default_factory=list)


__all__ = [
    "CandidateAnswer",
    "EvidenceItem",
    "EvidenceOutput",
    "QuestionAnalysis",
    "SearchQueryPlan",
    "SearchSourceCandidate",
]
