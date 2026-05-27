"""
定義搜索結果的配置類，包括搜索結果的格式、字段等信息
"""

from dataclasses import dataclass, field
from typing import Any



@dataclass
class SearchQueryPlan:
    """
    讓每個 query 有目的和優先級，讓 agent 可以根據目的和優先級來決定要先執行哪個 query
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
    每個 search result 都變成可追蹤 source。之後可以知道 evidence 從哪個 URL 來
    """
    source_id: str
    query_id: str
    title: str
    url: str
    domain: str = ""
    snippet: str = ""
    raw_content: str = ""
    rank: int = 0
    rerank_score: float = 0.0
    fetched: bool = False
    blocked: bool = False
    block_reason: str = ""

@dataclass
class EvidenceItem:
    """
    不要把整頁或 snippet 丟給 Agent，而是抽成 [E1], [E2]
    """
    evidence_id: str
    source_id: str
    query_id: str
    text: str
    title: str = ""
    url: str = ""
    relevance_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    extracted_answer: str = ""
    
@dataclass
class CandidateAnswer:
    """
    把 CandidateExtractor 回傳的候選答案正式化
    """
    answer: str
    answer_type: str
    support_count: int = 0
    verification_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    verified: bool = False

@dataclass
class EvidenceOutput:
    """
    整個 search_result_builder 的正式輸出，Stage1、Stage2都會接收
    """
    question: str
    queries: list[SearchQueryPlan]
    sources: list[SearchSourceCandidate]
    evidence_items: list[EvidenceItem]
    candidates: list[CandidateAnswer]
    summary: str
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    blocked_sources: list[SearchSourceCandidate] = field(default_factory=list)
