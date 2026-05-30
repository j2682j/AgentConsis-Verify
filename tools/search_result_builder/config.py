"""
定義搜索結果的配置類，包括搜索結果的格式、字段等信息
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionAnalysis:
    """
    Store lightweight question analysis for typed candidate generation.
    """
    answer_type: str
    target_terms: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    banned_answer_terms: list[str] = field(default_factory=list)
    requires_verification: bool = True
    needs_calculation: bool = False
    needs_multi_hop: bool = False

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
    把 TypedCandidateExtractor 回傳的候選答案正式化
    """
    answer: str
    answer_type: str
    support_count: int = 0
    verification_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    verified: bool = False
    probability_score: float = 0.0
    probability_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiedCandidate:
    """
    Store a candidate answer after rule-based support/refute verification.
    """
    candidate_id: str
    answer: str
    answer_type: str
    support_count: int = 0
    refute_count: int = 0
    neutral_count: int = 0
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    support_fact_ids: list[str] = field(default_factory=list)
    refute_fact_ids: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class FactCard:
    """
    Compact factual claim derived from an evidence chunk for SLM-friendly prompts.
    """
    fact_id: str
    claim: str
    relation: str
    candidate_id: str = ""
    source_id: str = ""
    evidence_id: str = ""
    confidence: float = 0.0
    constraint_matches: list[str] = field(default_factory=list)


@dataclass
class AgentEvidencePacket:
    """
    Compact evidence packet rendered for Stage1 agents and Stage2 judges.
    """
    question: str
    answer_type: str
    candidates: list[VerifiedCandidate] = field(default_factory=list)
    facts: list[FactCard] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

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
    verified_candidates: list[VerifiedCandidate] = field(default_factory=list)
    fact_cards: list[FactCard] = field(default_factory=list)
    agent_packet: AgentEvidencePacket | None = None
    question_analysis: QuestionAnalysis | None = None
    candidate_diagnostics: dict[str, Any] = field(default_factory=dict)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    blocked_sources: list[SearchSourceCandidate] = field(default_factory=list)
