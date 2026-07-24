from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceTier(str, Enum):
    """How much answer-support authority a passage carries.

    Only ANSWER_GROUNDED and BRIDGE_GROUNDED passages may create evidence
    support. RELAXED_CONTEXT passages are reading material for Stage1 agents
    and must never reach the support checker, the fact store, or contradiction
    checks — otherwise a passage that merely shares vocabulary with the
    question manufactures support for whatever answer an agent guessed.
    """

    ANSWER_GROUNDED = "answer_grounded"
    BRIDGE_GROUNDED = "bridge_grounded"
    RELAXED_CONTEXT = "relaxed_context"


SUPPORT_ELIGIBLE_TIERS = frozenset(
    {EvidenceTier.ANSWER_GROUNDED.value, EvidenceTier.BRIDGE_GROUNDED.value}
)


def is_support_eligible_payload(payload: Any) -> bool:
    """Return True only for evidence payloads allowed to create support.

    Accepts the dict form used across the search pipeline. Payloads without an
    explicit tier default to support-eligible so that strict evidence produced
    before this contract existed keeps working unchanged.
    """

    if not isinstance(payload, dict):
        return False
    if "support_eligible" in payload:
        return bool(payload.get("support_eligible"))
    tier = str(payload.get("evidence_tier") or "").strip()
    if not tier:
        return not bool(payload.get("relaxed"))
    return tier in SUPPORT_ELIGIBLE_TIERS


@dataclass
class SearchSignals:
    """
    描述搜尋前處理得到的任務訊號。

    Args:
     - answer_type: 預期答案型態，未知時使用 unknown。
     - target_terms: semantic impact / query generator 找到的目標片段。
     - constraints: 題目中必須保留的限制條件。
     - source_hints: 題目指定或暗示的資料來源。
     - needs_multi_hop: 任務是否可能需要多跳搜尋。

    Returns:
     - SearchSignals: 可傳遞給搜尋規劃與檢索控制的訊號。

    """

    answer_type: str = "unknown"
    target_terms: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    needs_multi_hop: bool = False


@dataclass
class SearchSourceCandidate:
    """
    表示 search backend 回傳後、等待過濾與全文抓取的來源候選。

    Args:
     - source_id: source id。
     - query_id: 來源 query id。
     - title: 搜尋結果標題。
     - url: 搜尋結果 URL。
     - domain: URL domain。
     - snippet: 搜尋結果摘要。
     - raw_content: full-page fetch 後取得的正文。
     - rank: 搜尋結果排名。
     - fetched: 是否已經抓取全文。
     - blocked: 是否被 source filter 擋下。
     - block_reason: 擋下原因。
     - should_fetch_full_page: 是否需要進一步抓取完整網頁。
     - filter_reasons: filter / fetch 過程記錄。

    Returns:
     - SearchSourceCandidate: 可進入 source analysis 的候選來源。

    """

    source_id: str
    query_id: str
    title: str
    url: str
    domain: str = ""
    snippet: str = ""
    # Fetch-selection signals. A source only earns a full-page fetch slot on
    # evidence — being named by the question, being found by more than one
    # query, or matching the question's constraints — never on search-engine
    # rank alone, which SEO pages built from the question text tend to win.
    canonical_url: str = ""
    matched_query_ids: list[str] = field(default_factory=list)
    query_hit_count: int = 0
    named_source_match: bool = False
    named_source_terms: list[str] = field(default_factory=list)
    url_echo: bool = False
    constraint_match_level: str = "no_match"
    fetch_priority_tier: int = 9
    fetch_priority_reasons: list[str] = field(default_factory=list)
    legacy_fetch_position: int = -1
    fetch_batch: int = 0
    raw_content: str = ""
    raw_html: str = ""
    content_complete: bool = False
    content_truncated: bool = False
    original_content_chars: int = 0
    final_url: str = ""
    rank: int = 0
    fetched: bool = False
    blocked: bool = False
    block_reason: str = ""
    should_fetch_full_page: bool = False
    filter_reasons: list[str] = field(default_factory=list)
    source_kind: str = "web"
    access_mode: str = "search"
    source_hint: str = ""
    required_content: str = "html_text"
    transport_ok: bool = False
    content_extracted: bool = False
    requirement_met: bool = False
    acquisition_state: str = "pending"
    missing_content: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """
    Source analysis 轉給 Agent 使用的 evidence chunk。

    Args:
     - evidence_id: evidence id。
     - source_id: 來源文件 id。
     - query_id: 來源 query id。
     - text: evidence 文字內容。
     - title: source title。
     - url: source URL。
     - matched_terms: labeler / span recovery 找到的 useful terms。
     - matched_spans: useful terms 還原後的 span metadata。
     - retrieval_score: Retriever / FAISS 回傳的相似度。
     - sequence_tag: EfficientRAG labeler 的 CONTINUE / FINISH / TERMINATE 標籤。
     - selection_reason: evidence bucket selection 選中此 chunk 的原因。
     - evidence_bucket: evidence contract bucket。
     - compatible_spans: 符合 answer role 的 useful spans。
     - helpfulness_score: optional helpfulness score。
     - evidence_quality: optional evidence quality metadata。
     - cleaning_reasons: cleaning / labeler / dedup 的記錄。

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
    evidence_bucket: str = ""
    compatible_spans: list[str] = field(default_factory=list)
    helpfulness_score: float = 0.0
    evidence_quality: float = 0.0
    cleaning_reasons: list[str] = field(default_factory=list)
    # Evidence trust contract. Defaults are deliberately the least-privileged
    # values so any passage that does not explicitly earn grounding stays
    # read-only context.
    evidence_tier: str = EvidenceTier.RELAXED_CONTEXT.value
    support_eligible: bool = False
    verification_ready: bool = False
    relation_goal_ids: list[str] = field(default_factory=list)
    grounding_status: str = ""
    promotion_reason: str = ""


@dataclass(frozen=True)
class UnverifiedReference:
    """A retrieved passage exposed to Stage1 without verification authority."""

    reference_id: str
    source_id: str
    title: str
    text: str
    url: str = ""
    retrieval_score: float = 0.0
    retrieval_round: int = 0
    source_type: str = "passage"
    fallback_reason: str = "strict_evidence_empty"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "retrieval_score": self.retrieval_score,
            "retrieval_round": self.retrieval_round,
            "source_type": self.source_type,
            "fallback_reason": self.fallback_reason,
            "verified": False,
        }


__all__ = [
    "EvidenceItem",
    "SearchSignals",
    "SearchSourceCandidate",
    "UnverifiedReference",
]
