from __future__ import annotations

import hashlib
import os
import pickle
import re
import shutil
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from utils.network_utils import normalize_text
from tools.evidence.fact_extraction import (
    AbsenceCheck,
    AggregationDerivation,
    CompletenessContract,
    CrossContextAssembler,
    CrossContextFactExtractor,
    DirectEvidencePromoter,
    EvidenceFact,
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactStore,
)

from .config import EvidenceItem, SearchSourceCandidate
from .corpus import (
    DocumentChunker,
    TaskCorpusSession,
    WebCorpusBuilder,
    build_page_id,
    canonicalize_page_url,
)
from .embeddings import Embedder
from .evidence import (
    ANSWER_SUPPORT,
    BRIDGE,
    CandidateSpan,
    CandidateSpanGrounder,
    CandidateSpanQualityGate,
    EvidenceUtilityGate,
    EvidenceRoleContractBuilder,
    PassageEvidenceUnitBuilder,
    RoleAwareSpanFinalizer,
    SpanRoleClassifier,
)
from .next_hop_query.coverage_assessor import CoverageAssessor
from .next_hop_query.evidence_sufficiency_gate import EvidenceSufficiencyGate
from .next_hop_query.goal_completion import GoalCompletionEvaluator
from .next_hop_query.intent_state_tracker import SearchIntentStateTracker
from .next_hop_query.next_hop_evidence_selector import NextHopEvidenceSelector
from .next_hop_query.next_hop_query_composer import NextHopQueryComposer
from .next_hop_query.next_hop_result import NextHopQueryResult
from .next_hop_query.relation_evidence_binder import RelationEvidenceBinder
from .next_hop_query.relation_goal_resolver import RelationGoalResolver
from .next_hop_query.retrieval_recovery_policy import RetrievalRecoveryPolicy
from .next_hop_query.query_guard import NextHopQueryGuard
from .passage_retriever import Retriever
from .passage_candidate_selector import PassageCandidateSelector
from .query import QueryGenerator, SearchIntentPlan, SearchQueryRequest
from .source_acquisition import SourceAcquisitionRouter, SourceAcquisitionTrace
from .source_analyze.rag_labeler import (
    CONTINUE_TAG,
    EfficientRAGLabelerAdapter,
    FINISH_TAG,
    RAGLabelResult,
)
from .source_analyze.label_contract import LabelContractValidator
from .source_analyze.labeler_input_builder import LabelerInputBuilder
from .source_analyze.seer import PageContentFetcher, SourceFilter


@dataclass
class RetrievedDocumentTrace:
    """
    保存單一檢索文件在一輪迭代中的檢索與標註結果。

    Args:
        - document_id: Corpus passage ID。
        - title: 文件標題。
        - text: 文件內容。
        - url: 文件來源 URL。
        - retrieval_score: FAISS inner-product 相似度分數。
        - label: Labeler 輸出的 useful 或 useless 標籤。
        - sequence_tag: EfficientRAG 的 CONTINUE 或 TERMINATE 標籤。
        - useful_tokens: Labeler 從文件抽出的 useful tokens。
        - continue_probability: CONTINUE 類別機率。
        - terminate_probability: TERMINATE 類別機率。
        - duplicate: 是否與先前輪次的 URL 或 chunk 重複。
        - duplicate_reason: 文件被判定重複的原因。

    Returns:
        - RetrievedDocumentTrace: 可序列化的單一文件執行紀錄。
    """

    document_id: str
    title: str
    text: str
    url: str
    retrieval_score: float
    selection_score: float = 0.0
    selection_sources: list[str] = field(default_factory=list)
    expanded_from: str = ""
    record_type: str = "passage"
    record_id: str = ""
    record_fields: dict[str, Any] = field(default_factory=dict)
    content_scope: str = "passage"
    content_complete: bool = False
    content_truncated: bool = False
    original_content_chars: int = 0
    required_content: str = "html_text"
    acquisition_state: str = "pending"
    page_id: str = ""
    source_url: str = ""
    canonical_url: str = ""
    section_title: str = ""
    section_index: int = 0
    passage_index: int = 0
    content_type: str = "text"
    table_id: str = ""
    table_headers: list[str] = field(default_factory=list)
    row_index: int = -1
    label: str = ""
    sequence_tag: str = ""
    useful_tokens: list[str] = field(default_factory=list)
    useful_spans: list[str] = field(default_factory=list)
    raw_labeler_spans: list[str] = field(default_factory=list)
    grounded_labeler_spans: list[str] = field(default_factory=list)
    classified_spans: list[str] = field(default_factory=list)
    answer_support_spans: list[str] = field(default_factory=list)
    bridge_spans: list[str] = field(default_factory=list)
    span_roles: list[dict[str, Any]] = field(default_factory=list)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    support_level: str = ""
    valid_for_next_hop: bool = False
    valid_for_evidence: bool = False
    direct_contracts: list[dict[str, Any]] = field(default_factory=list)
    bridge_contracts: list[dict[str, Any]] = field(default_factory=list)
    rejected_contracts: list[dict[str, Any]] = field(default_factory=list)
    continue_probability: float = 0.0
    terminate_probability: float = 0.0
    label_status: str = ""
    invalid_reasons: list[str] = field(default_factory=list)
    labeler_diagnostics: dict[str, object] = field(default_factory=dict)
    duplicate: bool = False
    duplicate_reason: str = ""


@dataclass
class PageRetrievalTrace:
    """
    記錄單一 retrieval round 內每個 Page 的處理結果。

    Args:
     - page_id: 穩定的 Page ID。
     - url: Page canonical URL。
     - title: Page 標題。
     - status: direct_found、bridge_found 或 no_usable_contract。
     - selected_passage_ids: 本輪實際處理的 passage IDs。

    Returns:
     - PageRetrievalTrace: 可直接匯出到任務 JSON 的 Page trace。
    """

    page_id: str
    url: str = ""
    title: str = ""
    status: str = "no_usable_contract"
    selection_scope: str = "global"
    selection_reasons: list[str] = field(default_factory=list)
    selected_passage_ids: list[str] = field(default_factory=list)
    direct_document_ids: list[str] = field(default_factory=list)
    bridge_document_ids: list[str] = field(default_factory=list)
    direct_fact_ids: list[str] = field(default_factory=list)
    bridge_goal_ids: list[str] = field(default_factory=list)


@dataclass
class NextHopDecisionTrace:
    """
    記錄本輪是否允許跨 Page 搜尋及其 grounded bridge。

    Args:
     - required: 是否執行下一跳搜尋。
     - decision_reason: 允許或拒絕下一跳的原因。
     - generated_query: 通過 Query Guard 的下一跳 query。
     - bridge_document_ids: 支撐下一跳的 passage IDs。

    Returns:
     - NextHopDecisionTrace: Next-hop 控制決策紀錄。
    """

    required: bool = False
    decision_reason: str = ""
    unresolved_goal: str = ""
    bridge_spans: list[str] = field(default_factory=list)
    bridge_document_ids: list[str] = field(default_factory=list)
    generated_query: str = ""
    query_guard_result: str = ""
    target_page_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalRoundTrace:
    """
    保存一次 Retriever、Labeler 與 Filter 迭代的完整狀態。

    Args:
        - round_index: 從 1 開始的迭代輪數。
        - query: 本輪送入 Retriever 的 query。
        - documents: 本輪檢索文件、分數與標籤。
        - useful_tokens: 本輪所有 CONTINUE 文件抽出的去重 tokens。
        - next_query: Filter 產生的下一輪 query。
        - filter_metadata: Filter 模型執行資訊。
        - stop_reason: 若本輪停止，記錄停止原因。

    Returns:
        - RetrievalRoundTrace: 單輪 retrieval trace。
    """

    round_index: int
    query: str
    branch_queries: list[str] = field(default_factory=list)
    documents: list[RetrievedDocumentTrace] = field(default_factory=list)
    pages: list[PageRetrievalTrace] = field(default_factory=list)
    next_hop_decision: NextHopDecisionTrace = field(
        default_factory=NextHopDecisionTrace
    )
    cross_context_facts: list[dict[str, Any]] = field(default_factory=list)
    useful_tokens: list[str] = field(default_factory=list)
    next_query: str = ""
    next_queries: list[str] = field(default_factory=list)
    coverage: dict[str, object] = field(default_factory=dict)
    filter_metadata: dict[str, object] = field(default_factory=dict)
    stop_reason: str = ""


@dataclass
class IterativeRetrievalResult:
    """
    保存多輪 EfficientRAG retrieval control 的最終結果。

    Args:
        - initial_query: 第一輪使用的原始 query。
        - final_query: 流程停止時最後使用或產生的 query。
        - rounds: 各輪 query、文件、標籤與分數。
        - stop_reason: 整體流程停止原因。
        - searched_queries: 實際送入 Retriever 的 query。
        - unique_document_count: 跨輪去重後的文件數量。

    Returns:
        - IterativeRetrievalResult: 完整迭代結果與 diagnostics。
    """

    initial_query: str
    final_query: str
    rounds: list[RetrievalRoundTrace]
    stop_reason: str
    searched_queries: list[str]
    unique_document_count: int
    relation_plan: dict[str, object] = field(default_factory=dict)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    completeness_contracts: list[dict[str, Any]] = field(default_factory=list)
    absence_checks: list[dict[str, Any]] = field(default_factory=list)
    set_difference_derivations: list[dict[str, Any]] = field(default_factory=list)
    aggregation_derivations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WebSearchTrace:
    """
    保存單一 query 的網頁搜尋結果摘要。

    Args:
        - query: 送入搜尋後端的 query。
        - backend: 實際使用的搜尋後端。
        - result_count: 搜尋後端回傳的結果數。
        - source_ids: 轉換成 SearchSourceCandidate 的來源 ID。
        - notices: 搜尋後端回傳的提示或警告。

    Returns:
        - WebSearchTrace: Query generation 後的網頁搜尋紀錄。
    """

    query: str
    backend: str
    result_count: int
    source_ids: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    requested_source_kind: str = "web"
    requested_access_mode: str = "search"
    source_hint: str = ""
    actual_acquirer: str = ""
    fallback_used: bool = False
    required_content: str = "html_text"


@dataclass
class WebRetrievalResult:
    """
    保存從 query generation 到多輪 retrieval 的完整結果。

    Args:
        - question: 原始問題。
        - generated_queries: Query Generator 產生的搜尋 query。
        - salient_spans: Query generation 使用的高語意影響 spans。
        - web_searches: 各 query 的搜尋後端紀錄。
        - corpus_path: 清洗、分割後的 JSONL corpus。
        - embedding_path: Passage embeddings 與 FAISS index 目錄。
        - corpus_record_count: 動態 corpus 的 passage 數量。
        - retrieval: Labeler、Filter 與 next-hop retrieval 結果。
        - diagnostics: Bootstrap 錯誤與模型資訊。

    Returns:
        - WebRetrievalResult: 可直接保存為實驗 trace 的端到端結果。
    """

    question: str
    generated_queries: list[str]
    salient_spans: list[str]
    web_searches: list[WebSearchTrace]
    corpus_path: str
    embedding_path: str
    corpus_record_count: int
    retrieval: IterativeRetrievalResult | None
    diagnostics: dict[str, object] = field(default_factory=dict)
    blocked_sources: list[SearchSourceCandidate] = field(default_factory=list)


class IterativeRetrievalControl:
    """
    執行 Retriever、FAISS、Labeler 與 Filter 的多輪檢索流程。

    Args:
        - retriever: 已載入 corpus、passage embeddings 與 FAISS index 的 Retriever。
        - labeler: 判斷 chunk 為 CONTINUE 或 TERMINATE 的 EfficientRAG Labeler。
        - next_hop_composer: 根據問題與 bridge evidence 組合下一輪 query。
        - max_iter: 最大 retrieval 輪數，預設為 4。
        - top_k: 每輪從 FAISS 取得的 passage 數量。

    Returns:
        - IterativeRetrievalControl: 支援停止保護與完整 trace 的迭代控制器。
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        labeler: EfficientRAGLabelerAdapter | None = None,
        next_hop_composer: NextHopQueryComposer | None = None,
        next_hop_evidence_selector: NextHopEvidenceSelector | None = None,
        coverage_assessor: CoverageAssessor | None = None,
        sufficiency_gate: EvidenceSufficiencyGate | None = None,
        evidence_utility_gate: Any | None = None,
        evidence_contract_builder: EvidenceRoleContractBuilder | None = None,
        direct_evidence_promoter: DirectEvidencePromoter | None = None,
        span_grounder: CandidateSpanGrounder | None = None,
        span_quality_gate: CandidateSpanQualityGate | None = None,
        span_finalizer: RoleAwareSpanFinalizer | None = None,
        span_role_classifier: SpanRoleClassifier | None = None,
        semantic_fact_extractor: SemanticFactExtractor | None = None,
        cross_context_assembler: CrossContextAssembler | None = None,
        cross_context_fact_extractor: CrossContextFactExtractor | None = None,
        labeler_input_builder: LabelerInputBuilder | None = None,
        passage_evidence_unit_builder: PassageEvidenceUnitBuilder | None = None,
        bypass_labeler: bool = False,
        intent_state_tracker: SearchIntentStateTracker | None = None,
        query_guard: NextHopQueryGuard | None = None,
        max_iter: int = 4,
        top_k: int = 16,
        candidate_pool_size: int = 30,
        passage_selector: PassageCandidateSelector | None = None,
        min_retrieval_score: float = 0.75,
        relative_score_margin: float = 0.08,
        relation_binder: RelationEvidenceBinder | None = None,
        relation_resolver: RelationGoalResolver | None = None,
        goal_completion_evaluator: GoalCompletionEvaluator | None = None,
        recovery_policy: RetrievalRecoveryPolicy | None = None,
        external_source_loader: Callable[[list[SearchQueryRequest]], int] | None = None,
        max_relation_branches: int = 2,
        anchor_span_role_in_extraction: bool = True,
    ) -> None:
        self.retriever = retriever
        self.bypass_labeler = bool(bypass_labeler)
        self.labeler = (
            labeler
            if labeler is not None
            else (None if self.bypass_labeler else EfficientRAGLabelerAdapter())
        )
        self.next_hop_composer = next_hop_composer or NextHopQueryComposer()
        self.next_hop_evidence_selector = (
            next_hop_evidence_selector or NextHopEvidenceSelector()
        )
        self.label_contract_validator = LabelContractValidator()
        self.coverage_assessor = coverage_assessor or CoverageAssessor()
        self.sufficiency_gate = sufficiency_gate or EvidenceSufficiencyGate()
        self.evidence_utility_gate = evidence_utility_gate or EvidenceUtilityGate()
        self.evidence_contract_builder = (
            evidence_contract_builder or EvidenceRoleContractBuilder()
        )
        self.direct_evidence_promoter = (
            direct_evidence_promoter or DirectEvidencePromoter()
        )
        self.anchor_span_role_in_extraction = bool(anchor_span_role_in_extraction)
        self.span_grounder = span_grounder or CandidateSpanGrounder()
        self.span_quality_gate = span_quality_gate or CandidateSpanQualityGate()
        self.span_finalizer = span_finalizer or RoleAwareSpanFinalizer()
        self.span_role_classifier = span_role_classifier or SpanRoleClassifier()
        self.semantic_fact_extractor = (
            semantic_fact_extractor or SemanticFactExtractor()
        )
        self.cross_context_assembler = (
            cross_context_assembler or CrossContextAssembler()
        )
        self.cross_context_fact_extractor = (
            cross_context_fact_extractor or CrossContextFactExtractor()
        )
        self.labeler_input_builder = labeler_input_builder or LabelerInputBuilder()
        self.passage_evidence_unit_builder = (
            passage_evidence_unit_builder or PassageEvidenceUnitBuilder()
        )
        self.intent_state_tracker = intent_state_tracker or SearchIntentStateTracker()
        self.query_guard = query_guard or NextHopQueryGuard()
        self.passage_selector = passage_selector or PassageCandidateSelector()
        self.max_iter = max(1, max_iter)
        self.top_k = max(1, top_k)
        self.candidate_pool_size = max(self.top_k, candidate_pool_size)
        self.min_retrieval_score = max(0.0, min(1.0, min_retrieval_score))
        self.relative_score_margin = max(
            0.0,
            min(1.0, relative_score_margin),
        )
        self.relation_resolver = relation_resolver or RelationGoalResolver()
        self.relation_binder = relation_binder or RelationEvidenceBinder(
            resolver=self.relation_resolver
        )
        self.external_source_loader = external_source_loader
        self.max_relation_branches = max(1, max_relation_branches)
        self.goal_completion_evaluator = (
            goal_completion_evaluator or GoalCompletionEvaluator()
        )
        self.recovery_policy = recovery_policy or RetrievalRecoveryPolicy()

    def run(
        self,
        query: str,
        *,
        intent_plan: SearchIntentPlan | None = None,
    ) -> IterativeRetrievalResult:
        """
        執行最多 max_iter 輪的 retrieval、label 與 query filtering。

        Args:
            - query: 第一輪送入 Retriever 的原始 query。

        Returns:
            - IterativeRetrievalResult: 各輪文件、標籤、分數及停止原因。
        """
        initial_query = normalize_text(query)
        if not initial_query:
            return IterativeRetrievalResult(
                initial_query="",
                final_query="",
                rounds=[],
                stop_reason="empty_initial_query",
                searched_queries=[],
                unique_document_count=0,
            )

        current_query = initial_query
        current_queries = [initial_query]
        use_intent_state = intent_plan is not None
        current_intent_plan = intent_plan
        current_relation_plan = (
            intent_plan.relation_plan if intent_plan is not None else None
        )
        rounds: list[RetrievalRoundTrace] = []
        searched_queries: list[str] = []
        seen_query_keys: set[str] = set()
        seen_document_ids: set[str] = set()
        seen_chunk_keys: set[str] = set()
        unique_document_count = 0
        stop_reason = "goal_incomplete_budget_exhausted"
        attempted_recoveries: set[str] = set()
        self._last_coverage_bridge_terms: list[str] = []
        self._last_coverage_missing_constraints: list[str] = []
        retrieval_top_k = self.top_k
        retrieval_candidate_pool_size = self.candidate_pool_size
        answer_gate_satisfied = False

        for round_index in range(1, self.max_iter + 1):
            active_queries: list[str] = []
            active_query_keys: list[str] = []
            for candidate_query in current_queries:
                candidate = normalize_text(candidate_query)
                key = self._query_key(candidate)
                if not key or key in seen_query_keys:
                    continue
                active_queries.append(candidate)
                active_query_keys.append(key)
            if not active_queries:
                round_trace = RetrievalRoundTrace(
                    round_index=round_index,
                    query=current_query,
                    branch_queries=list(current_queries),
                )
                round_trace.filter_metadata["next_query_failure"] = (
                    "duplicate_query" if self._query_key(current_query) else "empty_query"
                )
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break
            current_query = active_queries[0]
            if not current_query:
                stop_reason = "empty_query"
                break

            seen_query_keys.update(active_query_keys)
            searched_queries.extend(active_queries)
            retrieved = self._search_with_scores_many(
                active_queries,
                original_question=initial_query,
                top_k=retrieval_top_k,
                candidate_pool_size=retrieval_candidate_pool_size,
            )
            round_trace = RetrievalRoundTrace(
                round_index=round_index,
                query=current_query,
                branch_queries=list(active_queries),
            )

            label_documents: list[dict[str, Any]] = []
            label_trace_indexes: list[int] = []
            for document, score in retrieved:
                trace = self._document_trace(document, score)
                duplicate_reason = self._duplicate_reason(
                    document=document,
                    seen_document_ids=seen_document_ids,
                    seen_chunk_keys=seen_chunk_keys,
                )
                if duplicate_reason:
                    trace.duplicate = True
                    trace.duplicate_reason = duplicate_reason
                else:
                    self._remember_document(
                        document=document,
                        seen_document_ids=seen_document_ids,
                        seen_chunk_keys=seen_chunk_keys,
                    )
                    unique_document_count += 1
                    label_documents.append(document)
                    label_trace_indexes.append(len(round_trace.documents))
                round_trace.documents.append(trace)

            if not label_documents:
                if use_intent_state and current_intent_plan is not None:
                    round_trace.coverage = {
                        "intent_state": current_intent_plan.to_dict(),
                        "score_based_sufficient": False,
                        "sufficient": current_intent_plan.state == "sufficient",
                    }
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break

            if self.bypass_labeler:
                self._apply_labeler_bypass(
                    round_trace=round_trace,
                    question=initial_query,
                    documents=label_documents,
                    trace_indexes=label_trace_indexes,
                )
            else:
                labeler_batch = self.labeler_input_builder.build_batch(
                    question=initial_query,
                    current_query=current_query,
                    documents=label_documents,
                    intent_plan=current_intent_plan,
                )
                if self.labeler is None:
                    raise RuntimeError("Labeler is required when bypass_labeler is false.")
                label_results = self.labeler.label_texts(
                    question=labeler_batch.question_context,
                    texts=labeler_batch.texts,
                )
                for trace_index, result, prepared in zip(
                    label_trace_indexes,
                    label_results,
                    labeler_batch.documents,
                ):
                    result.metadata = {
                        **result.metadata,
                        **prepared.diagnostics,
                    }
                    self._apply_label(
                        round_trace.documents[trace_index],
                        result,
                        question=initial_query,
                        intent_plan=current_intent_plan,
                    )

            self._apply_span_role_classification(
                round_trace=round_trace,
                question=initial_query,
                intent_plan=current_intent_plan,
            )
            self._apply_cross_context_fact_extraction(
                round_trace=round_trace,
                question=initial_query,
                intent_plan=current_intent_plan,
            )
            self._refresh_page_traces(round_trace)

            relation_requires_more = bool(
                current_relation_plan is not None
                and current_relation_plan.goals
                and not current_relation_plan.complete
            )
            if relation_requires_more and current_relation_plan is not None:
                relation_documents = [
                    document
                    for observed_round in [*rounds, round_trace]
                    for document in observed_round.documents
                    if not document.duplicate
                ]
                direct_contracts = [
                    contract
                    for document in relation_documents
                    for contract in document.direct_contracts
                ]
                resolved_goal_ids: list[str] = []
                activated_goal_ids: list[str] = []
                transitions: list[dict[str, str]] = []
                bound_evidence: list[Any] = []
                rejected_bindings: list[str] = []
                rejected_direct_contracts: list[dict[str, str]] = []
                for _ in range(len(current_relation_plan.goals)):
                    direct_resolution = self.relation_resolver.resolve_direct(
                        current_relation_plan,
                        direct_contracts,
                    )
                    rejected_direct_contracts.extend(
                        direct_resolution.rejected_contracts
                    )
                    if direct_resolution.resolved_goal_ids:
                        current_relation_plan = direct_resolution.plan
                        resolved_goal_ids.extend(direct_resolution.resolved_goal_ids)
                        transitions.extend(direct_resolution.transitions)
                        if direct_resolution.activated_goal_id:
                            activated_goal_ids.append(
                                direct_resolution.activated_goal_id
                            )
                        continue

                    binding = self.relation_binder.bind(
                        plan=current_relation_plan,
                        documents=relation_documents,
                    )
                    bound_evidence.extend(binding.evidence)
                    rejected_bindings.extend(binding.rejected)
                    resolution = self.relation_resolver.resolve(
                        current_relation_plan,
                        binding.evidence,
                    )
                    if not resolution.resolved_goal_ids:
                        break
                    current_relation_plan = resolution.plan
                    resolved_goal_ids.extend(resolution.resolved_goal_ids)
                    transitions.extend(resolution.transitions)
                    if resolution.activated_goal_id:
                        activated_goal_ids.append(resolution.activated_goal_id)

                if current_intent_plan is not None:
                    current_intent_plan = current_intent_plan.replace(
                        relation_plan=current_relation_plan
                    )
                round_trace.filter_metadata = {
                    **round_trace.filter_metadata,
                    "direct_evidence_resolution": {
                        "resolved_goal_ids": [
                            item["goal_id"]
                            for item in transitions
                            if item.get("resolution_type") == "direct"
                        ],
                        "contract_count": len(direct_contracts),
                        "rejected_contracts": rejected_direct_contracts,
                        "plan": current_relation_plan.to_dict(),
                    },
                    "relation_binding": {
                        "evidence": [item.to_dict() for item in bound_evidence],
                        "rejected": list(rejected_bindings),
                        "resolved_goal_ids": self._dedupe_tokens(resolved_goal_ids),
                        "activated_goal_id": current_relation_plan.active_goal_id,
                        "activated_goal_ids": self._dedupe_tokens(activated_goal_ids),
                        "transitions": transitions,
                        "plan": current_relation_plan.to_dict(),
                    },
                }
                if (
                    not current_relation_plan.complete
                    and round_index < self.max_iter
                ):
                    transition_documents = self._documents_for_resolved_goals(
                        plan=current_relation_plan,
                        resolved_goal_ids=resolved_goal_ids,
                        documents=round_trace.documents,
                    )
                    round_trace.filter_metadata["relation_next_hop"] = {
                        "grounded_transition_document_ids": [
                            item.document_id for item in transition_documents
                        ],
                    }
                    if transition_documents:
                        relation_requests = (
                            self.next_hop_composer.build_relation_requests(
                                relation_plan=current_relation_plan,
                                constraints=(
                                    list(current_intent_plan.must_include or [])
                                    if current_intent_plan is not None
                                    else []
                                ),
                                answer_requirement=self._answer_requirement(
                                    current_intent_plan
                                ),
                                original_question=initial_query,
                                seen_query_keys=seen_query_keys,
                                max_requests=self.max_relation_branches,
                            )
                        )
                        requests = [item.request for item in relation_requests]
                        next_queries = [request.query for request in requests]
                        round_trace.next_queries = list(next_queries)
                        round_trace.next_query = (
                            next_queries[0] if next_queries else ""
                        )
                        round_trace.filter_metadata["relation_next_hop"][
                            "branches"
                        ] = [item.to_dict() for item in relation_requests]
                        added_count = self._load_external_sources(requests)
                        round_trace.filter_metadata["relation_next_hop"][
                            "added_record_count"
                        ] = added_count
                        if next_queries and added_count > 0:
                            self._record_next_hop_decision(
                                round_trace,
                                required=True,
                                reason="grounded_relation_bridge",
                                documents=transition_documents,
                                generated_query=next_queries[0],
                                relation_plan=current_relation_plan,
                            )
                            round_trace.stop_reason = "relation_next_hop"
                            rounds.append(round_trace)
                            current_queries = next_queries
                            current_query = next_queries[0]
                            continue
                        failure_reason = (
                            "relation_source_empty"
                            if next_queries
                            else "empty_relation_query"
                        )
                    else:
                        failure_reason = "no_grounded_relation_transition"
                    round_trace.filter_metadata["relation_next_hop"][
                        "failure_reason"
                    ] = failure_reason

            relation_requires_more = bool(
                current_relation_plan is not None
                and current_relation_plan.goals
                and not current_relation_plan.complete
            )

            non_duplicate_documents = [
                document
                for document in round_trace.documents
                if not document.duplicate
            ]
            coverage = self.coverage_assessor.assess(
                question=initial_query,
                documents=non_duplicate_documents,
                intent_plan=current_intent_plan,
            )
            self._last_coverage_bridge_terms = list(coverage.bridge_terms or [])
            self._last_coverage_missing_constraints = list(
                coverage.missing_constraints or []
            )
            gate_result = self.sufficiency_gate.assess(
                question=initial_query,
                documents=non_duplicate_documents,
                intent_plan=current_intent_plan,
                coverage=coverage,
            )
            answer_gate_satisfied = answer_gate_satisfied or gate_result.sufficient
            score_based_sufficient = bool(coverage.sufficient)
            if coverage.sufficient and not gate_result.sufficient:
                coverage = replace(
                    coverage,
                    sufficient=False,
                    missing_constraints=self._dedupe_tokens(
                        list(coverage.missing_constraints or [])
                        + list(gate_result.missing or [])
                    ),
                    trigger_reason=f"sufficiency_gate_failed:{gate_result.reason}",
                )
            if use_intent_state and current_intent_plan is not None:
                current_intent_plan = self.intent_state_tracker.update(
                    plan=current_intent_plan,
                    question=initial_query,
                    documents=non_duplicate_documents,
                )
                intent_state_before_gate = current_intent_plan
                intent_state_sufficient = current_intent_plan.state == "sufficient"
                if intent_state_sufficient and not gate_result.sufficient:
                    current_intent_plan = current_intent_plan.replace(
                        state="needs_next_hop",
                        missing_terms=self._dedupe_tokens(
                            list(current_intent_plan.missing_terms or [])
                            + [f"answer_support:{gate_result.answer_role}"]
                        ),
                    )
                round_trace.coverage = {
                    **coverage.to_dict(),
                    "intent_state": current_intent_plan.to_dict(),
                    "score_based_sufficient": score_based_sufficient,
                    "intent_state_before_gate": intent_state_before_gate.to_dict(),
                    "sufficiency_gate": gate_result.to_dict(),
                    "sufficient": current_intent_plan.state == "sufficient"
                    and gate_result.sufficient,
                }
                if (
                    intent_state_sufficient
                    and gate_result.sufficient
                    and not relation_requires_more
                ):
                    round_trace.filter_metadata["legacy_sufficiency_signal"] = (
                        "intent_state_sufficient"
                    )
                if intent_state_sufficient and not gate_result.sufficient:
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        "sufficiency_gate_failed": gate_result.to_dict(),
                    }
            else:
                round_trace.coverage = {
                    **coverage.to_dict(),
                    "score_based_sufficient": score_based_sufficient,
                    "sufficiency_gate": gate_result.to_dict(),
                    "sufficient": coverage.sufficient and gate_result.sufficient,
                }
                if (
                    coverage.sufficient
                    and gate_result.sufficient
                    and not relation_requires_more
                ):
                    round_trace.filter_metadata["legacy_sufficiency_signal"] = (
                        "coverage_sufficient"
                    )
                if coverage.sufficient and not gate_result.sufficient:
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        "sufficiency_gate_failed": gate_result.to_dict(),
                    }

            observed_documents = [
                document
                for previous_round in rounds
                for document in previous_round.documents
                if not document.duplicate
            ] + non_duplicate_documents
            goal_completion = self.goal_completion_evaluator.evaluate(
                relation_plan=current_relation_plan,
                documents=observed_documents,
                corpus_documents=self.retriever.passage_map.values(),
                answer_gate_sufficient=answer_gate_satisfied,
                fact_store=self._observed_fact_store(
                    documents=observed_documents,
                    rounds=rounds,
                ),
                question=initial_query,
                answer_requirement=self._answer_requirement(current_intent_plan),
            )
            current_relation_plan = goal_completion.relation_plan
            if current_intent_plan is not None:
                current_intent_plan = current_intent_plan.replace(
                    relation_plan=current_relation_plan
                )
            self._apply_negative_verification_contracts(
                documents=observed_documents,
                goal_completion=goal_completion,
                question=initial_query,
            )
            round_trace.filter_metadata["goal_completion"] = (
                goal_completion.to_dict()
            )
            round_trace.coverage["sufficient"] = goal_completion.sufficient
            if goal_completion.sufficient:
                self._record_next_hop_decision(
                    round_trace,
                    required=False,
                    reason="goal_completed",
                    relation_plan=current_relation_plan,
                )
                round_trace.stop_reason = "goal_completion_sufficient"
                rounds.append(round_trace)
                stop_reason = round_trace.stop_reason
                break

            continue_documents = [
                trace
                for trace in round_trace.documents
                if (
                    not trace.duplicate
                    and trace.valid_for_next_hop
                    and self._grounded_bridge_contracts(trace)
                )
            ]
            if not continue_documents:
                self._record_next_hop_decision(
                    round_trace,
                    required=False,
                    reason="no_grounded_bridge",
                    relation_plan=current_relation_plan,
                )
                fallback_result = self._try_fallback_next_query(
                    query=initial_query,
                    documents=[
                        document
                        for document in round_trace.documents
                        if (
                            not document.duplicate
                            and self._grounded_bridge_contracts(document)
                        )
                    ],
                    reason="no_continue_chunks",
                    intent_plan=current_intent_plan,
                )
                if fallback_result is not None and round_index < self.max_iter:
                    next_query = self._guard_next_query(
                        original_question=initial_query,
                        current_query=current_query,
                        result=fallback_result,
                        round_trace=round_trace,
                        intent_plan=current_intent_plan,
                        seen_query_keys=seen_query_keys,
                    )
                    round_trace.next_query = next_query
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        **fallback_result.metadata,
                        "fallback_used": True,
                        "kept_question_tokens": fallback_result.kept_question_tokens,
                        "kept_evidence_tokens": fallback_result.kept_evidence_tokens,
                    }
                    if next_query and not self._is_duplicate_query(next_query, seen_query_keys):
                        self._record_next_hop_decision(
                            round_trace,
                            required=True,
                            reason="grounded_bridge_fallback_next_hop",
                            documents=continue_documents,
                            generated_query=next_query,
                            relation_plan=current_relation_plan,
                        )
                        round_trace.stop_reason = "fallback_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break

            best_retrieval_score = max(
                document.retrieval_score
                for document in continue_documents
            )
            relative_threshold = max(
                self.min_retrieval_score,
                best_retrieval_score - self.relative_score_margin,
            )
            qualified_documents = [
                document
                for document in continue_documents
                if (
                    self._grounded_bridge_contracts(document)
                    and document.retrieval_score >= relative_threshold
                )
            ]
            round_trace.filter_metadata = {
                **round_trace.filter_metadata,
                "continue_document_count": len(continue_documents),
                "qualified_document_count": len(qualified_documents),
                "min_retrieval_score": self.min_retrieval_score,
                "relative_score_margin": self.relative_score_margin,
                "best_continue_score": round(best_retrieval_score, 6),
                "effective_retrieval_threshold": round(
                    relative_threshold,
                    6,
                ),
            }
            if not qualified_documents:
                self._record_next_hop_decision(
                    round_trace,
                    required=False,
                    reason="grounded_bridge_below_retrieval_threshold",
                    documents=continue_documents,
                    relation_plan=current_relation_plan,
                )
                fallback_result = self._try_fallback_next_query(
                    query=initial_query,
                    documents=continue_documents,
                    reason="no_qualified_continue_chunks",
                    intent_plan=current_intent_plan,
                )
                if fallback_result is not None and round_index < self.max_iter:
                    next_query = self._guard_next_query(
                        original_question=initial_query,
                        current_query=current_query,
                        result=fallback_result,
                        round_trace=round_trace,
                        intent_plan=current_intent_plan,
                        seen_query_keys=seen_query_keys,
                    )
                    round_trace.next_query = next_query
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        **fallback_result.metadata,
                        "fallback_used": True,
                        "kept_question_tokens": fallback_result.kept_question_tokens,
                        "kept_evidence_tokens": fallback_result.kept_evidence_tokens,
                    }
                    if next_query and not self._is_duplicate_query(next_query, seen_query_keys):
                        self._record_next_hop_decision(
                            round_trace,
                            required=True,
                            reason="grounded_bridge_fallback_next_hop",
                            documents=continue_documents,
                            generated_query=next_query,
                            relation_plan=current_relation_plan,
                        )
                        round_trace.stop_reason = "fallback_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break

            useful_tokens = self._dedupe_tokens(
                str(contract.get("bridge_span") or "")
                for document in qualified_documents
                for contract in self._grounded_bridge_contracts(document)
            )
            round_trace.useful_tokens = useful_tokens
            if not useful_tokens:
                self._record_next_hop_decision(
                    round_trace,
                    required=False,
                    reason="grounded_bridge_without_query_span",
                    documents=qualified_documents,
                    relation_plan=current_relation_plan,
                )
                fallback_result = self._try_fallback_next_query(
                    query=initial_query,
                    documents=continue_documents,
                    reason="no_bridge_tokens",
                    intent_plan=current_intent_plan,
                )
                if fallback_result is not None and round_index < self.max_iter:
                    next_query = self._guard_next_query(
                        original_question=initial_query,
                        current_query=current_query,
                        result=fallback_result,
                        round_trace=round_trace,
                        intent_plan=current_intent_plan,
                        seen_query_keys=seen_query_keys,
                    )
                    round_trace.next_query = next_query
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        **fallback_result.metadata,
                        "fallback_used": True,
                        "kept_question_tokens": fallback_result.kept_question_tokens,
                        "kept_evidence_tokens": fallback_result.kept_evidence_tokens,
                    }
                    if next_query and not self._is_duplicate_query(next_query, seen_query_keys):
                        self._record_next_hop_decision(
                            round_trace,
                            required=True,
                            reason="grounded_bridge_fallback_next_hop",
                            documents=continue_documents,
                            generated_query=next_query,
                            relation_plan=current_relation_plan,
                        )
                        round_trace.stop_reason = "coverage_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break

            filter_result = self._build_next_query(
                query=initial_query,
                documents=qualified_documents,
                intent_plan=current_intent_plan,
            )
            next_query = self._guard_next_query(
                original_question=initial_query,
                current_query=current_query,
                result=filter_result,
                round_trace=round_trace,
                intent_plan=current_intent_plan,
                seen_query_keys=seen_query_keys,
            )
            round_trace.next_query = next_query
            round_trace.filter_metadata = {
                **round_trace.filter_metadata,
                **filter_result.metadata,
                "fallback_used": filter_result.fallback_used,
                "kept_question_tokens": filter_result.kept_question_tokens,
                "kept_evidence_tokens": filter_result.kept_evidence_tokens,
            }

            if not next_query or self._is_duplicate_query(next_query, seen_query_keys):
                round_trace.filter_metadata["next_query_failure"] = (
                    "empty_next_query" if not next_query else "duplicate_next_query"
                )
                self._record_next_hop_decision(
                    round_trace,
                    required=False,
                    reason=round_trace.filter_metadata["next_query_failure"],
                    documents=qualified_documents,
                    relation_plan=current_relation_plan,
                )
                recovery = self._prepare_goal_recovery(
                    relation_plan=current_relation_plan,
                    attempted=attempted_recoveries,
                    top_k=retrieval_top_k,
                    candidate_pool_size=retrieval_candidate_pool_size,
                    original_question=initial_query,
                    round_trace=round_trace,
                    round_index=round_index,
                    seen_query_keys=seen_query_keys,
                )
                rounds.append(round_trace)
                if recovery is not None:
                    (
                        current_queries,
                        retrieval_top_k,
                        retrieval_candidate_pool_size,
                    ) = recovery
                    current_query = current_queries[0]
                    continue
                round_trace.stop_reason = "goal_incomplete_no_viable_recovery"
                stop_reason = round_trace.stop_reason
                break

            self._record_next_hop_decision(
                round_trace,
                required=True,
                reason="grounded_bridge_next_hop",
                documents=qualified_documents,
                generated_query=next_query,
                relation_plan=current_relation_plan,
            )
            round_trace.stop_reason = "grounded_bridge_next_hop"
            rounds.append(round_trace)
            self._load_external_sources([SearchQueryRequest.fallback(next_query)])
            current_query = next_query
            current_queries = [next_query]
        else:
            stop_reason = "goal_incomplete_budget_exhausted"

        final_query = (
            rounds[-1].next_query
            if rounds and rounds[-1].next_query
            else current_query
        )
        task_fact_store = TaskFactStore()
        task_fact_store.extend(
            EvidenceFact.from_dict(fact)
            for round_trace in rounds
            for fact in [
                *round_trace.cross_context_facts,
                *[
                    item
                    for document in round_trace.documents
                    for item in document.semantic_facts
                ],
            ]
            if isinstance(fact, dict)
        )
        for corpus_item in self.retriever.passage_map.values():
            for item in list(corpus_item.get("completeness_contracts") or []):
                if isinstance(item, dict):
                    task_fact_store.add_completeness_contract(
                        CompletenessContract.from_dict(item)
                    )
        for round_trace in rounds:
            goal_metadata = round_trace.filter_metadata.get("goal_completion")
            if not isinstance(goal_metadata, dict):
                continue
            for result in list(goal_metadata.get("negative_verifications") or []):
                if not isinstance(result, dict):
                    continue
                for item in list(result.get("completeness_contracts") or []):
                    if isinstance(item, dict):
                        task_fact_store.add_completeness_contract(
                            CompletenessContract.from_dict(item)
                        )
                for item in list(result.get("absence_checks") or []):
                    if isinstance(item, dict):
                        task_fact_store.add_absence_check(AbsenceCheck.from_dict(item))
            for item in list(goal_metadata.get("aggregation_derivations") or []):
                if isinstance(item, dict):
                    task_fact_store.add_aggregation_derivation(
                        AggregationDerivation.from_dict(item)
                    )
        return IterativeRetrievalResult(
            initial_query=initial_query,
            final_query=final_query,
            rounds=rounds,
            stop_reason=stop_reason,
            searched_queries=searched_queries,
            unique_document_count=unique_document_count,
            relation_plan=(
                current_relation_plan.to_dict()
                if current_relation_plan is not None
                else {}
            ),
            semantic_facts=[fact.to_dict() for fact in task_fact_store.all()],
            completeness_contracts=[
                item.to_dict() for item in task_fact_store.completeness_contracts()
            ],
            absence_checks=[
                item.to_dict() for item in task_fact_store.absence_checks()
            ],
            set_difference_derivations=[
                item.to_dict()
                for item in task_fact_store.set_difference_derivations()
            ],
            aggregation_derivations=[
                item.to_dict() for item in task_fact_store.aggregation_derivations()
            ],
        )

    def _observed_fact_store(
        self,
        *,
        documents: list[RetrievedDocumentTrace],
        rounds: list[RetrievalRoundTrace],
    ) -> TaskFactStore:
        store = TaskFactStore()
        store.extend(
            EvidenceFact.from_dict(item)
            for document in documents
            for item in document.semantic_facts
            if isinstance(item, dict)
        )
        for corpus_item in self.retriever.passage_map.values():
            for value in list(corpus_item.get("completeness_contracts") or []):
                if isinstance(value, dict):
                    store.add_completeness_contract(
                        CompletenessContract.from_dict(value)
                    )
        for previous_round in rounds:
            metadata = previous_round.filter_metadata.get("goal_completion")
            if not isinstance(metadata, dict):
                continue
            for result in list(metadata.get("negative_verifications") or []):
                if not isinstance(result, dict):
                    continue
                for value in list(result.get("completeness_contracts") or []):
                    if isinstance(value, dict):
                        store.add_completeness_contract(
                            CompletenessContract.from_dict(value)
                        )
        return store

    def _load_external_sources(self, requests: list[SearchQueryRequest]) -> int:
        if self.external_source_loader is None or not requests:
            return 0
        try:
            return max(0, int(self.external_source_loader(requests)))
        except Exception:
            return 0

    def _apply_negative_verification_contracts(
        self,
        *,
        documents: list[RetrievedDocumentTrace],
        goal_completion: Any,
        question: str,
    ) -> None:
        by_id = {document.document_id: document for document in documents}
        for result in goal_completion.negative_verifications:
            negative_by_title = {
                normalize_text(fact.subject).casefold(): fact
                for fact in result.negative_facts
            }
            if not result.resolved:
                continue
            for verification in result.verifications:
                if verification.status != "absent_verified":
                    continue
                document = by_id.get(verification.document_id)
                if document is None:
                    continue
                negative_fact = negative_by_title.get(
                    normalize_text(verification.title).casefold()
                )
                if negative_fact is not None:
                    serialized_fact = negative_fact.to_dict()
                    if serialized_fact not in document.semantic_facts:
                        document.semantic_facts.append(serialized_fact)
                contract = {
                    "goal_id": result.goal_id,
                    "answer_span": verification.title,
                    "context": (
                        f"Complete document verified absent term: {verification.target}"
                    ),
                    "document_id": verification.document_id,
                    "source_title": verification.title,
                    "url": document.url,
                    "answer_requirement": normalize_text(question),
                    "verification_scope": "full_document",
                    "fact_id": negative_fact.fact_id if negative_fact else "",
                    "polarity": "negative",
                    "grounding_status": "grounded",
                    "answer_binding": "direct",
                }
                if contract not in document.direct_contracts:
                    document.direct_contracts.append(contract)
                document.valid_for_evidence = True
                document.support_level = "direct"

    def _attempt_goal_recovery(
        self,
        *,
        relation_plan: Any,
        attempted: set[str],
        top_k: int,
        candidate_pool_size: int,
        original_question: str,
        round_trace: RetrievalRoundTrace,
    ):
        history: list[dict[str, Any]] = []
        for _ in range(4):
            decision = self.recovery_policy.decide(
                relation_plan=relation_plan,
                corpus_documents=self.retriever.passage_map.values(),
                attempted=attempted,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                original_question=original_question,
                bridge_terms=list(
                    getattr(self, "_last_coverage_bridge_terms", []) or []
                ),
                missing_constraints=list(
                    getattr(self, "_last_coverage_missing_constraints", []) or []
                ),
            )
            attempted.add(decision.fingerprint)
            entry = decision.to_dict()
            if not decision.viable:
                history.append(entry)
                round_trace.filter_metadata["goal_recovery"] = history
                return decision
            if decision.action == "expand_retrieval":
                history.append(entry)
                round_trace.filter_metadata["goal_recovery"] = history
                return decision

            added_count = self._load_external_sources(decision.requests)
            entry["added_record_count"] = added_count
            history.append(entry)
            if added_count > 0:
                if decision.action in {"direct_fetch", "browser"}:
                    decision = replace(
                        decision,
                        next_queries=[normalize_text(original_question)],
                    )
                round_trace.filter_metadata["goal_recovery"] = history
                return decision
        round_trace.filter_metadata["goal_recovery"] = history
        return self.recovery_policy.decide(
            relation_plan=relation_plan,
            corpus_documents=self.retriever.passage_map.values(),
            attempted=attempted,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            original_question=original_question,
            bridge_terms=list(
                getattr(self, "_last_coverage_bridge_terms", []) or []
            ),
            missing_constraints=list(
                getattr(self, "_last_coverage_missing_constraints", []) or []
            ),
        )

    def _prepare_goal_recovery(
        self,
        *,
        relation_plan: Any,
        attempted: set[str],
        top_k: int,
        candidate_pool_size: int,
        original_question: str,
        round_trace: RetrievalRoundTrace,
        round_index: int,
        seen_query_keys: set[str],
    ) -> tuple[list[str], int, int] | None:
        if round_index >= self.max_iter:
            return None
        decision = self._attempt_goal_recovery(
            relation_plan=relation_plan,
            attempted=attempted,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            original_question=original_question,
            round_trace=round_trace,
        )
        next_queries = [
            normalize_text(query) for query in decision.next_queries if normalize_text(query)
        ]
        if not decision.viable or not next_queries:
            return None
        query_reuse_allowed = decision.action in {
            "expand_retrieval",
            "direct_fetch",
            "browser",
        }
        if query_reuse_allowed:
            for query in next_queries:
                seen_query_keys.discard(self._query_key(query))
        elif any(self._query_key(query) in seen_query_keys for query in next_queries):
            round_trace.filter_metadata["goal_recovery_rejected"] = {
                "action": decision.action,
                "reason": "repeated_query_without_retrieval_change",
                "queries": list(next_queries),
            }
            return None
        round_trace.filter_metadata["goal_recovery_query_reuse"] = {
            "allowed": query_reuse_allowed,
            "action": decision.action,
            "reason": (
                "retrieval_state_changed"
                if query_reuse_allowed
                else "new_query_required"
            ),
        }
        round_trace.next_queries = list(next_queries)
        round_trace.next_query = next_queries[0]
        round_trace.stop_reason = f"goal_recovery:{decision.action}"
        return (
            next_queries,
            max(top_k, decision.top_k),
            max(candidate_pool_size, decision.candidate_pool_size),
        )

    def _search_with_scores_many(
        self,
        queries: list[str],
        *,
        original_question: str = "",
        top_k: int | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        active_top_k = max(1, int(top_k or self.top_k))
        active_candidate_pool_size = max(
            active_top_k,
            int(candidate_pool_size or self.candidate_pool_size),
        )
        ranked_dense_lists: dict[str, list[tuple[str, float]]] = {}
        for index, query in enumerate(queries, start=1):
            ranked_dense_lists[f"dense_branch_{index}"] = self._dense_rank(
                query,
                active_candidate_pool_size,
            )
        if all(
            normalize_text(original_question).casefold()
            != normalize_text(query).casefold()
            for query in queries
        ):
            ranked_dense_lists["dense_original"] = self._dense_rank(
                original_question,
                min(20, active_candidate_pool_size),
            )
        selections = self.passage_selector.select(
            passage_map=self.retriever.passage_map,
            ranked_dense_lists=ranked_dense_lists,
            lexical_query=" ".join([*queries, original_question]),
            max_items=active_top_k,
        )
        results: list[tuple[dict[str, Any], float]] = []
        for selection in selections:
            document = dict(selection.document)
            document["_selection_score"] = selection.fusion_score
            document["_selection_sources"] = list(selection.selection_sources)
            document["_expanded_from"] = selection.expanded_from
            results.append((document, selection.retrieval_score))
        return results

    def _search_with_scores(
        self,
        query: str,
        *,
        original_question: str = "",
    ) -> list[tuple[dict[str, Any], float]]:
        return self._search_with_scores_many(
            [query],
            original_question=original_question,
        )

    def _dense_rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        # Delegate the prefix convention to the embedder: E5 needs a "query: "
        # prefix, bge-m3 is degraded by one. Deciding it here as well meant a
        # new model had to be registered in two places to be encoded correctly.
        # Minimal embedders (including test doubles) may not implement it.
        prepare = getattr(self.retriever.embedder, "prepare_query_text", None)
        prepared_query = prepare(query) if callable(prepare) else query
        query_vector = self.retriever.embedder.embed([prepared_query])
        search_results = self.retriever.index.search(query_vector, top_k)
        if not search_results:
            return []
        document_ids, scores = search_results[0]
        return [
            (document_id, float(score))
            for document_id, score in zip(document_ids, scores, strict=False)
            if document_id in self.retriever.passage_map
        ]

    def _build_next_query(
        self,
        *,
        query: str,
        documents: list[RetrievedDocumentTrace],
        intent_plan: SearchIntentPlan | None = None,
    ) -> NextHopQueryResult:
        selected_spans = self._dedupe_tokens(
            str(contract.get("bridge_span") or "")
            for document in documents
            if document.valid_for_next_hop
            for contract in self._grounded_bridge_contracts(document)
        )[:3]
        if not selected_spans:
            return NextHopQueryResult(
                query="",
                kept_question_tokens=[],
                kept_evidence_tokens=[],
                fallback_used=False,
                metadata={
                    "method": "bridge_spans_next_hop",
                    "filter_model_used": False,
                    "selected_bridge_spans": [],
                    "empty_reason": "no_selected_bridge_spans",
                },
            )
        evidence_items = [
            EvidenceItem(
                evidence_id=f"R{index}",
                source_id=f"next-hop-selection-{index}",
                query_id="iterative_retrieval",
                title="",
                text=" ".join(selected_spans),
                url="",
                matched_terms=selected_spans,
                evidence_quality=0.0,
                cleaning_reasons=["efficientrag_labeler:valid_continue"],
            )
            for index in range(1, 2)
        ]
        result = self.next_hop_composer.build_query(
            question=query,
            evidence_items=evidence_items,
            intent_plan=intent_plan,
        )
        result.metadata = {
            **result.metadata,
            "method": "bridge_spans_next_hop",
            "selected_bridge_spans": selected_spans,
        }
        return result

    def _guard_next_query(
        self,
        *,
        original_question: str,
        current_query: str,
        result: NextHopQueryResult,
        round_trace: RetrievalRoundTrace,
        intent_plan: SearchIntentPlan | None,
        seen_query_keys: set[str],
    ) -> str:
        useful_spans = list(result.kept_evidence_tokens or [])
        if not str(result.metadata.get("method", "")).startswith("external_"):
            useful_spans.extend(round_trace.useful_tokens or [])
            for document in round_trace.documents:
                useful_spans.extend(document.bridge_spans or [])
        guard_result = self.query_guard.validate(
            original_question=original_question,
            current_query=current_query,
            proposed_next_query=result.query,
            intent_plan=intent_plan,
            useful_spans=useful_spans,
            seen_query_keys=seen_query_keys,
        )
        selected_query = normalize_text(guard_result.query)
        external_fallback_disabled = False
        if (
            not guard_result.accepted
            and str(result.metadata.get("method", "")).startswith("external_")
        ):
            proposed_key = self._query_key(result.query)
            selected_key = self._query_key(selected_query)
            if not selected_query or selected_key == proposed_key:
                selected_query = ""
                external_fallback_disabled = True
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "query_guard": guard_result.to_dict(),
            "query_guard_external_fallback_disabled": external_fallback_disabled,
        }
        return selected_query

    def _try_fallback_next_query(
        self,
        *,
        query: str,
        documents: list[RetrievedDocumentTrace],
        reason: str,
        intent_plan: SearchIntentPlan | None = None,
    ) -> NextHopQueryResult | None:
        candidates = [
            document
            for document in documents
            if (
                self._grounded_bridge_contracts(document)
                and document.retrieval_score > 0
                and document.valid_for_next_hop
            )
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.retrieval_score, reverse=True)
        best_score = candidates[0].retrieval_score
        threshold = max(0.0, best_score - self.relative_score_margin)
        selected = [
            document
            for document in candidates
            if document.retrieval_score >= threshold
        ][:3]
        if not selected:
            return None
        result = self._build_next_query(
            query=query,
            documents=selected,
            intent_plan=intent_plan,
        )
        if not result.query:
            return None
        result.fallback_used = True
        result.metadata = {
            **result.metadata,
            "method": "external_composer_fallback_next_query",
            "fallback_reason": reason,
            "fallback_document_count": len(selected),
            "best_retrieval_score": round(best_score, 6),
        }
        if self._query_key(result.query) == self._query_key(query):
            return None
        return result

    def _document_trace(
        self,
        document: dict[str, Any],
        score: float,
    ) -> RetrievedDocumentTrace:
        source_url = normalize_text(
            document.get("source_url", "") or document.get("url", "")
        )
        canonical_url = canonicalize_page_url(
            normalize_text(document.get("canonical_url", "")) or source_url
        )
        document_id = str(document.get("id", ""))
        page_id = normalize_text(document.get("page_id", "")) or build_page_id(
            canonical_url=canonical_url,
            fallback_identity=document_id,
        )
        return RetrievedDocumentTrace(
            document_id=document_id,
            title=normalize_text(document.get("title", "")),
            text=normalize_text(document.get("text", "")),
            url=normalize_text(document.get("url", "")),
            retrieval_score=round(float(score), 6),
            selection_score=round(
                float(document.get("_selection_score", 0.0) or 0.0),
                8,
            ),
            selection_sources=list(document.get("_selection_sources") or []),
            expanded_from=normalize_text(document.get("_expanded_from", "")),
            record_type=normalize_text(document.get("record_type", "")) or "passage",
            record_id=normalize_text(document.get("record_id", "")),
            record_fields=self._record_fields(document),
            content_scope=(
                normalize_text(document.get("content_scope", "")) or "passage"
            ),
            content_complete=bool(document.get("content_complete", False)),
            content_truncated=bool(document.get("content_truncated", False)),
            original_content_chars=int(
                document.get("original_content_chars", 0) or 0
            ),
            required_content=(
                normalize_text(document.get("required_content", "")) or "html_text"
            ),
            acquisition_state=(
                normalize_text(document.get("acquisition_state", "")) or "pending"
            ),
            page_id=page_id,
            source_url=source_url,
            canonical_url=canonical_url,
            section_title=normalize_text(document.get("section_title", "")),
            section_index=self._metadata_int(document.get("section_index"), 0),
            passage_index=self._metadata_int(document.get("passage_index"), 0),
            content_type=(
                normalize_text(document.get("content_type", "")) or "text"
            ),
            table_id=normalize_text(document.get("table_id", "")),
            table_headers=[
                normalize_text(value)
                for value in list(document.get("table_headers") or [])
                if normalize_text(value)
            ],
            row_index=self._metadata_int(document.get("row_index"), -1),
        )

    def _refresh_page_traces(self, round_trace: RetrievalRoundTrace) -> None:
        grouped: dict[str, list[RetrievedDocumentTrace]] = {}
        for trace in round_trace.documents:
            if trace.duplicate:
                continue
            page_id = trace.page_id or build_page_id(
                canonical_url=trace.canonical_url,
                fallback_identity=trace.document_id,
            )
            grouped.setdefault(page_id, []).append(trace)

        pages: list[PageRetrievalTrace] = []
        for page_id, documents in grouped.items():
            direct_documents = [
                trace.document_id for trace in documents if trace.direct_contracts
            ]
            bridge_documents = [
                trace.document_id
                for trace in documents
                if self._grounded_bridge_contracts(trace)
            ]
            direct_fact_ids = self._dedupe_tokens(
                str(contract.get("fact_id") or contract.get("answer_span") or "")
                for trace in documents
                for contract in trace.direct_contracts
                if isinstance(contract, dict)
            )
            bridge_goal_ids = self._dedupe_tokens(
                str(contract.get("goal_id") or "")
                for trace in documents
                for contract in self._grounded_bridge_contracts(trace)
            )
            status = (
                "direct_found"
                if direct_documents
                else ("bridge_found" if bridge_documents else "no_usable_contract")
            )
            pages.append(
                PageRetrievalTrace(
                    page_id=page_id,
                    url=next(
                        (
                            trace.canonical_url or trace.url
                            for trace in documents
                            if trace.canonical_url or trace.url
                        ),
                        "",
                    ),
                    title=next(
                        (trace.title for trace in documents if trace.title),
                        "",
                    ),
                    status=status,
                    selection_scope="global",
                    selection_reasons=self._dedupe_tokens(
                        source
                        for trace in documents
                        for source in trace.selection_sources
                    ),
                    selected_passage_ids=[
                        trace.document_id for trace in documents
                    ],
                    direct_document_ids=direct_documents,
                    bridge_document_ids=bridge_documents,
                    direct_fact_ids=direct_fact_ids,
                    bridge_goal_ids=bridge_goal_ids,
                )
            )
        round_trace.pages = pages

    def _grounded_bridge_contracts(
        self,
        trace: RetrievedDocumentTrace,
    ) -> list[dict[str, Any]]:
        grounded: list[dict[str, Any]] = []
        text_key = normalize_text(trace.text).casefold()
        for contract in trace.bridge_contracts:
            if not isinstance(contract, dict):
                continue
            span = normalize_text(str(contract.get("bridge_span") or ""))
            document_id = normalize_text(str(contract.get("document_id") or ""))
            context = normalize_text(str(contract.get("context") or ""))
            if not span or span.casefold() not in text_key:
                continue
            if document_id and document_id != trace.document_id:
                continue
            if context and span.casefold() not in context.casefold():
                continue
            grounded.append(contract)
        return grounded

    def _documents_for_resolved_goals(
        self,
        *,
        plan: Any,
        resolved_goal_ids: Iterable[str],
        documents: Iterable[RetrievedDocumentTrace],
    ) -> list[RetrievedDocumentTrace]:
        resolved = {
            normalize_text(str(goal_id or ""))
            for goal_id in resolved_goal_ids
            if normalize_text(str(goal_id or ""))
        }
        if not resolved:
            return []
        evidence_ids = {
            normalize_text(str(evidence_id or ""))
            for goal in list(getattr(plan, "goals", []) or [])
            if normalize_text(str(getattr(goal, "goal_id", "") or "")) in resolved
            for evidence_id in list(getattr(goal, "evidence_ids", []) or [])
            if normalize_text(str(evidence_id or ""))
        }
        return [
            trace
            for trace in documents
            if not trace.duplicate and trace.document_id in evidence_ids
        ]

    def _record_next_hop_decision(
        self,
        round_trace: RetrievalRoundTrace,
        *,
        required: bool,
        reason: str,
        documents: Iterable[RetrievedDocumentTrace] = (),
        generated_query: str = "",
        relation_plan: Any | None = None,
    ) -> None:
        bridge_documents = list(documents)
        bridge_spans = self._dedupe_tokens(
            str(contract.get("bridge_span") or "")
            for trace in bridge_documents
            for contract in self._grounded_bridge_contracts(trace)
        )
        if not bridge_spans:
            bridge_spans = self._dedupe_tokens(
                str(contract.get("answer_span") or contract.get("object") or "")
                for trace in bridge_documents
                for contract in trace.direct_contracts
                if isinstance(contract, dict)
            )
        guard = round_trace.filter_metadata.get("query_guard")
        guard_reason = (
            normalize_text(str(guard.get("reason") or ""))
            if isinstance(guard, dict)
            else ""
        )
        active_goal = getattr(relation_plan, "active_goal", None)
        unresolved_goal = ""
        if active_goal is not None:
            unresolved_goal = normalize_text(
                " ".join(
                    [
                        str(getattr(active_goal, "subject", "") or ""),
                        str(getattr(active_goal, "relation", "") or ""),
                        str(getattr(active_goal, "target", "") or ""),
                    ]
                )
            )
        round_trace.next_hop_decision = NextHopDecisionTrace(
            required=bool(required),
            decision_reason=reason,
            unresolved_goal=unresolved_goal,
            bridge_spans=bridge_spans,
            bridge_document_ids=[
                trace.document_id for trace in bridge_documents
            ],
            generated_query=normalize_text(generated_query),
            query_guard_result=guard_reason,
        )

    @staticmethod
    def _metadata_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _record_fields(self, document: dict[str, Any]) -> dict[str, Any]:
        record_type = normalize_text(document.get("record_type", ""))
        if record_type in {"", "passage"}:
            return {}
        return {
            "title": normalize_text(document.get("title", "")),
            "authors": list(document.get("authors") or []),
            "date": normalize_text(document.get("date", "")),
            "source": normalize_text(document.get("source", "")),
            "content_url": normalize_text(document.get("content_url", "")),
            "language": normalize_text(document.get("language", "")),
            "country": normalize_text(document.get("country", "")),
            "parent_url": normalize_text(document.get("parent_url", "")),
            "extra_fields": dict(document.get("extra_fields") or {}),
        }

    def _apply_label(
        self,
        trace: RetrievedDocumentTrace,
        result: RAGLabelResult,
        *,
        question: str,
        intent_plan: SearchIntentPlan | None,
    ) -> None:
        contract = self.label_contract_validator.validate(result)
        metadata = dict(result.metadata)
        trace.sequence_tag = str(metadata.get("sequence_tag", ""))
        trace.label = normalize_text(result.label)
        trace.useful_tokens = list(result.kept_tokens)
        trace.useful_spans = list(contract.useful_spans)
        trace.raw_labeler_spans = self._dedupe_tokens(
            list(contract.useful_spans) + list(result.kept_tokens)
        )
        trace.valid_for_next_hop = bool(contract.valid_for_next_hop)
        trace.valid_for_evidence = False
        trace.direct_contracts = []
        trace.bridge_contracts = []
        trace.rejected_contracts = []
        trace.label_status = contract.label_status
        trace.invalid_reasons = list(contract.invalid_reasons)
        self._apply_restricted_span_recovery(
            trace=trace,
            metadata=metadata,
            intent_plan=intent_plan,
            question=question,
        )
        utility = self.evidence_utility_gate.assess(
            question=question,
            document=trace,
            intent_plan=intent_plan,
        )
        utility_spans = self._dedupe_tokens(
            list(utility.answer_spans) + list(utility.bridge_spans)
        )
        trace.answer_support_spans = self._dedupe_tokens(utility.answer_spans)
        trace.bridge_spans = self._dedupe_tokens(utility.bridge_spans)
        trace.support_level = normalize_text(utility.support_level)
        if utility_spans:
            trace.useful_spans = utility_spans
            trace.useful_tokens = list(utility_spans)
        trace.grounded_labeler_spans = self._ground_spans(
            text=trace.text,
            spans=list(trace.useful_spans) + list(trace.raw_labeler_spans),
        )
        if trace.grounded_labeler_spans:
            trace.useful_spans = list(trace.grounded_labeler_spans)
            trace.useful_tokens = list(trace.grounded_labeler_spans)
        trace.valid_for_next_hop = self._can_sequence_tag_hop(trace.sequence_tag) and bool(
            utility.valid_for_next_hop
        )
        trace.label = "useful" if trace.useful_spans else "useless"
        trace.invalid_reasons = self._dedupe_tokens(
            list(trace.invalid_reasons) + list(utility.reasons)
        )
        if not trace.useful_spans:
            trace.valid_for_next_hop = False
        trace.continue_probability = float(
            metadata.get("continue_probability", 0.0) or 0.0
        )
        trace.terminate_probability = float(
            metadata.get("terminate_probability", 0.0) or 0.0
        )
        trace.labeler_diagnostics = self._labeler_diagnostics(
            metadata=metadata,
            trace=trace,
        )

    def _apply_labeler_bypass(
        self,
        *,
        round_trace: RetrievalRoundTrace,
        question: str,
        documents: list[dict[str, Any]],
        trace_indexes: list[int],
    ) -> None:
        """Prepare sentence-level span candidates without invoking Labeler."""

        result = self.passage_evidence_unit_builder.build(
            question=question,
            documents=documents,
            embedder=getattr(self.retriever, "embedder", None),
        )
        grouped: dict[int, list[str]] = {}
        for unit in result.units:
            if unit.document_index >= len(trace_indexes):
                continue
            trace_index = trace_indexes[unit.document_index]
            grouped.setdefault(trace_index, []).append(unit.text)

        for trace_index in trace_indexes:
            trace = round_trace.documents[trace_index]
            units = self._dedupe_tokens(grouped.get(trace_index, []))
            trace.sequence_tag = "<BYPASS>"
            trace.label = "candidate" if units else "useless"
            trace.useful_tokens = list(units)
            trace.useful_spans = list(units)
            trace.raw_labeler_spans = []
            trace.grounded_labeler_spans = list(units)
            trace.label_status = (
                "labeler_bypass_candidate_units"
                if units
                else "labeler_bypass_no_candidate_unit"
            )
            trace.valid_for_next_hop = False
            trace.valid_for_evidence = False
            trace.labeler_diagnostics = {
                "mode": "bypass",
                "labeler_called": False,
                "candidate_unit_count": len(units),
                "candidate_units": list(units),
            }
        round_trace.filter_metadata["labeler_bypass"] = result.to_dict()

    def _apply_span_role_classification(
        self,
        *,
        round_trace: RetrievalRoundTrace,
        question: str,
        intent_plan: SearchIntentPlan | None,
    ) -> None:
        candidates, candidate_map = self._span_role_candidates(round_trace)
        for trace in round_trace.documents:
            trace.answer_support_spans = []
            trace.bridge_spans = []
            trace.valid_for_next_hop = False
            trace.valid_for_evidence = False
            trace.direct_contracts = []
            trace.bridge_contracts = []
            trace.rejected_contracts = []
            trace.semantic_facts = []
        if not candidates:
            round_trace.filter_metadata = {
                **round_trace.filter_metadata,
                "span_role_classifier": {
                    "success": True,
                    "candidate_count": 0,
                    "empty_reason": "no_useful_spans",
                },
            }
            return

        grounding_result = self.span_grounder.expand_candidates(candidates)
        grounding_diagnostics = {
            **grounding_result.diagnostics,
            "grounded_spans": [
                item.to_dict()
                for item in grounding_result.grounded_spans[:50]
            ],
        }
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "candidate_span_grounder": grounding_diagnostics,
        }
        candidates = list(grounding_result.candidates)

        quality_result = self.span_quality_gate.filter_candidates(candidates)
        quality_diagnostics = {
            **quality_result.diagnostics,
            "dropped": list(quality_result.dropped),
        }
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "candidate_span_quality_gate": quality_diagnostics,
        }
        candidates = list(quality_result.candidates)
        reindexed_candidates: list[CandidateSpan] = []
        reindexed_map: dict[str, tuple[int, str]] = {}
        for index, candidate in enumerate(candidates, start=1):
            mapped = candidate_map.get(candidate.id)
            if mapped is None:
                continue
            span_id = str(index)
            reindexed_candidates.append(
                CandidateSpan(
                    id=span_id,
                    text=candidate.text,
                    local_context=candidate.local_context,
                    source_title=candidate.source_title,
                    source_id=candidate.source_id,
                    source_type=candidate.source_type,
                )
            )
            reindexed_map[span_id] = (mapped[0], candidate.text)
        candidates = reindexed_candidates
        candidate_map = reindexed_map
        if not candidates:
            self._mark_span_quality_rejected(
                round_trace,
                reason="no_quality_candidate_spans",
                diagnostics=quality_diagnostics,
            )
            round_trace.filter_metadata = {
                **round_trace.filter_metadata,
                "span_role_classifier": {
                    "success": True,
                    "candidate_count": 0,
                    "empty_reason": "no_quality_candidate_spans",
                },
            }
            return

        result = self.span_role_classifier.classify_batch(
            question=question,
            answer_requirement=self._answer_requirement(intent_plan),
            answer_target=self._answer_target(intent_plan),
            active_goal=self._relation_goal_text(intent_plan, active=True),
            next_goal=self._relation_goal_text(intent_plan, active=False),
            relation_goals=self._relation_goal_options(intent_plan),
            spans=candidates,
            keep_alive=0,
        )
        diagnostics = dict(result.diagnostics)
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "span_role_classifier": diagnostics,
        }
        if not result.results:
            for trace in round_trace.documents:
                self._retain_grounded_spans(
                    trace,
                    reason="span_role_classifier_failed",
                )
                trace.labeler_diagnostics = {
                    **trace.labeler_diagnostics,
                    "span_role_classifier": {
                        "success": False,
                        "error": diagnostics.get("error", ""),
                    },
                }
            return

        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        by_trace_index: dict[int, list[dict[str, Any]]] = {}
        extraction_units: list[SemanticSourceUnit] = []
        role_item_by_unit_id: dict[str, dict[str, Any]] = {}
        for role_result in result.results:
            mapped = candidate_map.get(role_result.id)
            if mapped is None:
                continue
            trace_index, span_text = mapped
            effective_role = role_result.role
            if (
                self.bypass_labeler
                and effective_role == "NOISE"
                and role_result.model_role == ANSWER_SUPPORT
            ):
                effective_role = BRIDGE
            role_item = {
                    "id": role_result.id,
                    "text": span_text,
                    "role": effective_role,
                    "model_role": role_result.model_role,
                    "role_repair": (
                        "unbound_answer_support_demoted_to_bridge"
                        if effective_role != role_result.role
                        else ""
                    ),
                    "goal_id": role_result.goal_id,
                    "semantic_facts": [],
                }
            by_trace_index.setdefault(trace_index, []).append(role_item)
            candidate = candidate_by_id.get(role_result.id)
            if candidate is None or effective_role == "NOISE":
                continue
            unit_id = f"SPAN-{role_result.id}"
            extraction_units.append(
                SemanticSourceUnit(
                    unit_id=unit_id,
                    text=candidate.local_context,
                    source_id=candidate.source_id or f"span:{role_result.id}",
                    source_type=candidate.source_type,
                    source_title=candidate.source_title,
                    candidate_span=candidate.text,
                    # SemanticFactExtractor treats requested_role as a hard
                    # override of the model's own role output, so forwarding the
                    # classifier's label made the extractor unable to correct it.
                    # The classifier emitted BRIDGE 337 times against
                    # ANSWER_SUPPORT 8 on level1_final_06, and that pinned 317 of
                    # 325 facts to BRIDGE — which is what starves direct
                    # contracts. Leave it blank so the extractor classifies.
                    requested_role=(
                        effective_role if self.anchor_span_role_in_extraction else ""
                    ),
                    goal_id=role_result.goal_id,
                    metadata={"answer_target": self._answer_target(intent_plan)},
                )
            )
            role_item_by_unit_id[unit_id] = role_item

        extraction_batches: list[dict[str, Any]] = []
        extracted_facts: list[EvidenceFact] = []
        batch_size = max(1, self.semantic_fact_extractor.max_units_per_call)
        for start in range(0, len(extraction_units), batch_size):
            batch = extraction_units[start : start + batch_size]
            is_last = start + batch_size >= len(extraction_units)
            extraction = self.semantic_fact_extractor.extract_batch(
                question=question,
                answer_requirement=self._answer_requirement(intent_plan),
                current_goal=self._relation_goal_text(intent_plan, active=True),
                units=batch,
                keep_alive=(0 if is_last else "2m"),
            )
            extraction_batches.append(extraction.to_dict())
            extracted_facts.extend(extraction.facts)
        for fact in extracted_facts:
            unit_id = normalize_text(fact.qualifiers.get("source_unit_id", ""))
            role_item = role_item_by_unit_id.get(unit_id)
            if role_item is not None:
                role_item["semantic_facts"].append(fact.to_dict())
        round_trace.filter_metadata["semantic_fact_extractor"] = {
            "model": self.semantic_fact_extractor.model_name,
            "input_count": len(extraction_units),
            "batch_count": len(extraction_batches),
            "fact_count": len(extracted_facts),
            "grounded_count": sum(
                fact.grounding_status == "grounded" for fact in extracted_facts
            ),
            "batches": [item.get("diagnostics", {}) for item in extraction_batches],
        }

        for trace_index, trace in enumerate(round_trace.documents):
            if trace.duplicate or trace_index in by_trace_index:
                continue
            if not trace.useful_spans and not trace.useful_tokens:
                continue
            self._retain_grounded_spans(
                trace,
                reason="no_classified_span_for_document",
            )
            trace.labeler_diagnostics = {
                **trace.labeler_diagnostics,
                "candidate_span_quality_gate": {
                    "success": True,
                    "kept_for_document": 0,
                },
            }

        for trace_index, role_items in by_trace_index.items():
            trace = round_trace.documents[trace_index]
            finalization = self.span_finalizer.finalize_batch(
                items=role_items,
                context=trace.text,
                source_title=trace.title,
            )
            finalized_items = [item.to_dict() for item in finalization.finalized]
            for index, finalized_item in enumerate(finalized_items):
                finalized_item["semantic_facts"] = list(
                    role_items[index].get("semantic_facts") or []
                )
                finalized_item["model_role"] = str(
                    role_items[index].get("model_role") or ""
                )
            answer_support = self._dedupe_tokens(
                item.finalized_text
                for item in finalization.finalized
                if item.accepted and item.role == ANSWER_SUPPORT
            )
            bridge = self._dedupe_tokens(
                item.finalized_text
                for item in finalization.finalized
                if item.accepted and item.role == BRIDGE
            )
            trace.span_roles = [
                {
                    **item,
                    "finalized_text": finalized_items[index].get("finalized_text", ""),
                    "accepted": finalized_items[index].get("accepted", False),
                    "finalize_reason": finalized_items[index].get("reason", ""),
                }
                for index, item in enumerate(role_items)
            ]
            contracts = self.evidence_contract_builder.build(
                question=question,
                answer_requirement=self._answer_requirement(intent_plan),
                answer_target=self._answer_target(intent_plan),
                relation_plan=(
                    intent_plan.relation_plan if intent_plan is not None else None
                ),
                document_id=trace.document_id,
                source_title=trace.title,
                url=trace.url,
                text=trace.text,
                span_assignments=finalized_items,
            )
            semantic_facts = [
                EvidenceFact.from_dict(fact)
                for item in role_items
                for fact in list(item.get("semantic_facts") or [])
                if isinstance(fact, dict)
            ]
            promoted_facts, promotion_diagnostics = self._promote_document_facts(
                trace=trace,
                question=question,
                intent_plan=intent_plan,
                role_items=role_items,
                finalized_items=finalized_items,
            )
            all_facts = self._dedupe_facts([*semantic_facts, *promoted_facts])
            fact_contracts = self.evidence_contract_builder.build_from_grounded_facts(
                question=question,
                answer_requirement=self._answer_requirement(intent_plan),
                answer_target=self._answer_target(intent_plan),
                relation_plan=(
                    intent_plan.relation_plan if intent_plan is not None else None
                ),
                document_id=trace.document_id,
                source_title=trace.title,
                url=trace.url,
                text=trace.text,
                facts=all_facts,
            )
            trace.direct_contracts = self._dedupe_contracts(
                [
                    *[item.to_dict() for item in contracts.direct],
                    *[item.to_dict() for item in fact_contracts.direct],
                ],
                span_key="answer_span",
            )
            trace.bridge_contracts = [item.to_dict() for item in contracts.bridge]
            trace.rejected_contracts = [
                *[item.to_dict() for item in contracts.unsupported],
                *[item.to_dict() for item in fact_contracts.unsupported],
            ]
            trace.semantic_facts = [fact.to_dict() for fact in all_facts]
            trace.answer_support_spans = [
                normalize_text(str(item.get("answer_span") or ""))
                for item in trace.direct_contracts
                if normalize_text(str(item.get("answer_span") or ""))
            ]
            trace.bridge_spans = [item.bridge_span for item in contracts.bridge]
            answer_support = list(trace.answer_support_spans)
            bridge = list(trace.bridge_spans)
            trace.classified_spans = self._dedupe_tokens([*answer_support, *bridge])
            trace.useful_spans = list(trace.classified_spans)
            trace.useful_tokens = list(trace.classified_spans)
            if answer_support:
                trace.support_level = "direct"
            elif bridge:
                trace.support_level = "bridge"
            else:
                trace.support_level = "unsupported"
            trace.valid_for_next_hop = self._can_document_hop(trace) and bool(bridge)
            trace.valid_for_evidence = bool(answer_support)
            if trace.useful_spans:
                trace.label = "useful"
                trace.label_status = "span_role_classified"
            else:
                self._retain_grounded_spans(
                    trace,
                    reason="classified_spans_not_accepted",
                )
            trace.labeler_diagnostics = {
                **trace.labeler_diagnostics,
                "span_role_classifier": {
                    "success": True,
                    "span_roles": list(role_items),
                    "finalized_spans": finalized_items,
                    "finalization": finalization.diagnostics,
                    "answer_support_count": len(answer_support),
                    "bridge_count": len(bridge),
                    "noise_count": sum(
                        1 for item in role_items if item.get("role") == "NOISE"
                    ),
                    "contracts": contracts.to_dict(),
                    "fact_promotion": {
                        "promoted_fact_count": len(promoted_facts),
                        "promoted_contract_count": len(fact_contracts.direct),
                        "diagnostics": promotion_diagnostics,
                        "fact_contract_rejections": [
                            item.to_dict() for item in fact_contracts.unsupported
                        ],
                    },
                },
            }

        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "evidence_contracts": {
                "direct_count": sum(
                    len(trace.direct_contracts) for trace in round_trace.documents
                ),
                "bridge_count": sum(
                    len(trace.bridge_contracts) for trace in round_trace.documents
                ),
                "rejected_count": sum(
                    len(trace.rejected_contracts) for trace in round_trace.documents
                ),
                "direct_document_ids": [
                    trace.document_id
                    for trace in round_trace.documents
                    if trace.direct_contracts
                ],
                "bridge_document_ids": [
                    trace.document_id
                    for trace in round_trace.documents
                    if trace.bridge_contracts
                ],
                "contracts_by_goal": self._contract_counts_by_goal(
                    round_trace.documents
                ),
                "orphan_direct_fact_count": sum(
                    self._orphan_direct_fact_count(trace)
                    for trace in round_trace.documents
                ),
            },
        }
        fact_store = TaskFactStore()
        fact_store.extend(
            EvidenceFact.from_dict(fact)
            for trace in round_trace.documents
            for fact in trace.semantic_facts
            if isinstance(fact, dict)
        )
        round_trace.filter_metadata["semantic_fact_store"] = fact_store.to_dict()

    def _promote_document_facts(
        self,
        *,
        trace: RetrievedDocumentTrace,
        question: str,
        intent_plan: SearchIntentPlan | None,
        role_items: list[dict[str, Any]],
        finalized_items: list[dict[str, Any]],
    ) -> tuple[list[EvidenceFact], list[dict[str, object]]]:
        promoted: list[EvidenceFact] = []
        diagnostics: list[dict[str, object]] = []
        for index, role_item in enumerate(role_items):
            model_role = normalize_text(
                str(role_item.get("model_role") or role_item.get("role") or "")
            ).upper()
            if model_role != ANSWER_SUPPORT:
                continue
            finalized = finalized_items[index] if index < len(finalized_items) else {}
            candidate_span = normalize_text(
                str(
                    finalized.get("finalized_text")
                    or role_item.get("text")
                    or ""
                )
            )
            semantic_facts = [
                EvidenceFact.from_dict(value)
                for value in list(role_item.get("semantic_facts") or [])
                if isinstance(value, dict)
            ]
            goal_id = normalize_text(str(role_item.get("goal_id") or ""))
            if not goal_id:
                goal_id = next(
                    (
                        fact.goal_id
                        for fact in semantic_facts
                        if normalize_text(fact.goal_id)
                    ),
                    "",
                )
            result = self.direct_evidence_promoter.promote(
                model_role=model_role,
                candidate_span=candidate_span,
                context=trace.text,
                question=question,
                answer_requirement=self._answer_requirement(intent_plan),
                answer_target=self._answer_target(intent_plan),
                source_id=trace.document_id,
                source_title=trace.title,
                document_id=trace.document_id,
                goal_id=goal_id,
                semantic_facts=semantic_facts,
            )
            promoted.extend(result.promoted_facts)
            diagnostics.extend(item.to_dict() for item in result.diagnostics)
        return self._dedupe_facts(promoted), diagnostics

    @staticmethod
    def _dedupe_facts(facts: Iterable[EvidenceFact]) -> list[EvidenceFact]:
        output: list[EvidenceFact] = []
        seen: set[tuple[str, ...]] = set()
        for fact in facts:
            key = (
                normalize_text(fact.fact_id).casefold(),
                normalize_text(fact.goal_id).casefold(),
                normalize_text(fact.subject).casefold(),
                normalize_text(fact.relation).casefold(),
                normalize_text(fact.object).casefold(),
            )
            if key in seen:
                continue
            output.append(fact)
            seen.add(key)
        return output

    @staticmethod
    def _dedupe_contracts(
        contracts: Iterable[dict[str, Any]],
        *,
        span_key: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for contract in contracts:
            goal_id = normalize_text(str(contract.get("goal_id") or ""))
            span = normalize_text(str(contract.get(span_key) or ""))
            key = (goal_id.casefold(), span.casefold())
            if not span or key in seen:
                continue
            output.append(dict(contract))
            seen.add(key)
        return output

    @staticmethod
    def _orphan_direct_fact_count(trace: RetrievedDocumentTrace) -> int:
        contract_fact_ids = {
            normalize_text(str(item.get("fact_id") or ""))
            for item in trace.direct_contracts
            if normalize_text(str(item.get("fact_id") or ""))
        }
        return sum(
            1
            for value in trace.semantic_facts
            if isinstance(value, dict)
            and normalize_text(str(value.get("grounding_status") or "")) == "grounded"
            and normalize_text(str(value.get("role") or "")).upper()
            == ANSWER_SUPPORT
            and normalize_text(
                str(dict(value.get("qualifiers") or {}).get("answer_binding") or "")
            )
            == "direct"
            and normalize_text(str(value.get("fact_id") or ""))
            not in contract_fact_ids
        )

    def _mark_span_quality_rejected(
        self,
        round_trace: RetrievalRoundTrace,
        *,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> None:
        for trace in round_trace.documents:
            if trace.duplicate:
                continue
            if not trace.useful_spans and not trace.useful_tokens:
                continue
            self._retain_grounded_spans(trace, reason=reason)
            trace.labeler_diagnostics = {
                **trace.labeler_diagnostics,
                "candidate_span_quality_gate": {
                    "success": True,
                    "reason": reason,
                    "kept_count": diagnostics.get("kept_count", 0),
                    "dropped_count": diagnostics.get("dropped_count", 0),
                },
            }

    def _apply_cross_context_fact_extraction(
        self,
        *,
        round_trace: RetrievalRoundTrace,
        question: str,
        intent_plan: SearchIntentPlan | None,
    ) -> None:
        """從檢索錨點的相鄰 corpus passages 抽取可追溯的跨單位事實。"""

        units = self._cross_context_source_units(intent_plan=intent_plan)
        anchor_ids = [
            trace.document_id
            for trace in round_trace.documents
            if (
                not trace.duplicate
                and (
                    trace.raw_labeler_spans
                    or trace.useful_spans
                    or trace.sequence_tag in {CONTINUE_TAG, FINISH_TAG}
                )
            )
        ]
        windows = self.cross_context_assembler.assemble(
            units,
            anchor_unit_ids=anchor_ids,
        )
        if not windows:
            round_trace.filter_metadata["cross_context_fact_extraction"] = {
                "success": True,
                "window_count": 0,
                "anchor_count": len(anchor_ids),
                "empty_reason": "no_eligible_cross_context_windows",
            }
            return

        result = self.cross_context_fact_extractor.extract_windows(
            question=question,
            answer_requirement=self._answer_requirement(intent_plan),
            answer_target=self._answer_target(intent_plan),
            current_goal=self._relation_goal_text(intent_plan, active=True),
            current_goal_id=self._relation_goal_id(intent_plan),
            windows=windows,
        )
        round_trace.cross_context_facts = [fact.to_dict() for fact in result.facts]
        round_trace.filter_metadata["cross_context_fact_extraction"] = {
            **dict(result.diagnostics),
            "anchor_count": len(anchor_ids),
            "window_ids": [window.window_id for window in windows],
            "rejected_items": list(result.rejected_items),
        }
        grounded_facts = [
            fact for fact in result.facts if fact.grounding_status == "grounded"
        ]
        if not grounded_facts:
            return

        windows_by_id = {window.window_id: window for window in windows}
        facts_by_window: dict[str, list[EvidenceFact]] = {}
        for fact in grounded_facts:
            window_id = normalize_text(
                str(fact.qualifiers.get("cross_context_window_id") or "")
            )
            if window_id in windows_by_id:
                facts_by_window.setdefault(window_id, []).append(fact)

        trace_by_id = {
            trace.document_id: trace
            for trace in round_trace.documents
            if not trace.duplicate
        }
        relation_plan = intent_plan.relation_plan if intent_plan is not None else None
        for window_id, facts in facts_by_window.items():
            window = windows_by_id[window_id]
            assignments = [
                {
                    "accepted": True,
                    "role": fact.role,
                    "goal_id": fact.goal_id,
                    "original_text": fact.object,
                    "finalized_text": fact.object,
                    "semantic_facts": [fact.to_dict()],
                }
                for fact in facts
                if fact.role in {ANSWER_SUPPORT, BRIDGE}
            ]
            if not assignments:
                continue
            first_unit = window.units[0]
            first_metadata = dict(first_unit.metadata or {})
            url = normalize_text(str(first_metadata.get("url") or window.source_id))
            synthetic_id = f"cross-context:{window_id}"
            contracts = self.evidence_contract_builder.build(
                question=question,
                answer_requirement=self._answer_requirement(intent_plan),
                answer_target=self._answer_target(intent_plan),
                relation_plan=relation_plan,
                document_id=synthetic_id,
                source_title=first_unit.source_title,
                url=url,
                text=window.text,
                span_assignments=assignments,
            )
            if not contracts.direct and not contracts.bridge:
                continue
            referenced_scores = [
                trace_by_id[ref.document_id].retrieval_score
                for fact in facts
                for ref in fact.evidence_refs
                if ref.document_id in trace_by_id
            ]
            direct_spans = [item.answer_span for item in contracts.direct]
            bridge_spans = [item.bridge_span for item in contracts.bridge]
            synthetic = RetrievedDocumentTrace(
                document_id=synthetic_id,
                title=normalize_text(first_unit.source_title),
                text=window.text,
                url=url,
                retrieval_score=max(referenced_scores, default=0.0),
                record_type="cross_context",
                content_scope="cross_context_window",
                sequence_tag="<CROSS_CONTEXT>",
                useful_tokens=self._dedupe_tokens([*direct_spans, *bridge_spans]),
                useful_spans=self._dedupe_tokens([*direct_spans, *bridge_spans]),
                classified_spans=self._dedupe_tokens([*direct_spans, *bridge_spans]),
                answer_support_spans=direct_spans,
                bridge_spans=bridge_spans,
                semantic_facts=[fact.to_dict() for fact in facts],
                support_level=("direct" if direct_spans else "bridge"),
                valid_for_next_hop=bool(bridge_spans),
                valid_for_evidence=bool(direct_spans),
                direct_contracts=[item.to_dict() for item in contracts.direct],
                bridge_contracts=[item.to_dict() for item in contracts.bridge],
                rejected_contracts=[item.to_dict() for item in contracts.unsupported],
                label="useful",
                label_status="cross_context_fact_contract",
                labeler_diagnostics={
                    "cross_context_window": {
                        "window_id": window.window_id,
                        "unit_ids": list(window.unit_ids),
                        "boundary_reason": window.boundary_reason,
                    }
                },
            )
            round_trace.documents.append(synthetic)

        round_trace.filter_metadata["cross_context_fact_extraction"].update(
            {
                "synthetic_document_count": sum(
                    trace.record_type == "cross_context"
                    for trace in round_trace.documents
                ),
                "direct_contract_count": sum(
                    len(trace.direct_contracts)
                    for trace in round_trace.documents
                    if trace.record_type == "cross_context"
                ),
                "bridge_contract_count": sum(
                    len(trace.bridge_contracts)
                    for trace in round_trace.documents
                    if trace.record_type == "cross_context"
                ),
            }
        )

    def _cross_context_source_units(
        self,
        *,
        intent_plan: SearchIntentPlan | None,
    ) -> list[SemanticSourceUnit]:
        units: list[SemanticSourceUnit] = []
        for order, document in enumerate(self.retriever.passage_map.values()):
            document_id = normalize_text(str(document.get("id") or ""))
            text = normalize_text(str(document.get("text") or ""))
            source_id = self._cross_context_source_id(document)
            if not document_id or not text or not source_id:
                continue
            extra_fields = dict(document.get("extra_fields") or {})
            units.append(
                SemanticSourceUnit(
                    unit_id=document_id,
                    text=text,
                    source_id=source_id,
                    source_type="web",
                    source_title=normalize_text(str(document.get("title") or "")),
                    goal_id=self._relation_goal_id(intent_plan),
                    metadata={
                        "order": order,
                        "document_id": document_id,
                        "url": normalize_text(str(document.get("url") or "")),
                        "record_id": normalize_text(str(document.get("record_id") or "")),
                        "record_type": normalize_text(str(document.get("record_type") or "passage")),
                        "section": normalize_text(str(extra_fields.get("section") or "")),
                        "page": extra_fields.get("page"),
                        "table_id": normalize_text(str(extra_fields.get("table_id") or "")),
                        "answer_target": self._answer_target(intent_plan),
                    },
                )
            )
        return units

    @staticmethod
    def _cross_context_source_id(document: dict[str, Any]) -> str:
        source = normalize_text(
            str(
                document.get("parent_url")
                or document.get("url")
                or document.get("content_url")
                or document.get("record_id")
                or ""
            )
        )
        return source.split("#", 1)[0].rstrip("/").casefold()

    def _relation_goal_id(self, intent_plan: SearchIntentPlan | None) -> str:
        if intent_plan is None or intent_plan.relation_plan.active_goal is None:
            return ""
        return normalize_text(intent_plan.relation_plan.active_goal.goal_id)

    def _retain_grounded_spans(
        self,
        trace: RetrievedDocumentTrace,
        *,
        reason: str,
    ) -> None:
        """Keep source-grounded Labeler output without granting next-hop authority."""

        grounded = self._ground_spans(
            text=trace.text,
            spans=list(trace.grounded_labeler_spans)
            + list(trace.useful_spans)
            + list(trace.raw_labeler_spans),
        )
        trace.answer_support_spans = []
        trace.bridge_spans = []
        trace.direct_contracts = []
        trace.bridge_contracts = []
        trace.rejected_contracts = []
        trace.classified_spans = []
        trace.span_roles = []
        trace.grounded_labeler_spans = list(grounded)
        trace.useful_spans = list(grounded)
        trace.useful_tokens = list(grounded)
        trace.support_level = "unclassified" if grounded else "unsupported"
        trace.valid_for_next_hop = False
        trace.valid_for_evidence = False
        trace.label = "useful" if grounded else "useless"
        trace.label_status = "grounded_unclassified" if grounded else "span_quality_rejected"
        trace.labeler_diagnostics = {
            **trace.labeler_diagnostics,
            "grounded_span_fallback": {
                "reason": reason,
                "kept_count": len(grounded),
                "next_hop_allowed": False,
            },
        }

    def _ground_spans(self, *, text: str, spans: Iterable[str]) -> list[str]:
        """Return normalized spans that can be located in the source passage."""

        source = normalize_text(text)
        source_folded = source.casefold()
        return self._dedupe_tokens(
            cleaned
            for span in spans
            if (cleaned := normalize_text(span))
            and cleaned.casefold() in source_folded
        )

    def _span_role_candidates(
        self,
        round_trace: RetrievalRoundTrace,
        *,
        max_total: int = 40,
        max_per_document: int = 6,
    ) -> tuple[list[CandidateSpan], dict[str, tuple[int, str]]]:
        """Collect the round's spans for role classification.

        `max_total` is the second of three stacked budgets on the same spans
        (PassageEvidenceUnitBuilder.max_units, this, then the classifier's
        per-call bound), so it has to track the first: while that one held
        rounds to 10 spans this never bound, and leaving it at 15 would simply
        re-truncate the wider budget and reproduce the loss it was raised to
        fix.
        """

        candidates: list[CandidateSpan] = []
        candidate_map: dict[str, tuple[int, str]] = {}
        seen: set[str] = set()
        ordered_indexes = sorted(
            range(len(round_trace.documents)),
            key=lambda index: round_trace.documents[index].retrieval_score,
            reverse=True,
        )
        for trace_index in ordered_indexes:
            trace = round_trace.documents[trace_index]
            if trace.duplicate:
                continue
            per_document = 0
            for span in list(trace.useful_spans or []):
                cleaned = normalize_text(span)
                key = cleaned.casefold()
                if not cleaned or key in seen:
                    continue
                span_id = str(len(candidates) + 1)
                candidates.append(
                    CandidateSpan(
                        id=span_id,
                        text=cleaned,
                        source_title=trace.title,
                        local_context=self._span_local_context(cleaned, trace.text),
                        source_id=trace.document_id,
                        source_type="web",
                    )
                )
                candidate_map[span_id] = (trace_index, cleaned)
                seen.add(key)
                per_document += 1
                if len(candidates) >= max_total or per_document >= max_per_document:
                    break
            if len(candidates) >= max_total:
                break
        return candidates, candidate_map

    def _span_local_context(self, span: str, text: str, *, max_chars: int = 260) -> str:
        cleaned_text = normalize_text(text)
        cleaned_span = normalize_text(span)
        if not cleaned_text:
            return ""
        index = cleaned_text.casefold().find(cleaned_span.casefold())
        if index < 0:
            return cleaned_text[:max_chars]
        start = max(0, index - max_chars // 2)
        end = min(len(cleaned_text), index + len(cleaned_span) + max_chars // 2)
        left_boundary = max(cleaned_text.rfind(".", 0, start), cleaned_text.rfind("?", 0, start), cleaned_text.rfind("!", 0, start))
        if left_boundary >= 0 and index - left_boundary < max_chars:
            start = left_boundary + 1
        right_boundaries = [
            position
            for position in (
                cleaned_text.find(".", end),
                cleaned_text.find("?", end),
                cleaned_text.find("!", end),
            )
            if position >= 0
        ]
        if right_boundaries:
            right_boundary = min(right_boundaries)
            if right_boundary - index < max_chars:
                end = right_boundary + 1
        return normalize_text(cleaned_text[start:end])[:max_chars]

    def _answer_requirement(self, intent_plan: SearchIntentPlan | None) -> str:
        role = normalize_text(
            str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
        )
        return "" if role.casefold() in {"unknown", "none", "null", "n/a"} else role

    def _answer_target(self, intent_plan: SearchIntentPlan | None) -> str:
        return normalize_text(
            str(getattr(intent_plan, "target", "") if intent_plan else "")
        )

    def _relation_goal_text(
        self,
        intent_plan: SearchIntentPlan | None,
        *,
        active: bool,
    ) -> str:
        if intent_plan is None:
            return ""
        plan = intent_plan.relation_plan
        goal = plan.active_goal
        if not active and goal is not None:
            active_index = next(
                (
                    index
                    for index, item in enumerate(plan.goals)
                    if item.goal_id == goal.goal_id
                ),
                -1,
            )
            goal = next(
                (
                    item
                    for item in plan.goals[active_index + 1 :]
                    if item.state == "pending"
                ),
                None,
            )
        if goal is None:
            return ""
        return normalize_text(
            " -> ".join(
                part
                for part in [goal.subject, goal.relation, goal.target]
                if normalize_text(part)
            )
        )

    def _relation_goal_options(
        self,
        intent_plan: SearchIntentPlan | None,
    ) -> list[dict[str, str]]:
        if intent_plan is None:
            return []
        return [
            {
                "goal_id": goal.goal_id,
                "state": goal.state,
                "subject": goal.subject,
                "relation": goal.relation,
                "target": goal.target,
            }
            for goal in intent_plan.relation_plan.goals
        ]

    def _contract_counts_by_goal(
        self,
        documents: Iterable[RetrievedDocumentTrace],
    ) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for document in documents:
            for contract_type, contracts in (
                ("direct", document.direct_contracts),
                ("bridge", document.bridge_contracts),
            ):
                for contract in contracts:
                    goal_id = normalize_text(str(contract.get("goal_id", "")))
                    if not goal_id:
                        continue
                    bucket = counts.setdefault(goal_id, {"direct": 0, "bridge": 0})
                    bucket[contract_type] += 1
        return counts

    def _apply_restricted_span_recovery(
        self,
        *,
        trace: RetrievedDocumentTrace,
        metadata: dict[str, object],
        intent_plan: SearchIntentPlan | None,
        question: str,
    ) -> None:
        sequence_tag = normalize_text(str(metadata.get("sequence_tag", "") or ""))
        if trace.useful_spans or sequence_tag not in {"<CONTINUE>", "<TERMINATE>", "<FINISH>"}:
            return
        selected_passage = normalize_text(str(metadata.get("selected_passage", "") or ""))
        if not selected_passage:
            return
        recovery = self.evidence_utility_gate.span_recovery.recover_restricted(
            question=question,
            title=trace.title,
            selected_passage=selected_passage,
            intent_plan=intent_plan,
            answer_role=getattr(intent_plan, "answer_role", "") if intent_plan else "",
            sequence_tag=sequence_tag,
        )
        recovered_spans = (
            list(recovery.bridge_spans)
            if sequence_tag == "<CONTINUE>"
            else list(recovery.answer_spans)
        )
        metadata["span_recovery_triggered"] = True
        metadata["span_recovery_mode"] = "restricted"
        metadata["recovered_span_count"] = len(recovered_spans)
        metadata["recovered_spans"] = list(recovered_spans)
        trace.invalid_reasons = self._dedupe_tokens(
            list(trace.invalid_reasons) + list(recovery.reasons)
        )
        if not recovered_spans:
            return
        trace.useful_spans = self._dedupe_tokens(recovered_spans)
        trace.useful_tokens = list(trace.useful_spans)
        trace.label_status = (
            "continue_with_recovered_span"
            if sequence_tag == "<CONTINUE>"
            else "terminal_with_recovered_span"
        )
        trace.valid_for_next_hop = self._can_sequence_tag_hop(sequence_tag)

    def _can_sequence_tag_hop(self, sequence_tag: str) -> bool:
        return normalize_text(sequence_tag) in {CONTINUE_TAG, FINISH_TAG}

    def _can_document_hop(self, trace: RetrievedDocumentTrace) -> bool:
        if self.bypass_labeler:
            return True
        return self._can_sequence_tag_hop(trace.sequence_tag)

    def _labeler_diagnostics(
        self,
        *,
        metadata: dict[str, object],
        trace: RetrievedDocumentTrace,
    ) -> dict[str, object]:
        return {
            "input_mode": metadata.get("input_mode", ""),
            "source_title": metadata.get("source_title", ""),
            "record_type": metadata.get("record_type", trace.record_type),
            "record_id": metadata.get("record_id", trace.record_id),
            "record_fields": metadata.get("record_fields", trace.record_fields),
            "labeler_input_text": metadata.get("labeler_input_text", ""),
            "labeler_input_char_count": metadata.get("labeler_input_char_count", 0),
            "sequence_tag": trace.sequence_tag,
            "continue_probability": round(trace.continue_probability, 6),
            "terminate_probability": round(trace.terminate_probability, 6),
            "finish_probability": metadata.get("finish_probability", 0.0),
            "useful_token_count": len(trace.useful_tokens),
            "useful_span_count": len(trace.useful_spans),
            "valid_for_next_hop": trace.valid_for_next_hop,
            "selected_passage": metadata.get("selected_passage", ""),
            "selected_sentence_count": metadata.get("selected_sentence_count", 0),
            "original_char_count": metadata.get("original_char_count", 0),
            "selected_char_count": metadata.get("selected_char_count", 0),
            "sentence_selection_used": bool(metadata.get("sentence_selection_used", False)),
            "sentence_selection_truncated": bool(metadata.get("sentence_selection_truncated", False)),
            "span_recovery_triggered": bool(metadata.get("span_recovery_triggered", False)),
            "span_recovery_mode": metadata.get("span_recovery_mode", ""),
            "recovered_span_count": metadata.get("recovered_span_count", 0),
            "final_label_status": trace.label_status,
        }

    def _duplicate_reason(
        self,
        *,
        document: dict[str, Any],
        seen_document_ids: set[str],
        seen_chunk_keys: set[str],
    ) -> str:
        document_id = normalize_text(document.get("id", "")).casefold()
        chunk_key = self._chunk_key(self._document_text(document))
        if document_id and document_id in seen_document_ids:
            return "duplicate_document"
        if chunk_key and chunk_key in seen_chunk_keys:
            return "duplicate_chunk"
        return ""

    def _remember_document(
        self,
        *,
        document: dict[str, Any],
        seen_document_ids: set[str],
        seen_chunk_keys: set[str],
    ) -> None:
        document_id = normalize_text(document.get("id", "")).casefold()
        chunk_key = self._chunk_key(self._document_text(document))
        if document_id:
            seen_document_ids.add(document_id)
        if chunk_key:
            seen_chunk_keys.add(chunk_key)

    def _document_text(self, document: dict[str, Any]) -> str:
        return normalize_text(
            " ".join(
                part
                for part in (
                    document.get("title", ""),
                    document.get("text", ""),
                )
                if part
            )
        )

    def _query_key(self, query: str) -> str:
        return re.sub(
            r"[^\w]+",
            " ",
            normalize_text(query).casefold(),
            flags=re.UNICODE,
        ).strip()

    def _is_duplicate_query(
        self,
        query: str,
        seen_query_keys: set[str],
        *,
        lexical_threshold: float = 0.88,
    ) -> bool:
        key = self._query_key(query)
        if not key or key in seen_query_keys:
            return True
        query_terms = set(self._keywords(query))
        if len(query_terms) < 2:
            return True
        for seen in seen_query_keys:
            seen_terms = set(self._keywords(seen))
            if not seen_terms:
                continue
            overlap = len(query_terms & seen_terms) / max(1, min(len(query_terms), len(seen_terms)))
            if overlap >= lexical_threshold:
                return True
        return False

    def _chunk_key(self, text: str) -> str:
        normalized = normalize_text(text).casefold()
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _dedupe_tokens(self, tokens: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            cleaned = normalize_text(token)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    def _keywords(self, text: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "answer",
            "question",
            "source",
        }
        result: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]{2,}", normalize_text(text)):
            key = token.casefold().strip("'_.-")
            if not key or key in stopwords or key in seen:
                continue
            seen.add(key)
            result.append(token)
        return result


class WebRetrievalControl:
    """
    從 query generation、網頁搜尋與 corpus 建立開始執行完整 retrieval。

    Args:
        - query_generator: 從原始問題產生網頁搜尋 query 的元件。
        - search_tool: 提供 run(parameters) 的網頁搜尋工具。
        - corpus_builder: 清洗全文、分割 passages 與去重的 corpus builder。
        - labeler: EfficientRAG Labeler。
        - next_hop_composer: 組合下一跳查詢的控制器。
        - model_type: Passage/query embedding 模型類型。
        - max_queries: 最多使用的初始搜尋 query 數量。
        - max_results_per_query: 每個 query 最多取得的網頁結果數。
        - max_pages_to_fetch: 最多抓取全文的頁面數。
        - max_corpus_records: 動態 corpus 最多保留的 passages。
        - max_iter: Labeler/Filter 最大迭代輪數。
        - top_k: 每輪 FAISS retrieval 的 passage 數量。

    Returns:
        - WebRetrievalControl: 動態建立 corpus 並執行 EfficientRAG 的控制器。
    """

    def __init__(
        self,
        *,
        query_generator: QueryGenerator | None = None,
        search_tool: Any | None = None,
        source_acquisition_router: SourceAcquisitionRouter | None = None,
        corpus_builder: WebCorpusBuilder | None = None,
        source_filter: SourceFilter | None = None,
        page_content_fetcher: PageContentFetcher | None = None,
        labeler: EfficientRAGLabelerAdapter | None = None,
        next_hop_composer: NextHopQueryComposer | None = None,
        next_hop_evidence_selector: NextHopEvidenceSelector | None = None,
        model_type: str = "",
        max_queries: int = 3,
        max_results_per_query: int = 8,
        max_pages_to_fetch: int = 24,
        max_chunks_per_url: int = 20,
        max_corpus_records: int = 300,
        max_urls_per_domain: int = 3,
        min_filtered_sources: int = 5,
        max_iter: int = 4,
        top_k: int = 16,
        min_retrieval_score: float = 0.75,
        relative_score_margin: float = 0.08,
        embedding_batch_size: int = 8,
        # A structured record is a stub: a title and a content_url, no page body.
        # This bounds how many of those links get read, so it decides whether the
        # page holding the answer is ever opened. At 3 it opened 42 of the 567
        # candidates level1_final_06 produced (7.4%), and six tasks opened none.
        # Replaying the run's candidates in rank order and fetching them, the
        # link carrying the answer sat at rank 1, 2, 5, 8 and 19 across the five
        # tasks where one existed: 3 reaches two of them, 10 reaches four, and 20
        # reaches the fifth. 10 is where the marginal cost turns -- going to 20
        # doubles the fetches and the records added to the corpus to gain one
        # task, at a rank deep enough that ordering is mostly noise. Fetch
        # latency is not the binding cost (about 1s each, so 7s per task).
        max_collection_links_to_fetch: int = 10,
        collection_link_fetch_tokens: int = 5000,
        bypass_labeler: bool = False,
    ) -> None:
        self.query_generator = query_generator or QueryGenerator()
        if search_tool is None:
            from ..search_tool import SearchTool

            search_tool = SearchTool()
        self.search_tool = search_tool
        self.source_acquisition_router = (
            source_acquisition_router
            or SourceAcquisitionRouter(search_tool=self.search_tool)
        )
        self.corpus_builder = corpus_builder or WebCorpusBuilder(
            chunker=DocumentChunker(
                max_chars=600,
                overlap_chars=80,
                min_chars=80,
            )
        )
        self.source_filter = source_filter or SourceFilter(
            max_urls_per_domain=max_urls_per_domain,
            min_sources=min_filtered_sources,
        )
        self.page_content_fetcher = (
            page_content_fetcher or PageContentFetcher(max_workers=4)
        )
        self.bypass_labeler = bool(bypass_labeler)
        self.labeler = (
            labeler
            if labeler is not None
            else (None if self.bypass_labeler else EfficientRAGLabelerAdapter())
        )
        self.next_hop_composer = next_hop_composer or NextHopQueryComposer()
        self.next_hop_evidence_selector = (
            next_hop_evidence_selector or NextHopEvidenceSelector()
        )
        # Retrieval embedding model. bge-m3 is the default because on saved
        # runs it placed the answer passage inside the 8-reference budget on
        # 8/10 tasks against 6/10 for multilingual-e5-base, and bounded the
        # worst rank at 18 instead of 26 — the deep-tail cases are exactly
        # where the previous model could not separate anything. Average rank
        # quality is a wash between the two, so this is overridable.
        self.model_type = (
            model_type
            or os.getenv("SEARCH_EMBED_MODEL", "").strip()
            or "bge-m3"
        )
        self.max_queries = max(1, max_queries)
        self.max_results_per_query = max(1, max_results_per_query)
        self.max_pages_to_fetch = max(0, max_pages_to_fetch)
        self.max_chunks_per_url = max(1, max_chunks_per_url)
        self.max_corpus_records = max(1, max_corpus_records)
        self.max_iter = max(1, max_iter)
        self.top_k = max(1, top_k)
        self.min_retrieval_score = max(
            0.0,
            min(1.0, min_retrieval_score),
        )
        self.relative_score_margin = max(
            0.0,
            min(1.0, relative_score_margin),
        )
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.max_collection_links_to_fetch = max(0, max_collection_links_to_fetch)
        self.collection_link_fetch_tokens = max(500, collection_link_fetch_tokens)

    def run(
        self,
        question: str,
        *,
        output_dir: str | Path,
    ) -> WebRetrievalResult:
        """
        建立問題專屬 corpus、FAISS index，並執行 Labeler/Filter 迭代。

        Args:
            - question: 原始問題。
            - output_dir: 儲存 corpus、embeddings 與 FAISS index 的目錄。

        Returns:
            - WebRetrievalResult: Query、網頁、corpus 與 retrieval trace。
        """
        text = normalize_text(question)
        root = Path(output_dir)
        corpus_path = root / "corpus.jsonl"
        embedding_path = root / "embeddings"
        root.mkdir(parents=True, exist_ok=True)

        if not text:
            return WebRetrievalResult(
                question="",
                generated_queries=[],
                salient_spans=[],
                web_searches=[],
                corpus_path=str(corpus_path),
                embedding_path=str(embedding_path),
                corpus_record_count=0,
                retrieval=None,
                diagnostics={"stop_reason": "empty_question"},
            )

        plan = self.query_generator.plan(
            question=text,
            max_queries=self.max_queries,
        )
        search_intent_plan = SearchIntentPlan.from_dict(
            plan.get("search_intent_plan") if isinstance(plan, dict) else None
        )
        query_requests = self._query_requests_from_plan(plan, fallback_question=text)
        query_requests = query_requests[: self.max_queries]
        generated_queries = [request.query for request in query_requests]
        if not search_intent_plan.search_needed:
            return WebRetrievalResult(
                question=text,
                generated_queries=[],
                salient_spans=list(plan.get("salient_spans") or []),
                web_searches=[],
                corpus_path=str(corpus_path),
                embedding_path=str(embedding_path),
                corpus_record_count=0,
                retrieval=None,
                diagnostics={
                    "query_plan": plan,
                    "stop_reason": "search_intent_no_search",
                    "search_intent_plan": search_intent_plan.to_dict(),
                },
            )
        if not generated_queries:
            generated_queries = [text]
            query_requests = [SearchQueryRequest.fallback(text)]

        sources, web_searches = self._acquire_sources(
            query_requests,
            question=text,
        )
        query_text_by_id = {
            f"Q{index}": query
            for index, query in enumerate(generated_queries, start=1)
        }
        filtered_sources = self.source_filter.filter_sources(
            sources,
            question=text,
            query_text_by_id=query_text_by_id,
            fetch_limit=self.max_pages_to_fetch,
        )
        blocked_sources = [source for source in sources if source.blocked]
        source_diagnostics = self._source_diagnostics(
            sources=sources,
            filtered_sources=filtered_sources,
            blocked_sources=blocked_sources,
        )
        fetch_target_sources = list(filtered_sources)
        fetched_pages = self.page_content_fetcher.fetch_sources(
            filtered_sources,
            max_pages=self.max_pages_to_fetch,
            max_tokens_per_source=5000,
        )
        filtered_sources = self.source_filter.apply_post_fetch_safety(
            filtered_sources,
            question=text,
        )
        blocked_sources = [source for source in sources if source.blocked]
        source_diagnostics = self._source_diagnostics(
            sources=sources,
            filtered_sources=filtered_sources,
            blocked_sources=blocked_sources,
        )
        fetch_diagnostics = self._fetch_diagnostics(fetch_target_sources)
        records = self.corpus_builder.build_records(
            filtered_sources,
            fetch_missing=False,
            max_chunks_per_url=self.max_chunks_per_url,
            max_records=self.max_corpus_records,
            question=text,
        )
        self.corpus_builder.exporter.export(records, corpus_path)

        diagnostics: dict[str, object] = {
            "query_plan": plan,
            "source_acquisition": [
                {
                    "query": trace.query,
                    "requested_source_kind": trace.requested_source_kind,
                    "requested_access_mode": trace.requested_access_mode,
                    "source_hint": trace.source_hint,
                    "required_content": trace.required_content,
                    "actual_acquirer": trace.actual_acquirer,
                    "fallback_used": trace.fallback_used,
                    "result_count": trace.result_count,
                    "source_ids": list(trace.source_ids),
                    "notices": list(trace.notices),
                }
                for trace in web_searches
            ],
            "source_count": len(sources),
            "filtered_source_count": len(filtered_sources),
            "blocked_source_count": len(blocked_sources),
            "fetched_page_count": fetched_pages,
            "source_filter": source_diagnostics,
            "full_page_fetch": fetch_diagnostics,
            "max_urls_per_domain": self.source_filter.max_urls_per_domain,
            "min_filtered_sources": self.source_filter.min_sources,
            "max_chunks_per_url": self.max_chunks_per_url,
            "max_corpus_records": self.max_corpus_records,
            "structured_record_count": sum(
                1 for record in records if record.record_type != "passage"
            ),
            "corpus_pipeline": (
                "web_search->seer_source_filter->seer_full_page_fetch"
                "->collection_record_or_passage->e5->faiss"
            ),
            "embedding_model": self.model_type,
            "labeler_mode": (
                "bypass_span_role" if self.bypass_labeler else "efficientrag"
            ),
        }
        if not records:
            diagnostics["stop_reason"] = "empty_corpus"
            return WebRetrievalResult(
                question=text,
                generated_queries=generated_queries,
                salient_spans=list(plan.get("salient_spans") or []),
                web_searches=web_searches,
                corpus_path=str(corpus_path),
                embedding_path=str(embedding_path),
                corpus_record_count=0,
                retrieval=None,
                diagnostics=diagnostics,
                blocked_sources=blocked_sources,
            )

        self._reset_embedding_dir(embedding_path)
        self._embed_records(
            records=[record.to_dict() for record in records],
            embedding_path=embedding_path,
        )
        retriever = Retriever(
            passage_path=str(corpus_path),
            passage_embedding_path=str(embedding_path),
            index_path_dir=str(embedding_path),
            model_type=self.model_type,
            save_or_load_index=True,
            batch_size=self.embedding_batch_size,
        )
        corpus_session = TaskCorpusSession(
            corpus_path=corpus_path,
            retriever=retriever,
            exporter=self.corpus_builder.exporter,
        )
        # Link enrichment only ever adds records; the corpus already stands
        # without it. It is wrapped because it did not degrade when it broke --
        # `not vectors` on a numpy array inside `_select_linked_chunks` raised,
        # unwound the whole of `run`, and left the task with no search evidence
        # at all. That emptied Evidence Prepare on 25 of 53 tasks in
        # level1_final_18 while the run still reported a normal score. An
        # optional stage failing should cost its own records and nothing else.
        link_enrichment_error = ""
        try:
            linked_records = self._enrich_collection_links(
                question=text,
                retriever=retriever,
                corpus_session=corpus_session,
            )
        except Exception as exc:
            linked_records = []
            link_enrichment_error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-2000:]
        diagnostics["collection_link_enrichment"] = {
            "enabled": self.max_collection_links_to_fetch > 0,
            "added_record_count": len(linked_records),
            "record_ids": [record.id for record in linked_records],
            "error_traceback": link_enrichment_error,
        }

        def load_external_sources(requests: list[SearchQueryRequest]) -> int:
            remaining = self.max_corpus_records - len(retriever.passage_map)
            if remaining <= 0 or not requests:
                return 0
            acquired, traces = self._acquire_sources(requests, question=text)
            source_offset = len(sources)
            source_id_map: dict[str, str] = {}
            for index, source in enumerate(acquired, start=1):
                previous_id = source.source_id
                source.source_id = f"H{source_offset + index}"
                source_id_map[previous_id] = source.source_id
            for trace in traces:
                trace.source_ids = [
                    source_id_map.get(source_id, source_id)
                    for source_id in trace.source_ids
                ]
            web_searches.extend(traces)
            sources.extend(acquired)
            query_map = {
                f"Q{index}": request.query
                for index, request in enumerate(requests, start=1)
            }
            filtered = self.source_filter.filter_sources(
                acquired,
                question=text,
                query_text_by_id=query_map,
                fetch_limit=self.max_pages_to_fetch,
            )
            full_document_recovery = any(
                request.source_requirement.access_mode in {"direct_fetch", "browser"}
                for request in requests
            )
            self.page_content_fetcher.fetch_sources(
                filtered,
                max_pages=self.max_pages_to_fetch,
                max_tokens_per_source=(12000 if full_document_recovery else 5000),
            )
            filtered = self.source_filter.apply_post_fetch_safety(
                filtered,
                question=text,
            )
            blocked_sources.extend(
                source
                for source in acquired
                if source.blocked and source not in blocked_sources
            )
            new_records = self.corpus_builder.build_records(
                filtered,
                fetch_missing=False,
                max_chunks_per_url=(
                    max(self.max_chunks_per_url, 50)
                    if full_document_recovery
                    else self.max_chunks_per_url
                ),
                max_records=remaining,
                question=text,
            )
            record_metadata_by_url: dict[str, dict[str, Any]] = {}
            for document in retriever.passage_map.values():
                content_url = normalize_text(document.get("content_url", ""))
                if not content_url:
                    continue
                record_metadata_by_url.setdefault(
                    content_url.casefold().rstrip("/"),
                    document,
                )
            linked_new_records = []
            for record in new_records:
                metadata = record_metadata_by_url.get(
                    normalize_text(record.url).casefold().rstrip("/")
                )
                if metadata is None:
                    linked_new_records.append(record)
                    continue
                linked_new_records.append(
                    replace(
                        record,
                        record_type=(
                            normalize_text(metadata.get("record_type", ""))
                            or record.record_type
                        ),
                        record_id=(
                            normalize_text(metadata.get("record_id", ""))
                            or record.record_id
                        ),
                        content_url=(
                            normalize_text(metadata.get("content_url", ""))
                            or record.content_url
                        ),
                        parent_url=(
                            normalize_text(metadata.get("parent_url", ""))
                            or record.parent_url
                        ),
                    )
                )
            return len(corpus_session.add_records(linked_new_records))

        retrieval = IterativeRetrievalControl(
            retriever=retriever,
            labeler=self.labeler,
            bypass_labeler=self.bypass_labeler,
            next_hop_composer=self.next_hop_composer,
            next_hop_evidence_selector=self.next_hop_evidence_selector,
            max_iter=self.max_iter,
            top_k=min(self.top_k, len(retriever.passage_map)),
            min_retrieval_score=self.min_retrieval_score,
            relative_score_margin=self.relative_score_margin,
            external_source_loader=load_external_sources,
        ).run(text, intent_plan=search_intent_plan)

        diagnostics["initial_retrieval_query"] = text
        diagnostics["stop_reason"] = retrieval.stop_reason
        diagnostics["coverage_summary"] = self._coverage_summary(retrieval)
        diagnostics["search_intent_plan"] = search_intent_plan.to_dict()
        diagnostics["relation_plan"] = retrieval.relation_plan
        diagnostics["dynamic_corpus_record_count"] = len(retriever.passage_map)
        diagnostics["source_count"] = len(sources)
        diagnostics["blocked_source_count"] = len(blocked_sources)
        diagnostics["source_acquisition"] = [
            {
                "query": trace.query,
                "requested_source_kind": trace.requested_source_kind,
                "requested_access_mode": trace.requested_access_mode,
                "source_hint": trace.source_hint,
                "required_content": trace.required_content,
                "actual_acquirer": trace.actual_acquirer,
                "fallback_used": trace.fallback_used,
                "result_count": trace.result_count,
                "source_ids": list(trace.source_ids),
                "notices": list(trace.notices),
            }
            for trace in web_searches
        ]
        return WebRetrievalResult(
            question=text,
            generated_queries=generated_queries,
            salient_spans=list(plan.get("salient_spans") or []),
            web_searches=web_searches,
            corpus_path=str(corpus_path),
            embedding_path=str(embedding_path),
            corpus_record_count=len(retriever.passage_map),
            retrieval=retrieval,
            diagnostics=diagnostics,
            blocked_sources=blocked_sources,
        )

    def _enrich_collection_links(
        self,
        *,
        question: str,
        retriever: Retriever,
        corpus_session: TaskCorpusSession,
    ) -> list[Any]:
        """只抓取與問題最相關的結構化記錄詳細頁。"""
        if self.max_collection_links_to_fetch <= 0:
            return []
        structured_count = sum(
            1
            for document in retriever.passage_map.values()
            if normalize_text(document.get("record_type", "")) not in {"", "passage"}
            and normalize_text(document.get("content_url", ""))
        )
        if structured_count <= 0:
            return []
        try:
            ranked = retriever.search(
                question,
                top_k=min(
                    len(retriever.passage_map),
                    max(self.max_collection_links_to_fetch * 3, 6),
                ),
            )
        except Exception:
            return []
        documents = ranked[0] if ranked else []
        selected: list[dict[str, Any]] = []
        seen_records: set[str] = set()
        seen_urls: set[str] = set()
        for document in documents:
            record_type = normalize_text(document.get("record_type", ""))
            record_id = normalize_text(document.get("record_id", ""))
            content_url = normalize_text(document.get("content_url", ""))
            parent_url = normalize_text(document.get("parent_url", ""))
            if record_type in {"", "passage"} or not content_url:
                continue
            if content_url == parent_url:
                continue
            record_key = record_id or content_url.casefold()
            url_key = content_url.casefold()
            if record_key in seen_records or url_key in seen_urls:
                continue
            selected.append(document)
            seen_records.add(record_key)
            seen_urls.add(url_key)
            if len(selected) >= self.max_collection_links_to_fetch:
                break
        added: list[Any] = []
        for document in selected:
            enriched = self.corpus_builder.build_enriched_records(
                document,
                max_tokens=self.collection_link_fetch_tokens,
                question=question,
                embedder=getattr(retriever, "embedder", None),
            )
            added.extend(corpus_session.add_records(enriched))
        return added

    def _coverage_summary(
        self,
        retrieval: IterativeRetrievalResult,
    ) -> dict[str, Any]:
        rounds = list(retrieval.rounds or [])
        coverage_items = [
            dict(round_item.coverage)
            for round_item in rounds
            if isinstance(round_item.coverage, dict) and round_item.coverage
        ]
        final = coverage_items[-1] if coverage_items else {}
        intent_states = [
            (item.get("intent_state") or {}).get("state", "")
            for item in coverage_items
            if isinstance(item.get("intent_state"), dict)
        ]
        final_intent_state = (
            final.get("intent_state", {})
            if isinstance(final.get("intent_state"), dict)
            else {}
        )
        final_goal_completion = {}
        recovery_actions: list[dict[str, Any]] = []
        for round_item in rounds:
            metadata = dict(round_item.filter_metadata or {})
            if isinstance(metadata.get("goal_completion"), dict):
                final_goal_completion = dict(metadata["goal_completion"])
            for action in list(metadata.get("goal_recovery") or []):
                if isinstance(action, dict):
                    recovery_actions.append(dict(action))
        next_hop_triggers = [
            item.get("trigger_reason", "")
            for item in coverage_items
            if item.get("trigger_reason") and item.get("trigger_reason") != "coverage_sufficient"
        ]
        return {
            "round_count": len(coverage_items),
            "final_sufficient": bool(final.get("sufficient", False)),
            "final_score": final.get("coverage_score", 0.0),
            "missing_constraints": list(final.get("missing_constraints") or []),
            "answer_type": final.get("answer_type", "unknown"),
            "answer_type_covered": bool(final.get("answer_type_covered", False)),
            "bridge_terms": list(final.get("bridge_terms") or []),
            "next_hop_triggered_by": next_hop_triggers[-1] if next_hop_triggers else "",
            "intent_states": intent_states,
            "final_intent_state": final_intent_state,
            "goal_completion": final_goal_completion,
            "recovery_actions": recovery_actions,
        }

    def _search_queries(
        self,
        queries: list[str],
    ) -> tuple[list[SearchSourceCandidate], list[WebSearchTrace]]:
        requests = [SearchQueryRequest.fallback(query) for query in queries]
        return self._acquire_sources(requests, question="")

    def _acquire_sources(
        self,
        requests: list[SearchQueryRequest],
        *,
        question: str,
    ) -> tuple[list[SearchSourceCandidate], list[WebSearchTrace]]:
        sources, acquisition_traces = self.source_acquisition_router.acquire_many(
            requests,
            question=question,
            max_results=self.max_results_per_query,
        )
        traces = [self._web_search_trace(item) for item in acquisition_traces]
        return sources, traces

    def _web_search_trace(self, trace: SourceAcquisitionTrace) -> WebSearchTrace:
        return WebSearchTrace(
            query=trace.query,
            backend=trace.actual_acquirer,
            result_count=trace.result_count,
            source_ids=list(trace.source_ids),
            notices=list(trace.notices),
            requested_source_kind=trace.requested_source_kind,
            requested_access_mode=trace.requested_access_mode,
            source_hint=trace.source_hint,
            required_content=trace.required_content,
            actual_acquirer=trace.actual_acquirer,
            fallback_used=trace.fallback_used,
        )

    def _query_requests_from_plan(
        self,
        plan: dict[str, Any] | Any,
        *,
        fallback_question: str,
    ) -> list[SearchQueryRequest]:
        if not isinstance(plan, dict):
            return [SearchQueryRequest.fallback(fallback_question)]
        requests: list[SearchQueryRequest] = []
        seen: set[str] = set()
        for item in list(plan.get("query_requests") or []):
            if not isinstance(item, dict):
                continue
            request = SearchQueryRequest.from_dict(item)
            if request is None:
                continue
            key = normalize_text(request.query).lower()
            if not key or key in seen:
                continue
            requests.append(request)
            seen.add(key)
        if requests:
            return requests
        for query in self._dedupe_queries(list(plan.get("queries") or [])):
            requests.append(SearchQueryRequest.fallback(query))
        return requests or [SearchQueryRequest.fallback(fallback_question)]

    def _source_diagnostics(
        self,
        *,
        sources: list[SearchSourceCandidate],
        filtered_sources: list[SearchSourceCandidate],
        blocked_sources: list[SearchSourceCandidate],
    ) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        rescued_count = 0
        for source in sources:
            if source.block_reason:
                reason_counts[source.block_reason] = reason_counts.get(source.block_reason, 0) + 1
            if any(str(reason).startswith("rescued_soft_block:") for reason in source.filter_reasons):
                rescued_count += 1
        return {
            "source_count": len(sources),
            "filtered_source_count": len(filtered_sources),
            "blocked_source_count": len(blocked_sources),
            "rescued_soft_block_count": rescued_count,
            "block_reason_counts": reason_counts,
            "fetch_candidate_count": sum(
                1 for source in filtered_sources if source.should_fetch_full_page
            ),
            "filtered_source_ids": [source.source_id for source in filtered_sources[:20]],
            "selection_mode": self.source_filter.last_fetch_selection.mode,
            # Minimal snapshot of every candidate so a fetch-selection change
            # can be replayed offline. Without this only fetched pages survive
            # in the trace, and the sources that lost a slot — exactly the ones
            # a ranking change is meant to rescue — are unrecoverable.
            "fetch_selection_candidates": [
                {
                    "source_id": source.source_id,
                    "url": source.url,
                    "domain": source.domain,
                    "title": source.title[:160],
                    "matched_query_ids": list(source.matched_query_ids),
                    "query_hit_count": source.query_hit_count,
                    "named_source_match": source.named_source_match,
                    "named_source_terms": list(source.named_source_terms),
                    "url_echo": source.url_echo,
                    "constraint_match_level": source.constraint_match_level,
                    "priority_tier": source.fetch_priority_tier,
                    "legacy_position": source.legacy_fetch_position,
                    "fetch_batch": source.fetch_batch,
                    "selection_reasons": list(source.fetch_priority_reasons),
                }
                for source in filtered_sources[:40]
            ],
            "blocked_source_details": [
                {
                    "source_id": source.source_id,
                    "query_id": source.query_id,
                    "title": source.title,
                    "url": source.url,
                    "domain": source.domain,
                    "block_reason": source.block_reason,
                    "filter_reasons": list(source.filter_reasons),
                }
                for source in blocked_sources[:20]
            ],
            "content_requirement_counts": {
                state: sum(
                    1 for source in sources if source.acquisition_state == state
                )
                for state in {
                    source.acquisition_state
                    for source in sources
                    if source.acquisition_state
                }
            },
        }

    def _fetch_diagnostics(
        self,
        sources: list[SearchSourceCandidate],
    ) -> dict[str, Any]:
        method_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        failure_count = 0
        low_quality_count = 0
        for source in sources:
            for reason in source.filter_reasons:
                if reason.startswith("fetch_method:"):
                    method = reason.split(":", 1)[1]
                    method_counts[method] = method_counts.get(method, 0) + 1
                elif reason.startswith("fetch_quality:"):
                    quality = reason.split(":", 1)[1]
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1
                elif reason == "full_page_fetch_failed":
                    failure_count += 1
                elif reason == "low_quality_full_page":
                    low_quality_count += 1
        return {
            "attempted_fetch_count": sum(
                1
                for source in sources
                if any(
                    reason.startswith("fetch_status:")
                    or reason == "full_page_fetch_failed"
                    for reason in source.filter_reasons
                )
            ),
            "fetched_page_count": sum(1 for source in sources if source.fetched),
            "fetch_failure_count": failure_count,
            "low_quality_full_page_count": low_quality_count,
            "method_counts": method_counts,
            "quality_counts": quality_counts,
            "requirement_met_count": sum(
                1 for source in sources if source.requirement_met
            ),
            "requirement_unmet": [
                {
                    "source_id": source.source_id,
                    "required_content": source.required_content,
                    "state": source.acquisition_state,
                    "missing_content": list(source.missing_content),
                }
                for source in sources
                if source.content_extracted and not source.requirement_met
            ],
        }

    def _embed_records(
        self,
        *,
        records: list[dict[str, Any]],
        embedding_path: Path,
    ) -> None:
        embedder = Embedder(
            self.model_type,
            batch_size=self.embedding_batch_size,
            chunk_size=max(1, self.max_corpus_records),
            text_normalize=True,
        )
        for index, (ids, embeddings) in embedder.embed_passages(records):
            output_file = embedding_path / f"passages_{index:02d}"
            with output_file.open("wb") as handle:
                pickle.dump((ids, embeddings), handle)

    def _reset_embedding_dir(self, path: Path) -> None:
        resolved = path.resolve()
        root = path.parent.resolve()
        if resolved.parent != root:
            raise ValueError(f"Unexpected embedding directory: {resolved}")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for query in queries:
            cleaned = normalize_text(query)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result


__all__ = [
    "IterativeRetrievalControl",
    "IterativeRetrievalResult",
    "NextHopDecisionTrace",
    "PageRetrievalTrace",
    "RetrievalRoundTrace",
    "RetrievedDocumentTrace",
    "WebRetrievalControl",
    "WebRetrievalResult",
    "WebSearchTrace",
]
