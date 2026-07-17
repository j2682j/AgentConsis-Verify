from __future__ import annotations

import hashlib
import pickle
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from utils.network_utils import normalize_text

from .config import EvidenceItem, SearchSourceCandidate
from .corpus import DocumentChunker, TaskCorpusSession, WebCorpusBuilder
from .embeddings import Embedder
from .evidence import (
    ANSWER_SUPPORT,
    BRIDGE,
    CandidateSpan,
    CandidateSpanGrounder,
    CandidateSpanQualityGate,
    EvidenceUtilityGate,
    EvidenceRoleContractBuilder,
    RoleAwareSpanFinalizer,
    SpanRoleClassifier,
)
from .next_hop_query.coverage_assessor import CoverageAssessor
from .next_hop_query.evidence_sufficiency_gate import EvidenceSufficiencyGate
from .next_hop_query.intent_state_tracker import SearchIntentStateTracker
from .next_hop_query.next_hop_evidence_selector import NextHopEvidenceSelector
from .next_hop_query.next_hop_query_composer import NextHopQueryComposer
from .next_hop_query.relation_evidence_binder import RelationEvidenceBinder
from .next_hop_query.relation_goal_resolver import RelationGoalResolver
from .next_hop_query.query_guard import NextHopQueryGuard
from .next_hop_query.rag_filter import EfficientRAGFilterAdapter, RAGFilterResult
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
    label: str = ""
    sequence_tag: str = ""
    useful_tokens: list[str] = field(default_factory=list)
    useful_spans: list[str] = field(default_factory=list)
    raw_labeler_spans: list[str] = field(default_factory=list)
    grounded_labeler_spans: list[str] = field(default_factory=list)
    classified_spans: list[str] = field(default_factory=list)
    answer_support_spans: list[str] = field(default_factory=list)
    bridge_spans: list[str] = field(default_factory=list)
    span_roles: list[dict[str, str]] = field(default_factory=list)
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
        - rag_filter: 根據 query 與 useful tokens 產生下一輪 query 的 Filter。
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
        rag_filter: EfficientRAGFilterAdapter | None = None,
        next_hop_composer: NextHopQueryComposer | None = None,
        next_hop_evidence_selector: NextHopEvidenceSelector | None = None,
        coverage_assessor: CoverageAssessor | None = None,
        sufficiency_gate: EvidenceSufficiencyGate | None = None,
        evidence_utility_gate: Any | None = None,
        evidence_contract_builder: EvidenceRoleContractBuilder | None = None,
        span_grounder: CandidateSpanGrounder | None = None,
        span_quality_gate: CandidateSpanQualityGate | None = None,
        span_finalizer: RoleAwareSpanFinalizer | None = None,
        span_role_classifier: SpanRoleClassifier | None = None,
        labeler_input_builder: LabelerInputBuilder | None = None,
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
        external_source_loader: Callable[[list[SearchQueryRequest]], int] | None = None,
        max_relation_branches: int = 2,
    ) -> None:
        self.retriever = retriever
        self.labeler = labeler or EfficientRAGLabelerAdapter()
        self.rag_filter = rag_filter or EfficientRAGFilterAdapter()
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
        self.span_grounder = span_grounder or CandidateSpanGrounder()
        self.span_quality_gate = span_quality_gate or CandidateSpanQualityGate()
        self.span_finalizer = span_finalizer or RoleAwareSpanFinalizer()
        self.span_role_classifier = span_role_classifier or SpanRoleClassifier()
        self.labeler_input_builder = labeler_input_builder or LabelerInputBuilder()
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
        stop_reason = "max_iter_reached"

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
                query_key = self._query_key(current_query)
                stop_reason = "duplicate_query" if query_key else "empty_query"
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
                round_trace.stop_reason = "no_new_documents"
                rounds.append(round_trace)
                stop_reason = round_trace.stop_reason
                break

            labeler_batch = self.labeler_input_builder.build_batch(
                question=initial_query,
                current_query=current_query,
                documents=label_documents,
                intent_plan=current_intent_plan,
            )
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

            relation_requires_more = bool(
                current_relation_plan is not None
                and current_relation_plan.is_multihop
                and not current_relation_plan.complete
            )
            if relation_requires_more and current_relation_plan is not None:
                relation_documents = [
                    document
                    for document in round_trace.documents
                    if not document.duplicate
                ]
                direct_contracts = [
                    contract
                    for document in relation_documents
                    for contract in document.direct_contracts
                ]
                direct_resolution = self.relation_resolver.resolve_direct(
                    current_relation_plan,
                    direct_contracts,
                )
                if direct_resolution.resolved_goal_ids:
                    current_relation_plan = direct_resolution.plan
                    if current_intent_plan is not None:
                        current_intent_plan = current_intent_plan.replace(
                            relation_plan=current_relation_plan
                        )
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        "direct_evidence_resolution": {
                            "resolved_goal_ids": direct_resolution.resolved_goal_ids,
                            "contract_count": len(direct_contracts),
                            "plan": current_relation_plan.to_dict(),
                        },
                    }
                    round_trace.stop_reason = "direct_evidence_resolved"
                    rounds.append(round_trace)
                    stop_reason = round_trace.stop_reason
                    break
                binding = self.relation_binder.bind(
                    plan=current_relation_plan,
                    documents=relation_documents,
                )
                resolution = self.relation_resolver.resolve(
                    current_relation_plan,
                    binding.evidence,
                )
                current_relation_plan = resolution.plan
                if current_intent_plan is not None:
                    current_intent_plan = current_intent_plan.replace(
                        relation_plan=current_relation_plan
                    )
                round_trace.filter_metadata = {
                    **round_trace.filter_metadata,
                    "relation_binding": {
                        "evidence": [item.to_dict() for item in binding.evidence],
                        "rejected": list(binding.rejected),
                        "resolved_goal_ids": list(resolution.resolved_goal_ids),
                        "activated_goal_id": resolution.activated_goal_id,
                        "plan": current_relation_plan.to_dict(),
                    },
                }
                if current_relation_plan.complete:
                    round_trace.stop_reason = "relation_goals_resolved"
                    rounds.append(round_trace)
                    stop_reason = round_trace.stop_reason
                    break
                if resolution.activated_goal_id and round_index < self.max_iter:
                    relation_requests = self.next_hop_composer.build_relation_requests(
                        relation_plan=current_relation_plan,
                        constraints=(
                            list(current_intent_plan.must_include or [])
                            if current_intent_plan is not None
                            else []
                        ),
                        max_requests=self.max_relation_branches,
                    )
                    requests = [item.request for item in relation_requests]
                    next_queries = [request.query for request in requests]
                    round_trace.next_queries = list(next_queries)
                    round_trace.next_query = next_queries[0] if next_queries else ""
                    round_trace.filter_metadata["relation_next_hop"] = {
                        "branches": [item.to_dict() for item in relation_requests],
                    }
                    added_count = self._load_external_sources(requests)
                    round_trace.filter_metadata["relation_next_hop"][
                        "added_record_count"
                    ] = added_count
                    if next_queries and added_count > 0:
                        round_trace.stop_reason = "relation_next_hop"
                        rounds.append(round_trace)
                        current_queries = next_queries
                        current_query = next_queries[0]
                        continue
                    round_trace.stop_reason = (
                        "relation_source_empty" if next_queries else "empty_relation_query"
                    )
                    rounds.append(round_trace)
                    stop_reason = round_trace.stop_reason
                    break

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
            gate_result = self.sufficiency_gate.assess(
                question=initial_query,
                documents=non_duplicate_documents,
                intent_plan=current_intent_plan,
                coverage=coverage,
            )
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
                    round_trace.stop_reason = "intent_state_sufficient"
                    rounds.append(round_trace)
                    stop_reason = round_trace.stop_reason
                    break
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
                    round_trace.stop_reason = "coverage_sufficient"
                    rounds.append(round_trace)
                    stop_reason = round_trace.stop_reason
                    break
                if coverage.sufficient and not gate_result.sufficient:
                    round_trace.filter_metadata = {
                        **round_trace.filter_metadata,
                        "sufficiency_gate_failed": gate_result.to_dict(),
                    }

            continue_documents = [
                trace
                for trace in round_trace.documents
                if (
                    not trace.duplicate
                    and trace.valid_for_next_hop
                    and trace.bridge_spans
                )
            ]
            if not continue_documents:
                fallback_result = self._try_fallback_next_query(
                    query=initial_query,
                    documents=[
                        document
                        for document in round_trace.documents
                        if (
                            not document.duplicate
                            and document.bridge_spans
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
                        round_trace.stop_reason = "fallback_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                round_trace.stop_reason = "no_continue_chunks"
                rounds.append(round_trace)
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
                    document.bridge_spans
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
                        round_trace.stop_reason = "fallback_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                round_trace.stop_reason = "no_qualified_continue_chunks"
                rounds.append(round_trace)
                stop_reason = round_trace.stop_reason
                break

            useful_tokens = self._dedupe_tokens(
                token
                for document in qualified_documents
                for token in document.bridge_spans
            )
            round_trace.useful_tokens = useful_tokens
            if not useful_tokens:
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
                        round_trace.stop_reason = "coverage_next_query"
                        rounds.append(round_trace)
                        self._load_external_sources([SearchQueryRequest.fallback(next_query)])
                        current_query = next_query
                        current_queries = [next_query]
                        continue
                round_trace.stop_reason = "no_useful_tokens"
                rounds.append(round_trace)
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

            if not next_query:
                round_trace.stop_reason = "empty_next_query"
                rounds.append(round_trace)
                stop_reason = round_trace.stop_reason
                break
            if self._is_duplicate_query(next_query, seen_query_keys):
                round_trace.stop_reason = "duplicate_next_query"
                rounds.append(round_trace)
                stop_reason = round_trace.stop_reason
                break

            rounds.append(round_trace)
            self._load_external_sources([SearchQueryRequest.fallback(next_query)])
            current_query = next_query
            current_queries = [next_query]
        else:
            stop_reason = "max_iter_reached"

        final_query = (
            rounds[-1].next_query
            if rounds and rounds[-1].next_query
            else current_query
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
        )

    def _load_external_sources(self, requests: list[SearchQueryRequest]) -> int:
        if self.external_source_loader is None or not requests:
            return 0
        try:
            return max(0, int(self.external_source_loader(requests)))
        except Exception:
            return 0

    def _search_with_scores_many(
        self,
        queries: list[str],
        *,
        original_question: str = "",
    ) -> list[tuple[dict[str, Any], float]]:
        ranked_dense_lists: dict[str, list[tuple[str, float]]] = {}
        for index, query in enumerate(queries, start=1):
            ranked_dense_lists[f"dense_branch_{index}"] = self._dense_rank(
                query,
                self.candidate_pool_size,
            )
        if all(
            normalize_text(original_question).casefold()
            != normalize_text(query).casefold()
            for query in queries
        ):
            ranked_dense_lists["dense_original"] = self._dense_rank(
                original_question,
                min(20, self.candidate_pool_size),
            )
        selections = self.passage_selector.select(
            passage_map=self.retriever.passage_map,
            ranked_dense_lists=ranked_dense_lists,
            lexical_query=" ".join([*queries, original_question]),
            max_items=self.top_k,
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
        prepared_query = query
        if self.retriever.model_type == "multilingual-e5-base":
            prepared_query = self.retriever.embedder.prepare_query_text(query)
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
    ) -> RAGFilterResult:
        selected_spans = self._dedupe_tokens(
            span
            for document in documents
            if document.valid_for_next_hop
            for span in document.bridge_spans
        )[:3]
        if not selected_spans:
            return RAGFilterResult(
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
        result: RAGFilterResult,
        round_trace: RetrievalRoundTrace,
        intent_plan: SearchIntentPlan | None,
        seen_query_keys: set[str],
    ) -> str:
        if intent_plan is None:
            return normalize_text(result.query)
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
    ) -> RAGFilterResult | None:
        candidates = [
            document
            for document in documents
            if (
                document.bridge_spans
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
        return RetrievedDocumentTrace(
            document_id=str(document.get("id", "")),
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
        )

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
            spans=candidates,
        )
        diagnostics = dict(result.diagnostics)
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "span_role_classifier": diagnostics,
        }
        if not diagnostics.get("success"):
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

        by_trace_index: dict[int, list[dict[str, str]]] = {}
        for role_result in result.results:
            mapped = candidate_map.get(role_result.id)
            if mapped is None:
                continue
            trace_index, span_text = mapped
            by_trace_index.setdefault(trace_index, []).append(
                {
                    "text": span_text,
                    "role": role_result.role,
                }
            )

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
                direct_spans=answer_support,
                bridge_spans=bridge,
            )
            trace.direct_contracts = [item.to_dict() for item in contracts.direct]
            trace.bridge_contracts = [item.to_dict() for item in contracts.bridge]
            trace.rejected_contracts = [
                item.to_dict() for item in contracts.unsupported
            ]
            trace.answer_support_spans = [
                item.answer_span for item in contracts.direct
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
            trace.valid_for_next_hop = self._can_sequence_tag_hop(trace.sequence_tag) and bool(bridge)
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
            },
        }

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
        max_total: int = 15,
        max_per_document: int = 3,
    ) -> tuple[list[CandidateSpan], dict[str, tuple[int, str]]]:
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
        return normalize_text(query).casefold().strip(" \"'`.,;:-")

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
        - rag_filter: EfficientRAG Filter。
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
        rag_filter: EfficientRAGFilterAdapter | None = None,
        next_hop_composer: NextHopQueryComposer | None = None,
        next_hop_evidence_selector: NextHopEvidenceSelector | None = None,
        model_type: str = "multilingual-e5-base",
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
        max_collection_links_to_fetch: int = 3,
        collection_link_fetch_tokens: int = 5000,
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
        self.labeler = labeler or EfficientRAGLabelerAdapter()
        self.rag_filter = rag_filter or EfficientRAGFilterAdapter()
        self.next_hop_composer = next_hop_composer or NextHopQueryComposer()
        self.next_hop_evidence_selector = (
            next_hop_evidence_selector or NextHopEvidenceSelector()
        )
        self.model_type = model_type
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
        linked_records = self._enrich_collection_links(
            question=text,
            retriever=retriever,
            corpus_session=corpus_session,
        )
        diagnostics["collection_link_enrichment"] = {
            "enabled": self.max_collection_links_to_fetch > 0,
            "added_record_count": len(linked_records),
            "record_ids": [record.id for record in linked_records],
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
            self.page_content_fetcher.fetch_sources(
                filtered,
                max_pages=self.max_pages_to_fetch,
                max_tokens_per_source=5000,
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
                max_chunks_per_url=self.max_chunks_per_url,
                max_records=remaining,
            )
            return len(corpus_session.add_records(new_records))

        retrieval = IterativeRetrievalControl(
            retriever=retriever,
            labeler=self.labeler,
            rag_filter=self.rag_filter,
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
    "RetrievalRoundTrace",
    "RetrievedDocumentTrace",
    "WebRetrievalControl",
    "WebRetrievalResult",
    "WebSearchTrace",
]
