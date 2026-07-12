from __future__ import annotations

import hashlib
import pickle
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from utils.network_utils import normalize_text

from .config import EvidenceItem, SearchSourceCandidate
from .corpus import DocumentChunker, WebCorpusBuilder
from .embeddings import Embedder
from .evidence import EvidenceUtilityGate
from .next_hop_query.coverage_assessor import CoverageAssessor
from .next_hop_query.evidence_sufficiency_gate import EvidenceSufficiencyGate
from .next_hop_query.intent_state_tracker import SearchIntentStateTracker
from .next_hop_query.next_hop_evidence_selector import NextHopEvidenceSelector
from .next_hop_query.next_hop_query_composer import NextHopQueryComposer
from .next_hop_query.query_guard import NextHopQueryGuard
from .next_hop_query.rag_filter import EfficientRAGFilterAdapter, RAGFilterResult
from .passage_retriever import Retriever
from .query import QueryGenerator, SearchIntentPlan
from .source_analyze.rag_labeler import (
    CONTINUE_TAG,
    EfficientRAGLabelerAdapter,
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
    label: str = ""
    sequence_tag: str = ""
    useful_tokens: list[str] = field(default_factory=list)
    continue_probability: float = 0.0
    terminate_probability: float = 0.0
    label_status: str = ""
    valid_for_next_hop: bool = False
    valid_for_evidence: bool = False
    useful_spans: list[str] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)
    support_level: str = ""
    support_strength: str = ""
    normalized_constraints: list[str] = field(default_factory=list)
    answer_spans: list[str] = field(default_factory=list)
    bridge_spans: list[str] = field(default_factory=list)
    utility_context: str = ""
    utility_reasons: list[str] = field(default_factory=list)
    span_recovery_used: bool = False
    can_support_sufficient: bool = False
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
    documents: list[RetrievedDocumentTrace] = field(default_factory=list)
    useful_tokens: list[str] = field(default_factory=list)
    next_query: str = ""
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
        evidence_utility_gate: EvidenceUtilityGate | None = None,
        labeler_input_builder: LabelerInputBuilder | None = None,
        intent_state_tracker: SearchIntentStateTracker | None = None,
        query_guard: NextHopQueryGuard | None = None,
        max_iter: int = 4,
        top_k: int = 10,
        min_retrieval_score: float = 0.75,
        relative_score_margin: float = 0.08,
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
        self.labeler_input_builder = labeler_input_builder or LabelerInputBuilder()
        self.intent_state_tracker = intent_state_tracker or SearchIntentStateTracker()
        self.query_guard = query_guard or NextHopQueryGuard()
        self.max_iter = max(1, max_iter)
        self.top_k = max(1, top_k)
        self.min_retrieval_score = max(0.0, min(1.0, min_retrieval_score))
        self.relative_score_margin = max(
            0.0,
            min(1.0, relative_score_margin),
        )

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
        use_intent_state = intent_plan is not None
        current_intent_plan = intent_plan
        rounds: list[RetrievalRoundTrace] = []
        searched_queries: list[str] = []
        seen_query_keys: set[str] = set()
        seen_document_ids: set[str] = set()
        seen_chunk_keys: set[str] = set()
        unique_document_count = 0
        stop_reason = "max_iter_reached"

        for round_index in range(1, self.max_iter + 1):
            query_key = self._query_key(current_query)
            if not query_key:
                stop_reason = "empty_query"
                break
            if query_key in seen_query_keys:
                stop_reason = "duplicate_query"
                break

            seen_query_keys.add(query_key)
            searched_queries.append(current_query)
            retrieved = self._search_with_scores(current_query)
            round_trace = RetrievalRoundTrace(
                round_index=round_index,
                query=current_query,
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
                if intent_state_sufficient and gate_result.sufficient:
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
                if coverage.sufficient and gate_result.sufficient:
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
                    and trace.support_level == "bridge"
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
                            and document.support_level != "unsupported"
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
                        current_query = next_query
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
                        current_query = next_query
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
                        current_query = next_query
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
            current_query = next_query
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
        )

    def _search_with_scores(
        self,
        query: str,
    ) -> list[tuple[dict[str, Any], float]]:
        prepared_query = query
        if self.retriever.model_type == "multilingual-e5-base":
            prepared_query = self.retriever.embedder.prepare_query_text(query)
        query_vector = self.retriever.embedder.embed([prepared_query])
        search_results = self.retriever.index.search(query_vector, self.top_k)
        if not search_results:
            return []

        document_ids, scores = search_results[0]
        return [
            (self.retriever.passage_map[document_id], float(score))
            for document_id, score in zip(document_ids, scores)
            if document_id in self.retriever.passage_map
        ]

    def _build_next_query(
        self,
        *,
        query: str,
        documents: list[RetrievedDocumentTrace],
        intent_plan: SearchIntentPlan | None = None,
    ) -> RAGFilterResult:
        selection = self.next_hop_evidence_selector.select(
            documents=documents,
            question=query,
            intent_plan=intent_plan,
        )
        if not selection.bridge_spans:
            return RAGFilterResult(
                query="",
                kept_question_tokens=[],
                kept_evidence_tokens=[],
                fallback_used=False,
                metadata={
                    "method": "external_semantic_role_next_hop",
                    "filter_model_used": False,
                    "evidence_selector": selection.to_dict(),
                    "empty_reason": "no_selected_bridge_spans",
                },
            )
        evidence_items = [
            EvidenceItem(
                evidence_id=f"R{index}",
                source_id=f"next-hop-selection-{index}",
                query_id="iterative_retrieval",
                title="",
                text=" ".join(selection.bridge_spans),
                url="",
                matched_terms=selection.bridge_spans,
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
            "evidence_selector": selection.to_dict(),
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
        if (
            not guard_result.accepted
            and str(result.metadata.get("method", "")).startswith("external_")
        ):
            selected_query = ""
        round_trace.filter_metadata = {
            **round_trace.filter_metadata,
            "query_guard": guard_result.to_dict(),
            "query_guard_external_fallback_disabled": (
                not guard_result.accepted
                and str(result.metadata.get("method", "")).startswith("external_")
            ),
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
                and document.support_level == "bridge"
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
        )

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
        trace.label = "useful" if contract.valid_for_evidence else "useless"
        trace.sequence_tag = str(metadata.get("sequence_tag", ""))
        trace.useful_tokens = list(result.kept_tokens)
        trace.useful_spans = list(contract.useful_spans)
        trace.label_status = contract.label_status
        trace.valid_for_next_hop = contract.valid_for_next_hop
        trace.valid_for_evidence = contract.valid_for_evidence
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
        trace.support_level = utility.support_level
        trace.support_strength = utility.support_strength
        trace.normalized_constraints = list(utility.normalized_constraints)
        trace.answer_spans = list(utility.answer_spans)
        trace.bridge_spans = list(utility.bridge_spans)
        trace.utility_context = utility.supporting_context
        trace.utility_reasons = list(utility.reasons)
        trace.span_recovery_used = utility.span_recovery_used
        trace.can_support_sufficient = utility.can_support_sufficient
        trace.valid_for_evidence = utility.valid_for_evidence
        trace.valid_for_next_hop = utility.valid_for_next_hop
        trace.label = "useful" if utility.valid_for_evidence else "useless"
        trace.invalid_reasons = self._dedupe_tokens(
            list(trace.invalid_reasons) + list(utility.reasons)
        )
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
        if not recovered_spans:
            metadata["span_recovery_triggered"] = True
            metadata["span_recovery_mode"] = "restricted"
            metadata["recovered_span_count"] = 0
            metadata["recovered_spans"] = []
            trace.invalid_reasons = self._dedupe_tokens(
                list(trace.invalid_reasons) + list(recovery.reasons)
            )
            return

        trace.useful_spans = self._dedupe_tokens(recovered_spans)
        trace.useful_tokens = list(trace.useful_spans)
        if sequence_tag == "<CONTINUE>":
            trace.label_status = "continue_with_recovered_span"
            trace.valid_for_next_hop = True
            trace.valid_for_evidence = True
        else:
            trace.label_status = "terminate_with_recovered_span"
            trace.valid_for_next_hop = False
            trace.valid_for_evidence = True
        trace.invalid_reasons = self._dedupe_tokens(
            list(trace.invalid_reasons) + list(recovery.reasons)
        )
        metadata["span_recovery_triggered"] = True
        metadata["span_recovery_mode"] = "restricted"
        metadata["recovered_span_count"] = len(trace.useful_spans)
        metadata["recovered_spans"] = list(trace.useful_spans)

    def _labeler_diagnostics(
        self,
        *,
        metadata: dict[str, object],
        trace: RetrievedDocumentTrace,
    ) -> dict[str, object]:
        return {
            "input_mode": metadata.get("input_mode", ""),
            "source_title": metadata.get("source_title", ""),
            "labeler_input_text": metadata.get("labeler_input_text", ""),
            "labeler_input_char_count": metadata.get("labeler_input_char_count", 0),
            "sequence_tag": trace.sequence_tag,
            "continue_probability": round(trace.continue_probability, 6),
            "terminate_probability": round(trace.terminate_probability, 6),
            "finish_probability": metadata.get("finish_probability", 0.0),
            "token_span_count": len(trace.useful_spans),
            "selected_passage": metadata.get("selected_passage", ""),
            "selected_sentence_count": metadata.get("selected_sentence_count", 0),
            "original_char_count": metadata.get("original_char_count", 0),
            "selected_char_count": metadata.get("selected_char_count", 0),
            "sentence_selection_used": bool(metadata.get("sentence_selection_used", False)),
            "sentence_selection_truncated": bool(metadata.get("sentence_selection_truncated", False)),
            "evidence_unit_selector_used": bool(metadata.get("evidence_unit_selector_used", False)),
            "evidence_unit_count": metadata.get("evidence_unit_count", 0),
            "evidence_unit_selected_count": metadata.get("evidence_unit_selected_count", 0),
            "evidence_unit_dropped_count": metadata.get("evidence_unit_dropped_count", 0),
            "evidence_unit_empty_fallback": bool(metadata.get("evidence_unit_empty_fallback", False)),
            "evidence_unit_should_fallback": bool(metadata.get("evidence_unit_should_fallback", False)),
            "evidence_unit_empty_reason": metadata.get("evidence_unit_empty_reason", ""),
            "evidence_unit_retried_raw_text": bool(metadata.get("evidence_unit_retried_raw_text", False)),
            "evidence_unit_selected_types": metadata.get("evidence_unit_selected_types", []),
            "evidence_unit_dropped_flags": metadata.get("evidence_unit_dropped_flags", []),
            "evidence_unit_avg_relevance": metadata.get("evidence_unit_avg_relevance", 0.0),
            "evidence_unit_avg_novelty": metadata.get("evidence_unit_avg_novelty", 0.0),
            "evidence_unit_passage": metadata.get("evidence_unit_passage", ""),
            "evidence_unit_selected": metadata.get("evidence_unit_selected", []),
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
        max_chunks_per_url: int = 12,
        max_corpus_records: int = 300,
        max_urls_per_domain: int = 3,
        min_filtered_sources: int = 5,
        max_iter: int = 4,
        top_k: int = 10,
        min_retrieval_score: float = 0.75,
        relative_score_margin: float = 0.08,
        embedding_batch_size: int = 8,
    ) -> None:
        self.query_generator = query_generator or QueryGenerator()
        if search_tool is None:
            from ..search_tool import SearchTool

            search_tool = SearchTool()
        self.search_tool = search_tool
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
        generated_queries = self._dedupe_queries(
            list(plan.get("queries") or [])
        )[: self.max_queries]
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

        sources, web_searches = self._search_queries(generated_queries)
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
            "corpus_pipeline": (
                "web_search->seer_source_filter->seer_full_page_fetch"
                "->clean->chunk->content_hash_dedup->ngram_dedup"
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
        retrieval = IterativeRetrievalControl(
            retriever=retriever,
            labeler=self.labeler,
            rag_filter=self.rag_filter,
            next_hop_composer=self.next_hop_composer,
            next_hop_evidence_selector=self.next_hop_evidence_selector,
            max_iter=self.max_iter,
            top_k=min(self.top_k, len(records)),
            min_retrieval_score=self.min_retrieval_score,
            relative_score_margin=self.relative_score_margin,
        ).run(text, intent_plan=search_intent_plan)

        diagnostics["initial_retrieval_query"] = text
        diagnostics["stop_reason"] = retrieval.stop_reason
        diagnostics["coverage_summary"] = self._coverage_summary(retrieval)
        diagnostics["search_intent_plan"] = search_intent_plan.to_dict()
        return WebRetrievalResult(
            question=text,
            generated_queries=generated_queries,
            salient_spans=list(plan.get("salient_spans") or []),
            web_searches=web_searches,
            corpus_path=str(corpus_path),
            embedding_path=str(embedding_path),
            corpus_record_count=len(records),
            retrieval=retrieval,
            diagnostics=diagnostics,
            blocked_sources=blocked_sources,
        )

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
        sources: list[SearchSourceCandidate] = []
        traces: list[WebSearchTrace] = []
        for query_index, query in enumerate(queries, start=1):
            try:
                payload = self.search_tool.run(
                    {
                        "input": query,
                        "mode": "structured",
                        "max_results": self.max_results_per_query,
                    }
                )
            except Exception as exc:
                traces.append(
                    WebSearchTrace(
                        query=query,
                        backend="",
                        result_count=0,
                        notices=[f"{type(exc).__name__}: {exc}"],
                    )
                )
                continue

            raw_results = payload.get("results") or []
            source_ids: list[str] = []
            for rank, item in enumerate(raw_results, start=1):
                if not isinstance(item, dict):
                    continue
                url = normalize_text(item.get("url", ""))
                if not url:
                    continue
                source_id = f"S{len(sources) + 1}"
                source_ids.append(source_id)
                sources.append(
                    SearchSourceCandidate(
                        source_id=source_id,
                        query_id=f"Q{query_index}",
                        title=normalize_text(item.get("title", "")) or url,
                        url=url,
                        snippet=normalize_text(
                            item.get("content", item.get("snippet", ""))
                        ),
                        rank=rank,
                    )
                )
            traces.append(
                WebSearchTrace(
                    query=query,
                    backend=str(payload.get("backend", "")),
                    result_count=len(source_ids),
                    source_ids=source_ids,
                    notices=[
                        str(notice)
                        for notice in list(payload.get("notices") or [])
                    ],
                )
            )
        return sources, traces

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
