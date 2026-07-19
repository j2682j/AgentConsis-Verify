from __future__ import annotations

from dataclasses import asdict
from threading import Lock
import time
from typing import Any

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    AnswerCandidate,
    CandidatePathEvaluation,
    VerifierScoreByReasoning,
    NetworkSummary,
)
from core.evidence_runner import EvidenceRunner
from core.candidate_path_evaluator import CandidatePathEvaluator
from core.slm_agent import SLM_Agent
from core.stage1_runner import Stage1Runner
from core.stage2_runner import Stage2Runner
from score import (
    AnswerCandidateClusterer,
    AnswerValidator,
    EvidenceSupportChecker,
    FinalWinnerSelector,
)
from score.versa_prm_scorer import (
    DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    DEFAULT_VERSA_PRM_MODEL_ID,
)
from score.answer_requirement_contract import TaskAnswerRequirementContract
from tools.attachment_workspace import AttachmentWorkspace
from tools.evidence.fact_extraction import TaskFactStore
from utils.network_utils import normalize_for_exact


class Network:
    """
    銝餅銝甈∪? Agent ?函?隞餃?嚗?隤?evidence 皞??tage1 ?蝑?????    early-stop ?文??tage2 cross-agent judging???貉?蝞??蝯?獢??
    Args:
        - question: 雿輻?撓?亦?????        - agents: ???函????? AgentConfig 皜??        - attachment: 憿??鞈??歇閫???批捆??        - tool_manager: ?臬銵?search?alculator 蝑極?瑞?蝞∠??具?        - stage1_runs_per_agent: 瘥?Agent ??Stage1 閬?銴?reasoning ?活?詻?        - max_stage1_workers: Stage1 撟唾? worker ?賊?銝???        - max_stage2_workers: Stage2 judge pair 撟唾? worker ?賊?銝???        - stage2_max_tokens: Stage2 judge ?格活????憭?token ?詻?        - enable_stage1_early_stop: ?臬? Stage1 early-stop??        - enable_stage1_tool_use: ?臬?迂 Stage1 Agent ??reasoning 銝凋蝙?典極?瑯?        - max_stage1_tool_turns: Stage1 tool-use 璅∪?銝???run ?憭極?瑕????        - previous_best_agent_id: ??憿”?暹?雿喟? Agent id嚗??early-stop judge??        - stage1_early_stop_max_retries: early-stop 璇辣銝???憭??啣銵?Stage1 ?活?詻?        - search_result: 憭??????search evidence??        - attachment_result: 憭??????attachment evidence??
    Returns:
        - NetworkSummary: ? final answer?inner agent?tage1 蝯??tage2 judge 蝯???          Agent ??esponse time?oken usage ??tool usage metadata??    """

    EARLY_STOP_VERIFIER_THRESHOLD = 0.95
    EARLY_STOP_RETRY_REASONS = {
        "all_stage1_answers_invalid",
        "all_candidates_contradicted",
        "recoverable_tool_failure_without_answer",
    }

    def __init__(
        self,
        question: str,
        agents: list[AgentConfig],
        attachment: dict[str, Any] | None = None,
        tool_manager: Any | None = None,
        *,
        stage1_runs_per_agent: int = 3,
        max_stage1_workers: int | None = None,
        max_stage2_workers: int | None = None,
        stage2_max_tokens: int = 512,
        enable_stage2_score: bool = True,
        enable_stage1_early_stop: bool = False,
        enable_stage1_tool_use: bool = False,
        max_stage1_tool_turns: int = 2,
        stage1_prepared_search_budget: int = 1,
        stage1_supplemental_evidence_max_items: int = 3,
        previous_best_agent_id: str | None = None,
        stage1_early_stop_max_retries: int = 1,
        enable_evidence_prepare: bool = True,
        enable_compact_search_evidence: bool = False,
        enable_evidence_driven_search: bool = True,
        bypass_search_labeler: bool = False,
        max_parallel_next_hop_queries: int = 2,
        search_result: str = "",
        attachment_result: str = "",
        reference_answer: str = "",
        stage2_verifier: str = "versa",
        versa_prm_model: str = DEFAULT_VERSA_PRM_MODEL_ID,
        versa_prm_base_model: str = DEFAULT_VERSA_PRM_BASE_MODEL_ID,
        versa_prm_device: str = "auto",
        versa_prm_dtype: str = "auto",
        versa_prm_local_files_only: bool = True,
    ) -> None:
        self.question = question
        self.agents = agents
        self.attachment = attachment or {}
        self.tool_manager = tool_manager
        self.stage1_runs_per_agent = stage1_runs_per_agent
        self.max_stage1_workers = max_stage1_workers
        self.max_stage2_workers = max_stage2_workers
        self.stage2_max_tokens = stage2_max_tokens
        self.enable_stage2_score = enable_stage2_score
        self.enable_stage1_early_stop = enable_stage1_early_stop
        self.enable_stage1_tool_use = enable_stage1_tool_use
        self.max_stage1_tool_turns = max(0, max_stage1_tool_turns)
        self.stage1_prepared_search_budget = int(stage1_prepared_search_budget)
        self.stage1_supplemental_evidence_max_items = max(
            1,
            int(stage1_supplemental_evidence_max_items),
        )
        self.previous_best_agent_id = previous_best_agent_id
        self.stage1_early_stop_max_retries = max(0, stage1_early_stop_max_retries)
        self.enable_evidence_prepare = enable_evidence_prepare
        self.enable_compact_search_evidence = enable_compact_search_evidence
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.bypass_search_labeler = bool(bypass_search_labeler)
        self.max_parallel_next_hop_queries = max(0, max_parallel_next_hop_queries)
        self.search_result = search_result
        self.attachment_result = attachment_result
        self.reference_answer = reference_answer
        self.stage2_verifier = stage2_verifier
        self.versa_prm_model = versa_prm_model
        self.versa_prm_base_model = versa_prm_base_model
        self.versa_prm_device = versa_prm_device
        self.versa_prm_dtype = versa_prm_dtype
        self.versa_prm_local_files_only = versa_prm_local_files_only

        self._slm_agents: dict[str, SLM_Agent] = {}
        self._slm_agents_lock = Lock()
        self._token_usage_lock = Lock()
        self._token_usage: dict[str, dict[str, int]] = {}

        self.answer_validator = AnswerValidator()
        self.answer_candidate_clusterer = AnswerCandidateClusterer(
            self.answer_validator
        )
        self.final_winner_selector = FinalWinnerSelector(
            clusterer=self.answer_candidate_clusterer,
            answer_validator=self.answer_validator,
            question=self.question,
        )
        self._last_winner_selection_trace: dict[str, Any] = {}
        self.attachment_workspace = AttachmentWorkspace(self.attachment)
        self.evidence_runner = EvidenceRunner(
            question=self.question,
            attachment=self.attachment,
            tool_manager=self.tool_manager,
            search_result=self.search_result,
            attachment_result=self.attachment_result,
            compact_search_evidence=self.enable_compact_search_evidence,
            enable_evidence_driven_search=self.enable_evidence_driven_search,
            bypass_search_labeler=self.bypass_search_labeler,
            max_parallel_next_hop_queries=self.max_parallel_next_hop_queries,
            attachment_workspace=self.attachment_workspace,
        )
        self.stage1_runner = Stage1Runner(
            question=self.question,
            agents=self.agents,
            get_agent=self._get_slm_agent,
            record_token_usage=self._record_token_usage,
            attachment=self.attachment,
            stage1_runs_per_agent=self.stage1_runs_per_agent,
            max_workers=self.max_stage1_workers,
            enable_tool_use=self.enable_stage1_tool_use,
            max_tool_turns=self.max_stage1_tool_turns,
            tool_manager=self.tool_manager,
            prepared_search_refinement_budget=self.stage1_prepared_search_budget,
            supplemental_search_evidence_max_items=self.stage1_supplemental_evidence_max_items,
            attachment_workspace=self.attachment_workspace,
        )
        self.stage2_runner = Stage2Runner(
            question=self.question,
            agents=self.agents,
            verifier_mode=self.stage2_verifier,
            versa_prm_model=self.versa_prm_model,
            versa_prm_base_model=self.versa_prm_base_model,
            versa_prm_device=self.versa_prm_device,
            versa_prm_dtype=self.versa_prm_dtype,
            versa_prm_local_files_only=self.versa_prm_local_files_only,
        )
        self.evidence_support_checker = EvidenceSupportChecker(self.answer_validator)
        self.candidate_path_evaluator = CandidatePathEvaluator(
            question=self.question,
            clusterer=self.answer_candidate_clusterer,
            evidence_support_checker=self.evidence_support_checker,
            stage2_runner=self.stage2_runner,
        )

    def run(self) -> NetworkSummary:
        """
        ?瑁?摰 Network 隞餃?瘚?嚗???evidence?tage1?arly-stop?tage2??        penalty?core calculation ??winner selection??
        Args:
            - ?～?
        Returns:
            - NetworkSummary: ?祆活隞餃???蝯?獢??挾蝯????貉? metadata??        """
        response_started_at = time.perf_counter()
        self._reset_token_usage()
        self.candidate_path_evaluator.clear_cache()
        evidence_started_at = time.perf_counter()
        evidence = self.evidence_runner.run() if self.enable_evidence_prepare else self._empty_evidence_bundle()
        evidence_prepare_seconds = time.perf_counter() - evidence_started_at
        if self.enable_evidence_prepare:
            evidence["_fact_store"] = self.evidence_runner.fact_store
        if not isinstance(evidence.get("_fact_store"), TaskFactStore):
            evidence["_fact_store"] = TaskFactStore.from_dict(evidence.get("fact_store"))
        evidence["fact_store"] = evidence["_fact_store"].to_dict()
        task_contract = TaskAnswerRequirementContract.build(
            question=self.question,
            answer_requirement=str(evidence.get("answer_requirement") or ""),
            answer_role=str(evidence.get("answer_role") or ""),
            answer_target=str(evidence.get("answer_target") or ""),
            relation_plan=(
                evidence.get("relation_plan")
                if isinstance(evidence.get("relation_plan"), dict)
                else None
            ),
            source=str(evidence.get("answer_requirement_source") or ""),
        )
        evidence["task_answer_requirement_contract"] = task_contract.to_dict()
        evidence["answer_requirement"] = task_contract.requirement_text
        evidence["answer_role"] = task_contract.answer_role
        evidence["answer_target"] = task_contract.answer_target
        evidence["required_relation"] = task_contract.required_relation
        evidence["required_relation_goal_id"] = (
            task_contract.required_relation_goal_id
        )

        stage1_attempts = 0
        early_stop_reason = ""
        early_stop_verifier_results: list[VerifierScoreByReasoning] = []
        direct_consensus_winner: AgentReasoningSummary | None = None
        direct_consensus_supporting_agents: list[str] = []
        stage1_slm_stop_records: list[dict[str, Any]] = []
        stage1_model_lifecycle_records: list[dict[str, Any]] = []
        versa_unload_records: list[dict[str, Any]] = []
        early_stop_trace: list[dict[str, Any]] = []
        stage1_retry_reasons: list[str] = []
        stage1_seconds = 0.0
        versa_scoring_seconds = 0.0
        while True:
            stage1_attempts += 1
            stage1_started_at = time.perf_counter()
            stage1_results = self.stage1_runner.run(evidence)
            stage1_seconds += time.perf_counter() - stage1_started_at
            evidence["fact_store"] = evidence["_fact_store"].to_dict()
            evidence_revision = int(evidence["_fact_store"].revision)
            for record in list(self.stage1_runner.model_lifecycle_records):
                enriched = dict(record)
                enriched["stage1_attempt"] = stage1_attempts
                stage1_model_lifecycle_records.append(enriched)
            if self.enable_stage1_early_stop:
                (
                    direct_consensus_winner,
                    direct_consensus_supporting_agents,
                ) = self._confidence_one_answer_consensus(stage1_results)
            else:
                direct_consensus_winner = None
                direct_consensus_supporting_agents = []
            if direct_consensus_winner is not None:
                versa_started_at = time.perf_counter()
                direct_consensus_results = self._evaluate_early_candidates(
                    [direct_consensus_winner],
                    stage1_results=stage1_results,
                    evidence=evidence,
                    evidence_revision=evidence_revision,
                )
                versa_scoring_seconds += time.perf_counter() - versa_started_at
                direct_consensus_verifier = (
                    direct_consensus_results[0]
                    if direct_consensus_results
                    else VerifierScoreByReasoning(
                        verifier_id="versa_prm",
                        target_agent_id=direct_consensus_winner.agent_id,
                        verifier_score=0.0,
                        metadata={"process_verification": {"versa_status": "unavailable"}},
                    )
                )
                early_stop_verifier_results = direct_consensus_results
                if (
                    self._verifier_critical_floor(direct_consensus_verifier)
                    > self.EARLY_STOP_VERIFIER_THRESHOLD
                    and self._verifier_has_evidence_support(
                        direct_consensus_verifier
                    )
                ):
                    early_stop_winner = direct_consensus_winner
                    early_stop_reason = (
                        "cross_agent_confidence_1.0_answer_consensus_versa_verified"
                    )
                elif (
                    self._verifier_critical_floor(direct_consensus_verifier)
                    <= self.EARLY_STOP_VERIFIER_THRESHOLD
                ):
                    early_stop_winner = None
                    early_stop_reason = (
                        "cross_agent_confidence_1.0_answer_consensus_versa_below_threshold"
                    )
                else:
                    early_stop_winner = None
                    early_stop_reason = (
                        "cross_agent_confidence_1.0_answer_consensus_evidence_unsupported"
                    )
            else:
                versa_started_at = time.perf_counter()
                early_stop_winner, early_stop_verifier_results, early_stop_reason = (
                    self._stage1_early_stop_decision(
                        stage1_results,
                        evidence=evidence,
                        evidence_revision=evidence_revision,
                    )
                )
                if self.enable_stage1_early_stop:
                    versa_scoring_seconds += time.perf_counter() - versa_started_at
            if early_stop_verifier_results:
                versa_unload_records.append(
                    self._unload_versa_scorer(
                        phase="stage1_gate",
                        stage1_attempt=stage1_attempts,
                    )
                )
            retry_reason = self._stage1_retry_reason(
                stage1_results,
                early_stop_verifier_results,
            )
            should_retry_stage1 = (
                self.enable_stage1_early_stop
                and
                early_stop_winner is None
                and retry_reason in self.EARLY_STOP_RETRY_REASONS
                and stage1_attempts <= self.stage1_early_stop_max_retries
            )
            early_stop_trace.append(
                {
                    "attempt": stage1_attempts,
                    "decision_reason": early_stop_reason,
                    "retry_reason": retry_reason if should_retry_stage1 else "",
                    "retry": should_retry_stage1,
                    "winner_agent_id": (
                        early_stop_winner.agent_id if early_stop_winner else ""
                    ),
                    "verifier_results": [
                        {
                            "agent_id": item.target_agent_id,
                            "critical_step_floor": self._verifier_critical_floor(item),
                            "evidence_supported": self._verifier_has_evidence_support(item),
                        }
                        for item in early_stop_verifier_results
                    ],
                }
            )
            if should_retry_stage1:
                stage1_retry_reasons.append(retry_reason)
            if not should_retry_stage1:
                break

        active_results = [result for result in stage1_results if result.active]
        stage1_early_stop_used = early_stop_winner is not None
        stage2_skipped = stage1_early_stop_used or not self.enable_stage2_score
        candidates = self.answer_candidate_clusterer.cluster(
            active_results,
            answer_requirement=str(evidence.get("answer_requirement") or ""),
            answer_role=str(evidence.get("answer_role") or ""),
        )
        if stage1_early_stop_used:
            verifier_results = early_stop_verifier_results
            stage2_skip_reason = "stage1_early_stop"
        else:
            candidate_evaluation_started_at = time.perf_counter()
            evaluation_bundle = self.candidate_path_evaluator.evaluate_candidates(
                candidates=candidates,
                stage1_results=stage1_results,
                evidence=evidence,
                enable_versa=self.enable_stage2_score,
                evidence_revision=evidence_revision,
            )
            if self.enable_stage2_score:
                versa_scoring_seconds += (
                    time.perf_counter() - candidate_evaluation_started_at
                )
            verifier_results = evaluation_bundle.verifier_results
            if self.enable_stage2_score:
                versa_unload_records.append(
                    self._unload_versa_scorer(
                        phase="stage2_scoring",
                        stage1_attempt=stage1_attempts,
                    )
                )
                stage2_skip_reason = ""
            else:
                stage2_skip_reason = "stage2_score_disabled"
        if stage1_early_stop_used:
            evaluation_bundle = self.candidate_path_evaluator.evaluate_candidates(
                candidates=candidates,
                stage1_results=stage1_results,
                evidence=evidence,
                enable_versa=False,
                evidence_revision=evidence_revision,
            )
        direct_consensus_used = (
            stage1_early_stop_used
            and direct_consensus_winner is not None
            and early_stop_reason.startswith("cross_agent_")
        )
        if direct_consensus_used and not verifier_results:
            self._write_direct_consensus_scores(stage1_results)
        else:
            self._write_agent_scores(stage1_results, verifier_results)
        if stage1_early_stop_used:
            winner = early_stop_winner
            self._last_winner_selection_trace = {
                "strategy": "verified_stage1_early_stop",
                "status": "answerable",
                "selection_reason": early_stop_reason,
                "selected_agent_id": winner.agent_id if winner else "",
                "selected_answer": winner.compressed_answer if winner else "",
            }
        else:
            winner_selection_started_at = time.perf_counter()
            winner = self._select_winner(
                stage1_results,
                candidates=candidates,
                path_evaluations=evaluation_bundle.path_evaluations,
                verifier_results=verifier_results,
                evidence=evidence,
            )
            winner_selection_seconds = time.perf_counter() - winner_selection_started_at
        if stage1_early_stop_used:
            winner_selection_seconds = 0.0
        selected_answer = (
            winner.compressed_answer
            if winner is not None
            else str(self._last_winner_selection_trace.get("selected_answer") or "")
        )
        selected_agent_id = (
            winner.agent_id
            if winner is not None
            else (
                "evidence_resolver"
                if self._last_winner_selection_trace.get("selection_origin")
                == "evidence_only_resolution"
                and selected_answer
                else ""
            )
        )
        response_time_seconds = time.perf_counter() - response_started_at

        return NetworkSummary(
            question=self.question,
            final_answer=selected_answer,
            winner_agent_id=selected_agent_id,
            stage1_results=stage1_results,
            verifier_results=verifier_results,
            agent_scores=self.agents,
            metadata={
                "stage1_runs_per_agent": self.stage1_runs_per_agent,
                "response_time_seconds": response_time_seconds,
                "response_time_ms": round(response_time_seconds * 1000, 3),
                "token_usage": self._token_usage_snapshot(),
                "max_stage1_workers": self.stage1_runner.worker_count(),
                "max_stage2_workers": self.stage2_runner.worker_count(active_results),
                "stage2_max_tokens": self.stage2_max_tokens,
                "enable_stage2_score": self.enable_stage2_score,
                "stage2_verifier": self.stage2_verifier,
                "versa_prm_model": self.versa_prm_model,
                "versa_prm_base_model": self.versa_prm_base_model,
                "versa_prm_device": self.versa_prm_device,
                "versa_prm_dtype": self.versa_prm_dtype,
                "versa_prm_local_files_only": self.versa_prm_local_files_only,
                "enable_stage1_tool_use": self.enable_stage1_tool_use,
                "enable_evidence_prepare": self.enable_evidence_prepare,
                "enable_compact_search_evidence": self.enable_compact_search_evidence,
                "query_planner": "signal",
                "enable_evidence_driven_search": self.enable_evidence_driven_search,
                "bypass_search_labeler": self.bypass_search_labeler,
                "max_parallel_next_hop_queries": self.max_parallel_next_hop_queries,
                "max_stage1_tool_turns": self.max_stage1_tool_turns,
                "stage1_prepared_search_budget": self.stage1_prepared_search_budget,
                "stage1_supplemental_evidence_max_items": self.stage1_supplemental_evidence_max_items,
                "stage1_search_gate": self.stage1_runner.search_gate_metadata(),
                "stage1_tool_cache": self.stage1_runner.tool_cache_metadata(),
                "stage1_attachment_reuse": self.attachment_workspace.snapshot(),
                "enable_stage1_early_stop": self.enable_stage1_early_stop,
                "previous_best_agent_id": self.previous_best_agent_id or "",
                "stage1_early_stop_max_retries": self.stage1_early_stop_max_retries,
                "early_stop_verifier_threshold": self.EARLY_STOP_VERIFIER_THRESHOLD,
                "stage1_attempts": stage1_attempts,
                "stage1_retry_reasons": stage1_retry_reasons,
                "early_stop_trace": early_stop_trace,
                "stage1_model_switch_stop_records": [],
                "stage1_model_lifecycle_records": stage1_model_lifecycle_records,
                "stage1_slm_stop_records": stage1_slm_stop_records,
                "versa_unload_records": versa_unload_records,
                "stage1_early_stop": stage1_early_stop_used,
                "stage1_early_stop_reason": early_stop_reason if stage1_early_stop_used else "",
                "stage1_early_stop_last_reason": early_stop_reason,
                "stage2_skipped": stage2_skipped,
                "stage2_skip_reason": stage2_skip_reason,
                "candidate_evaluation": {
                    "evidence_revision": evaluation_bundle.evidence_revision,
                    "cache_hits": evaluation_bundle.cache_hits,
                    "cache_misses": evaluation_bundle.cache_misses,
                    "support_context": dict(
                        evaluation_bundle.support_context_metadata
                    ),
                    "paths": [
                        asdict(item) for item in evaluation_bundle.path_evaluations
                    ],
                },
                "cross_agent_consensus_used": direct_consensus_used,
                "cross_agent_consensus_supporting_agents": (
                    direct_consensus_supporting_agents if direct_consensus_used else []
                ),
                "cross_agent_consensus_answer": (
                    direct_consensus_winner.compressed_answer
                    if direct_consensus_used and direct_consensus_winner is not None
                    else ""
                ),
                "winner_selection": self._winner_selection_metadata(
                    stage1_results,
                    winner,
                    verifier_results=verifier_results,
                ),
                "stage1_context_budget": self._stage1_context_budget_metadata(
                    stage1_results,
                ),
                "reasoning_tokens_by_agent": self._reasoning_tokens_by_agent(
                    stage1_results,
                ),
                "stage_timings_seconds": {
                    "evidence_prepare": evidence_prepare_seconds,
                    "stage1": stage1_seconds,
                    "versa_scoring": versa_scoring_seconds,
                    "winner_selection": winner_selection_seconds,
                },
                "active_agent_count": len(active_results),
                "search_used": bool(evidence["search_result"].strip()),
                "attachment_used": bool(evidence["attachment_result"].strip()),
                "solver_used": bool(evidence["solver_result"].strip()),
                "routing": evidence.get("routing", {}),
                "tool_usage": evidence.get("tool_usage", []),
                "fact_store": (
                    evidence["_fact_store"].to_dict()
                    if isinstance(evidence.get("_fact_store"), TaskFactStore)
                    else evidence.get("fact_store", {})
                ),
            },
        )

    def _empty_evidence_bundle(self) -> dict[str, Any]:
        return {
            "search_result": "",
            "attachment_result": "",
            "attachment_profile": {},
            "solver_result": "",
            "answer_requirement": "",
            "answer_role": "",
            "answer_target": "",
            "routing": {
                "evidence_prepare_enabled": False,
                "use_search": False,
                "use_attachment": False,
                "use_deterministic_solver": False,
                "use_python_solver": False,
            },
            "tool_usage": [],
            "fact_store": TaskFactStore().to_dict(),
        }

    def _confidence_one_answer_consensus(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> tuple[AgentReasoningSummary | None, list[str]]:
        """
        ?曉憭?Agent ??? confidence=1.0 銝?normalized answer ?詨??楊 Agent ?梯???        Args:
            - stage1_results: Stage1Runner ?????Agent ?函?????        Returns:
            - AgentReasoningSummary | None: ?亙??典霅??隞?”閰脩?獢? winner??            - list[str]: ?舀?閰脣霅?獢? Agent id 皜??        """
        confident_results = [
            result
            for result in stage1_results
            if (
                result.active
                and result.winner_selection_eligible
                and result.confidence_score >= 1.0
                and result.compressed_answer.strip()
                and self.answer_validator.is_valid(result.compressed_answer)
            )
        ]
        if len(confident_results) < 2:
            return None, []

        groups: list[list[AgentReasoningSummary]] = []
        for result in confident_results:
            for group in groups:
                if self._same_normalized_answer(
                    result.compressed_answer,
                    group[0].compressed_answer,
                ):
                    group.append(result)
                    break
            else:
                groups.append([result])

        consensus_groups = [group for group in groups if len(group) >= 2]
        if not consensus_groups:
            return None, []

        best_group = max(
            consensus_groups,
            key=lambda group: (
                len(group),
                sum(result.confidence_score for result in group),
            ),
        )
        winner = best_group[0]
        return winner, [result.agent_id for result in best_group]

    def _same_normalized_answer(self, answer_a: str, answer_b: str) -> bool:
        """
        ?斗?拙楊 Agent ?蝑??臬?箇??獢?        Args:
            - answer_a: 蝚砌??蝑???            - answer_b: 蝚砌??蝑???        Returns:
            - bool: ?拙?獢? exact normalization 敺?衣??        """
        return normalize_for_exact(answer_a) == normalize_for_exact(answer_b)

    def _unload_versa_scorer(
        self,
        *,
        phase: str,
        stage1_attempt: int,
    ) -> dict[str, Any]:
        """
        Release the VersaPRM scorer after a verifier scoring phase.

        Args:
            - phase: Scoring phase that just finished.
            - stage1_attempt: Current Stage1 attempt index.

        Returns:
            - dict[str, Any]: Unload status record.
        """
        try:
            record = dict(self.stage2_runner.unload())
            record["ok"] = True
        except Exception as exc:
            record = {
                "ok": False,
                "was_loaded": False,
                "warning": f"{type(exc).__name__}: {exc}",
            }
        record["phase"] = phase
        record["stage1_attempt"] = stage1_attempt
        return record

    def _stage1_context_budget_metadata(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> dict[str, Any]:
        budgets = [
            dict(run.context_budget or {})
            for result in stage1_results
            for run in result.runs
            if isinstance(run.context_budget, dict) and run.context_budget
        ]
        if not budgets:
            return {}
        original_chars = [int(item.get("original_chars", 0) or 0) for item in budgets]
        final_chars = [int(item.get("final_chars", 0) or 0) for item in budgets]
        truncated_sections = sorted(
            {
                str(section)
                for item in budgets
                for section in list(item.get("truncated_sections") or [])
            }
        )
        return {
            "run_count": len(budgets),
            "original_chars_total": sum(original_chars),
            "final_chars_total": sum(final_chars),
            "original_chars_avg": round(sum(original_chars) / max(1, len(original_chars)), 2),
            "final_chars_avg": round(sum(final_chars) / max(1, len(final_chars)), 2),
            "chars_reduction_total": max(0, sum(original_chars) - sum(final_chars)),
            "truncation_applied_count": sum(
                1 for item in budgets if bool(item.get("truncation_applied"))
            ),
            "dropped_evidence_count": sum(
                int(item.get("dropped_evidence_count", 0) or 0) for item in budgets
            ),
            "truncated_sections": truncated_sections,
        }

    def _reasoning_tokens_by_agent(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> dict[str, dict[str, int]]:
        """Aggregate Stage1 run token usage by Agent for cost diagnostics."""

        return {
            result.agent_id: {
                "run_count": len(result.runs),
                "prompt_tokens": sum(int(run.prompt_tokens or 0) for run in result.runs),
                "completion_tokens": sum(
                    int(run.completion_tokens or 0) for run in result.runs
                ),
                "total_tokens": sum(int(run.total_tokens or 0) for run in result.runs),
            }
            for result in stage1_results
        }

    def _write_direct_consensus_scores(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> None:
        """
        頝?Agent ?梯??湔頛詨???身 AgentConfig ??敹?閰?甈???        Args:
            - stage1_results: Stage1Runner ?????Agent ?函?????        Returns:
            - None??        """
        result_by_agent = {result.agent_id: result for result in stage1_results}
        for config in self.agents:
            result = result_by_agent.get(config.agent_id)
            config.confidence_score = result.confidence_score if result else 0.0
            config.verifier_scores = []
            config.avg_verifier_score = 0.0
            config.penalty_score = 0.0
            config.penalty_reasons = []
            config.total_score = (
                config.confidence_score
                if result is not None and result.active
                else float("-inf")
            )

    def _write_agent_scores(
        self,
        stage1_results: list[AgentReasoningSummary],
        verifier_results: list[VerifierScoreByReasoning],
    ) -> None:
        """
        撠?Stage1 confidence ??Stage2 judge score 撖怠? AgentConfig??
        Args:
            - stage1_results: Stage1Runner ?Ｙ??? Agent ??蝯???            - verifier_results: Stage2 ?Ｙ???reasoning 閰?蝯???
        Returns:
            - None??        """
        result_by_agent = {result.agent_id: result for result in stage1_results}
        scores_by_target: dict[str, list[float]] = {}
        for result in verifier_results:
            scores_by_target.setdefault(result.target_agent_id, []).append(
                result.verifier_score
            )

        for config in self.agents:
            result = result_by_agent.get(config.agent_id)
            config.confidence_score = result.confidence_score if result else 0.0
            config.verifier_scores = list(scores_by_target.get(config.agent_id, []))
            config.avg_verifier_score = (
                sum(config.verifier_scores) / len(config.verifier_scores)
                if config.verifier_scores
                else 0.0
            )
            config.penalty_score = 0.0
            config.penalty_reasons = []
            config.total_score = (
                config.confidence_score + config.avg_verifier_score
                if result is not None and result.active
                else float("-inf")
            )

    def _stage1_early_stop_decision(
        self,
        stage1_results: list[AgentReasoningSummary],
        *,
        evidence: dict[str, Any] | None = None,
        evidence_revision: int = 0,
    ) -> tuple[AgentReasoningSummary | None, list[VerifierScoreByReasoning], str]:
        """
        ?寞? Stage1 confidence ??previous-best judge 蝯??斗?臬???迫??
        Args:
            - stage1_results: Stage1Runner ?Ｙ??? Agent ?蝯???
        Returns:
            - AgentReasoningSummary | None: ??early-stop ??嚗??喳??箏??            - list[VerifierScoreByReasoning]: early-stop ???Ｙ???judge 蝯???            - str: early-stop ?斗????        """
        if not self.enable_stage1_early_stop:
            return None, [], ""

        active_results = [
            result
            for result in stage1_results
            if (
                result.active
                and result.winner_selection_eligible
                and result.compressed_answer.strip()
            )
        ]
        if not active_results:
            return None, [], "no_active_stage1_result"

        confident_results = [
            result
            for result in active_results
            if result.confidence_score >= 1.0
        ]
        if confident_results:
            verifier_results = self._evaluate_early_candidates(
                confident_results,
                stage1_results=stage1_results,
                evidence=evidence or {},
                evidence_revision=evidence_revision,
            )
            verified_results = [
                result
                for result in verifier_results
                if self._verifier_critical_floor(result)
                > self.EARLY_STOP_VERIFIER_THRESHOLD
            ]
            if not verified_results:
                return None, verifier_results, "confidence_1.0_versa_reward_below_threshold"
            supported_results = [
                result
                for result in verified_results
                if self._verifier_has_evidence_support(result)
            ]
            if not supported_results:
                return None, verifier_results, "confidence_1.0_evidence_unsupported"

            best_verifier_result = max(
                supported_results,
                key=lambda result: (
                    self._verifier_support_priority(result),
                    result.verifier_score,
                ),
            )
            winner = next(
                result
                for result in confident_results
                if result.agent_id == best_verifier_result.target_agent_id
            )
            return winner, verifier_results, "confidence_1.0_positive_versa_reward"

        max_confidence = max(result.confidence_score for result in active_results)
        if max_confidence != 0.67:
            return None, [], "max_confidence_not_0.67"

        candidates = [
            result
            for result in active_results
            if result.confidence_score == max_confidence
        ]
        verifier_results = self._evaluate_early_candidates(
            candidates,
            stage1_results=stage1_results,
            evidence=evidence or {},
            evidence_revision=evidence_revision,
        )
        verified_results = [
            result
            for result in verifier_results
            if self._verifier_critical_floor(result)
            > self.EARLY_STOP_VERIFIER_THRESHOLD
        ]
        if not verified_results:
            return None, verifier_results, "confidence_0.67_versa_reward_below_threshold"
        supported_results = [
            result
            for result in verified_results
            if self._verifier_has_evidence_support(result)
        ]
        if not supported_results:
            return None, verifier_results, "confidence_0.67_evidence_unsupported"

        best_verifier_result = max(
            supported_results,
            key=lambda result: (
                self._verifier_support_priority(result),
                result.verifier_score,
            ),
        )
        winner = next(
            result
            for result in candidates
            if result.agent_id == best_verifier_result.target_agent_id
        )
        return winner, verifier_results, "confidence_0.67_positive_versa_reward"

    def _stage1_retry_reason(
        self,
        stage1_results: list[AgentReasoningSummary],
        verifier_results: list[VerifierScoreByReasoning],
    ) -> str:
        """Return a retry reason only for failures that another attempt can repair."""

        valid_results = [
            result
            for result in stage1_results
            if (
                result.active
                and result.winner_selection_eligible
                and self.answer_validator.is_valid(result.compressed_answer)
            )
        ]
        if not valid_results:
            retryable_tool_failure = any(
                bool(tool_result.get("retryable"))
                for summary in stage1_results
                for run in summary.runs
                for tool_result in list(run.tool_results or [])
                if isinstance(tool_result, dict)
            )
            return (
                "recoverable_tool_failure_without_answer"
                if retryable_tool_failure
                else "all_stage1_answers_invalid"
            )

        verification_statuses = [
            str(
                (
                    result.metadata.get("evidence_support", {})
                    if isinstance(result.metadata, dict)
                    else {}
                ).get("verification_status")
                or (
                    result.metadata.get("evidence_support", {})
                    if isinstance(result.metadata, dict)
                    else {}
                ).get("status")
                or ""
            ).strip().casefold()
            for result in verifier_results
        ]
        if verification_statuses and all(
            status in {"contradicted", "evidence_contradicted"}
            for status in verification_statuses
        ):
            return "all_candidates_contradicted"
        return ""

    def _select_winner(
        self,
        stage1_results: list[AgentReasoningSummary],
        *,
        candidates: list[AnswerCandidate] | None = None,
        path_evaluations: list[CandidatePathEvaluation] | None = None,
        verifier_results: list[VerifierScoreByReasoning] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> AgentReasoningSummary | None:
        """
        ?寞? AgentConfig 銝剔? total_score?onfidence_score ??avg_verifier_score ?詨?蝯?winner??
        Args:
            - stage1_results: Stage1Runner ?Ｙ??? Agent ?蝯???
        Returns:
            - AgentReasoningSummary | None: ?蝯??箇??蝯?嚗瘝? active agent ????None??        """
        candidate_evidence = evidence or {}
        candidates = candidates or self.answer_candidate_clusterer.cluster(
            stage1_results,
            answer_requirement=str(candidate_evidence.get("answer_requirement") or ""),
            answer_role=str(candidate_evidence.get("answer_role") or ""),
        )
        selection = self.final_winner_selector.select(
            stage1_results=stage1_results,
            candidates=candidates,
            path_evaluations=path_evaluations or [],
            verifier_results=verifier_results or [],
            evidence=evidence or {},
        )
        self._last_winner_selection_trace = selection.to_dict()
        return selection.winner

    def _evaluate_early_candidates(
        self,
        summaries: list[AgentReasoningSummary],
        *,
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any],
        evidence_revision: int,
    ) -> list[VerifierScoreByReasoning]:
        """Evaluate the representative path for each early-stop candidate."""

        clustered = self.answer_candidate_clusterer.cluster(
            stage1_results,
            answer_requirement=str(evidence.get("answer_requirement") or ""),
            answer_role=str(evidence.get("answer_role") or ""),
        )
        requested = {
            (
                summary.agent_id,
                self.answer_candidate_clusterer.candidate_key(summary.compressed_answer),
            )
            for summary in summaries
        }
        paths = [
            member
            for candidate in clustered
            for member in candidate.members
            if (member.agent_id, candidate.candidate_key) in requested
        ]
        evaluations = self.candidate_path_evaluator.evaluate_paths(
            paths=paths,
            stage1_results=stage1_results,
            evidence=evidence,
            enable_versa=True,
            evidence_revision=evidence_revision,
        )
        best_by_agent: dict[str, Any] = {}
        for item in evaluations:
            current = best_by_agent.get(item.identity.agent_id)
            key = (
                int(item.direct_support),
                float(item.critical_step_floor or 0.0),
                float(item.critical_step_geometric_mean or 0.0),
                -item.identity.run_index,
            )
            if current is None or key > current[0]:
                best_by_agent[item.identity.agent_id] = (key, item)
        return [
            self.candidate_path_evaluator.to_verifier_result(value[1])
            for value in best_by_agent.values()
            if value[1].versa_available
        ]

    def _support_metadata_by_agent(
        self,
        verifier_results: list[VerifierScoreByReasoning],
    ) -> dict[str, dict[str, Any]]:
        support_by_agent: dict[str, dict[str, Any]] = {}
        for result in verifier_results or []:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            support = metadata.get("evidence_support")
            if isinstance(support, dict):
                existing = support_by_agent.get(result.target_agent_id, {})
                try:
                    existing_priority = int(existing.get("priority") or -1)
                    candidate_priority = int(support.get("priority") or -1)
                except (TypeError, ValueError):
                    existing_priority = candidate_priority = -1
                if not existing or candidate_priority > existing_priority:
                    support_by_agent[result.target_agent_id] = support
        return support_by_agent

    def _verifier_support_priority(
        self,
        result: VerifierScoreByReasoning,
    ) -> int:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        support = metadata.get("evidence_support")
        if not isinstance(support, dict):
            return 0
        try:
            return int(support.get("priority") or 0)
        except (TypeError, ValueError):
            return 0

    def _verifier_has_evidence_support(
        self,
        result: VerifierScoreByReasoning,
    ) -> bool:
        return self._verifier_support_priority(result) >= 3

    def _verifier_critical_floor(
        self,
        result: VerifierScoreByReasoning,
    ) -> float:
        """取得 early-stop 使用的最低關鍵步驟 reward probability。"""
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        process = metadata.get("process_verification")
        if isinstance(process, dict) and "critical_step_floor" in process:
            try:
                return float(process.get("critical_step_floor") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return float(result.verifier_score or 0.0)

    def _winner_selection_metadata(
        self,
        stage1_results: list[AgentReasoningSummary],
        winner: AgentReasoningSummary | None,
        *,
        verifier_results: list[VerifierScoreByReasoning] | None = None,
    ) -> dict[str, Any]:
        """
        Summarize abstention-aware winner selection without adding scores.
        """
        answerable = [
            result
            for result in stage1_results
            if result.winner_selection_eligible and result.compressed_answer.strip()
        ]
        abstained = [
            result
            for result in stage1_results
            if result.winner_selection_status == "all_runs_abstained"
        ]
        invalid = [
            result
            for result in stage1_results
            if result.winner_selection_status in {"all_runs_invalid", "no_final_answer", "no_stage1_runs"}
        ]
        low_coverage = [
            result
            for result in stage1_results
            if result.winner_selection_status == "mixed_low_coverage"
        ]
        selection_trace = dict(self._last_winner_selection_trace or {})
        trace_status = str(selection_trace.get("status") or "")
        if winner is not None:
            status = "answerable"
        elif answerable and trace_status in {
            "unresolved",
            "unresolved_requirement_conflict",
            "unresolved_unsupported_factual_conflict",
            "unresolved_factual_without_support",
            "unresolved_missing_verification",
            "unresolved_exact_tie",
            "no_eligible_candidate",
            "selected_member_missing",
        }:
            status = trace_status
        elif stage1_results and len(abstained) == len(stage1_results):
            status = "all_agents_abstained"
        elif stage1_results:
            status = "all_agents_abstained_or_invalid"
        else:
            status = "no_stage1_results"
        support_by_agent = self._support_metadata_by_agent(verifier_results or [])

        return {
            "status": status,
            "winner_agent_id": winner.agent_id if winner else "",
            "winner_answer": winner.compressed_answer if winner else "",
            "selection_trace": selection_trace,
            "answerable_agent_count": len(answerable),
            "abstained_agent_count": len(abstained),
            "invalid_agent_count": len(invalid),
            "low_coverage_agent_count": len(low_coverage),
            "agent_statuses": [
                {
                    "agent_id": result.agent_id,
                    "winner_selection_eligible": result.winner_selection_eligible,
                    "winner_selection_status": result.winner_selection_status,
                    "eligible_run_count": result.eligible_run_count,
                    "abstention_run_count": result.abstention_run_count,
                    "invalid_run_count": result.invalid_run_count,
                    "run_validity_labels": result.run_validity_labels,
                    "evidence_support_status": str(
                        support_by_agent.get(result.agent_id, {}).get("status")
                        or "no_support"
                    ),
                    "evidence_support_priority": int(
                        support_by_agent.get(result.agent_id, {}).get("priority")
                        or 0
                    ),
                }
                for result in stage1_results
            ],
        }

    def _get_slm_agent(self, config: AgentConfig) -> SLM_Agent:
        """
        敺遙?敹怠??? SLM_Agent嚗撠撱箇??? AgentConfig 撱箇???
        Args:
            - config: ?? agent_id?odel_name ??temperature ??AgentConfig??
        Returns:
            - SLM_Agent: ?舫?銴蝙?函?璅∪??澆?拐辣??        """
        with self._slm_agents_lock:
            agent = self._slm_agents.get(config.agent_id)
            if agent is None:
                agent = SLM_Agent(
                    model_name=config.model_name,
                    temperature=config.temperature,
                )
                self._slm_agents[config.agent_id] = agent
            return agent

    def _reset_token_usage(self) -> None:
        """
        ?蔭?祆活隞餃???Stage1?tage2 ??total token usage 蝯梯???
        Args:
            - ?～?
        Returns:
            - None??        """
        with self._token_usage_lock:
            self._token_usage = {
                "stage1": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "stage2": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    def _record_token_usage(
        self,
        *,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        蝝臬????挾??prompt token?ompletion token ??total token??
        Args:
            - stage: token usage ?撅祇?畾蛛?靘? stage1 ??stage2??            - prompt_tokens: ?祆活?澆瘨? prompt token ?詻?            - completion_tokens: ?祆活?澆?Ｙ???completion token ?詻?
        Returns:
            - None??        """
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = prompt_tokens + completion_tokens

        with self._token_usage_lock:
            if stage not in self._token_usage:
                self._token_usage[stage] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            for bucket_name in (stage, "total"):
                bucket = self._token_usage[bucket_name]
                bucket["prompt_tokens"] += prompt_tokens
                bucket["completion_tokens"] += completion_tokens
                bucket["total_tokens"] += total_tokens

    def _token_usage_snapshot(self) -> dict[str, dict[str, int]]:
        """
        撱箇??桀? token usage ???典翰?改?靘?NetworkSummary metadata 雿輻??
        Args:
            - ?～?
        Returns:
            - dict[str, dict[str, int]]: ??畾菔? total ??token 蝯梯???        """
        with self._token_usage_lock:
            return {
                stage: dict(values)
                for stage, values in self._token_usage.items()
            }


__all__ = ["Network"]
