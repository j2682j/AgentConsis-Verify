from __future__ import annotations

import re
from threading import Lock
import time
from typing import Any

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    VerifierScoreByReasoning,
    NetworkSummary,
)
from core.evidence_runner import EvidenceRunner
from core.slm_agent import SLM_Agent
from core.stage1_runner import Stage1Runner
from core.stage2_runner import Stage2Runner
from score import AnswerValidator
from score.versa_prm_scorer import (
    DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    DEFAULT_VERSA_PRM_MODEL_ID,
)
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
        "cross_agent_confidence_1.0_answer_consensus_versa_below_threshold",
        "confidence_1.0_versa_reward_below_threshold",
        "confidence_0.67_versa_reward_below_threshold",
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
        previous_best_agent_id: str | None = None,
        stage1_early_stop_max_retries: int = 1,
        enable_evidence_prepare: bool = True,
        enable_compact_search_evidence: bool = False,
        enable_evidence_driven_search: bool = True,
        enable_deterministic_handler_router: bool = False,
        enable_tool_planner: bool = False,
        max_parallel_next_hop_queries: int = 2,
        search_result: str = "",
        attachment_result: str = "",
        reference_answer: str = "",
        stage2_verifier: str = "versa",
        versa_prm_model: str = DEFAULT_VERSA_PRM_MODEL_ID,
        versa_prm_base_model: str = DEFAULT_VERSA_PRM_BASE_MODEL_ID,
        versa_prm_device: str = "auto",
        versa_prm_dtype: str = "auto",
        versa_prm_local_files_only: bool = False,
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
        self.previous_best_agent_id = previous_best_agent_id
        self.stage1_early_stop_max_retries = max(0, stage1_early_stop_max_retries)
        self.enable_evidence_prepare = enable_evidence_prepare
        self.enable_compact_search_evidence = enable_compact_search_evidence
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.enable_deterministic_handler_router = enable_deterministic_handler_router
        self.enable_tool_planner = enable_tool_planner
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
        self._last_winner_selection_trace: dict[str, Any] = {}
        self.evidence_runner = EvidenceRunner(
            question=self.question,
            attachment=self.attachment,
            tool_manager=self.tool_manager,
            search_result=self.search_result,
            attachment_result=self.attachment_result,
            compact_search_evidence=self.enable_compact_search_evidence,
            enable_evidence_driven_search=self.enable_evidence_driven_search,
            enable_deterministic_handler_router=self.enable_deterministic_handler_router,
            enable_tool_planner=self.enable_tool_planner,
            max_parallel_next_hop_queries=self.max_parallel_next_hop_queries,
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

    def run(self) -> NetworkSummary:
        """
        ?瑁?摰 Network 隞餃?瘚?嚗???evidence?tage1?arly-stop?tage2??        penalty?core calculation ??winner selection??
        Args:
            - ?～?
        Returns:
            - NetworkSummary: ?祆活隞餃???蝯?獢??挾蝯????貉? metadata??        """
        response_started_at = time.perf_counter()
        self._reset_token_usage()
        evidence = self.evidence_runner.run() if self.enable_evidence_prepare else self._empty_evidence_bundle()

        stage1_attempts = 0
        early_stop_reason = ""
        early_stop_verifier_results: list[VerifierScoreByReasoning] = []
        direct_consensus_winner: AgentReasoningSummary | None = None
        direct_consensus_supporting_agents: list[str] = []
        stage1_slm_stop_records: list[dict[str, Any]] = []
        stage1_model_lifecycle_records: list[dict[str, Any]] = []
        versa_unload_records: list[dict[str, Any]] = []
        while True:
            stage1_attempts += 1
            stage1_results = self.stage1_runner.run(evidence)
            for record in list(self.stage1_runner.model_lifecycle_records):
                enriched = dict(record)
                enriched["stage1_attempt"] = stage1_attempts
                stage1_model_lifecycle_records.append(enriched)
            (
                direct_consensus_winner,
                direct_consensus_supporting_agents,
            ) = self._confidence_one_answer_consensus(stage1_results)
            if direct_consensus_winner is not None:
                direct_consensus_verifier = self.stage2_runner.score_candidate(
                    direct_consensus_winner
                )
                early_stop_verifier_results = [direct_consensus_verifier]
                if (
                    direct_consensus_verifier.verifier_score
                    > self.EARLY_STOP_VERIFIER_THRESHOLD
                ):
                    early_stop_winner = direct_consensus_winner
                    early_stop_reason = (
                        "cross_agent_confidence_1.0_answer_consensus_versa_verified"
                    )
                else:
                    early_stop_winner = None
                    early_stop_reason = (
                        "cross_agent_confidence_1.0_answer_consensus_versa_below_threshold"
                    )
            else:
                early_stop_winner, early_stop_verifier_results, early_stop_reason = (
                    self._stage1_early_stop_decision(stage1_results)
                )
            if early_stop_verifier_results:
                versa_unload_records.append(
                    self._unload_versa_scorer(
                        phase="stage1_gate",
                        stage1_attempt=stage1_attempts,
                    )
                )
            should_retry_stage1 = (
                early_stop_winner is None
                and early_stop_reason in self.EARLY_STOP_RETRY_REASONS
                and stage1_attempts <= self.stage1_early_stop_max_retries
            )
            if not should_retry_stage1:
                break

        active_results = [result for result in stage1_results if result.active]
        stage1_early_stop_used = early_stop_winner is not None
        stage2_skipped = stage1_early_stop_used or not self.enable_stage2_score
        if stage1_early_stop_used:
            verifier_results = early_stop_verifier_results
            stage2_skip_reason = "stage1_early_stop"
        elif not self.enable_stage2_score:
            verifier_results = []
            stage2_skip_reason = "stage2_score_disabled"
        else:
            verifier_results = self.stage2_runner.run(active_results)
            versa_unload_records.append(
                self._unload_versa_scorer(
                    phase="stage2_scoring",
                    stage1_attempt=stage1_attempts,
                )
            )
            stage2_skip_reason = ""
        direct_consensus_used = (
            stage1_early_stop_used
            and direct_consensus_winner is not None
            and early_stop_reason.startswith("cross_agent_")
        )
        if direct_consensus_used and not verifier_results:
            self._write_direct_consensus_scores(stage1_results)
        else:
            self._write_agent_scores(stage1_results, verifier_results)
        winner = early_stop_winner or self._select_winner(
            stage1_results,
            verifier_results=verifier_results,
            evidence=evidence,
        )
        response_time_seconds = time.perf_counter() - response_started_at

        return NetworkSummary(
            question=self.question,
            final_answer=winner.compressed_answer if winner else "",
            winner_agent_id=winner.agent_id if winner else "",
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
                "enable_deterministic_handler_router": self.enable_deterministic_handler_router,
                "enable_tool_planner": self.enable_tool_planner,
                "max_parallel_next_hop_queries": self.max_parallel_next_hop_queries,
                "max_stage1_tool_turns": self.max_stage1_tool_turns,
                "enable_stage1_early_stop": self.enable_stage1_early_stop,
                "previous_best_agent_id": self.previous_best_agent_id or "",
                "stage1_early_stop_max_retries": self.stage1_early_stop_max_retries,
                "early_stop_verifier_threshold": self.EARLY_STOP_VERIFIER_THRESHOLD,
                "stage1_attempts": stage1_attempts,
                "stage1_model_switch_stop_records": [],
                "stage1_model_lifecycle_records": stage1_model_lifecycle_records,
                "stage1_slm_stop_records": stage1_slm_stop_records,
                "versa_unload_records": versa_unload_records,
                "stage1_early_stop": stage1_early_stop_used,
                "stage1_early_stop_reason": early_stop_reason if stage1_early_stop_used else "",
                "stage1_early_stop_last_reason": early_stop_reason,
                "stage2_skipped": stage2_skipped,
                "stage2_skip_reason": stage2_skip_reason,
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
                ),
                "stage1_context_budget": self._stage1_context_budget_metadata(
                    stage1_results,
                ),
                "active_agent_count": len(active_results),
                "search_used": bool(evidence["search_result"].strip()),
                "attachment_used": bool(evidence["attachment_result"].strip()),
                "solver_used": bool(evidence["solver_result"].strip()),
                "routing": evidence.get("routing", {}),
                "tool_usage": evidence.get("tool_usage", []),
            },
        )

    def _empty_evidence_bundle(self) -> dict[str, Any]:
        return {
            "search_result": "",
            "attachment_result": "",
            "solver_result": "",
            "routing": {
                "evidence_prepare_enabled": False,
                "use_search": False,
                "use_attachment": False,
                "use_deterministic_solver": False,
                "use_python_solver": False,
            },
            "tool_usage": [],
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
            verifier_results = [
                self.stage2_runner.score_candidate(candidate)
                for candidate in confident_results
            ]
            verified_results = [
                result
                for result in verifier_results
                if result.verifier_score > self.EARLY_STOP_VERIFIER_THRESHOLD
            ]
            if not verified_results:
                return None, verifier_results, "confidence_1.0_versa_reward_below_threshold"

            best_verifier_result = max(
                verified_results,
                key=lambda result: result.verifier_score,
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
        verifier_results = [
            self.stage2_runner.score_candidate(candidate)
            for candidate in candidates
        ]
        verified_results = [
            result
            for result in verifier_results
            if result.verifier_score > self.EARLY_STOP_VERIFIER_THRESHOLD
        ]
        if not verified_results:
            return None, verifier_results, "confidence_0.67_versa_reward_below_threshold"

        best_verifier_result = max(
            verified_results,
            key=lambda result: result.verifier_score,
        )
        winner = next(
            result
            for result in candidates
            if result.agent_id == best_verifier_result.target_agent_id
        )
        return winner, verifier_results, "confidence_0.67_positive_versa_reward"

    def _select_winner(
        self,
        stage1_results: list[AgentReasoningSummary],
        *,
        verifier_results: list[VerifierScoreByReasoning] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> AgentReasoningSummary | None:
        """
        ?寞? AgentConfig 銝剔? total_score?onfidence_score ??avg_verifier_score ?詨?蝯?winner??
        Args:
            - stage1_results: Stage1Runner ?Ｙ??? Agent ?蝯???
        Returns:
            - AgentReasoningSummary | None: ?蝯??箇??蝯?嚗瘝? active agent ????None??        """
        result_by_agent = {result.agent_id: result for result in stage1_results}
        review_fallback_winner = self._select_all_review_fallback_winner(stage1_results)
        if review_fallback_winner is not None:
            return review_fallback_winner

        clustered_winner = self._select_winner_by_answer_clusters(
            stage1_results,
            verifier_results=verifier_results or [],
            evidence=evidence or {},
        )
        if clustered_winner is not None:
            return clustered_winner

        active_agents = [
            config
            for config in self.agents
            if (
                result_by_agent.get(config.agent_id)
                and result_by_agent[config.agent_id].active
                and result_by_agent[config.agent_id].winner_selection_eligible
                and result_by_agent[config.agent_id].compressed_answer.strip()
            )
        ]
        if not active_agents:
            return None
        winner_config = max(
            active_agents,
            key=lambda config: (
                config.total_score,
                config.confidence_score,
                config.avg_verifier_score,
            ),
        )
        return result_by_agent[winner_config.agent_id]

    def _select_all_review_fallback_winner(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> AgentReasoningSummary | None:
        """
        當所有 Agent 都只能靠 self-review fallback 給答案時，優先選目前最佳 Agent。

        Args:
         - stage1_results: Stage1Runner 回傳的 Agent summaries。

        Returns:
         - AgentReasoningSummary | None: previous_best_agent_id 對應的可用 review answer。
        """
        fallback_results = [
            result
            for result in stage1_results
            if (result.self_review_metadata or {}).get("fallback_scope")
            == "all_agents_need_review"
        ]
        if not fallback_results:
            return None
        if len(fallback_results) != len(stage1_results):
            return None
        if not self.previous_best_agent_id:
            return None

        review_results = [
            result
            for result in fallback_results
            if (
                result.active
                and result.winner_selection_eligible
                and result.compressed_answer.strip()
                and (result.self_review_metadata or {}).get("applied")
            )
        ]
        winner = next(
            (
                result
                for result in review_results
                if result.agent_id == self.previous_best_agent_id
            ),
            None,
        )
        if winner is None:
            return None
        self._last_winner_selection_trace = {
            "strategy": "all_agents_self_review_fallback",
            "status": "answerable",
            "selection_reason": "previous_best_agent_after_all_agents_review",
            "previous_best_agent_id": self.previous_best_agent_id,
            "selected_answer": winner.compressed_answer,
            "selected_agents": [winner.agent_id],
            "fallback_agents": [result.agent_id for result in fallback_results],
            "review_agents": [result.agent_id for result in review_results],
        }
        return winner

    def _select_winner_by_answer_clusters(
        self,
        stage1_results: list[AgentReasoningSummary],
        *,
        verifier_results: list[VerifierScoreByReasoning],
        evidence: dict[str, Any],
    ) -> AgentReasoningSummary | None:
        result_by_agent = {result.agent_id: result for result in stage1_results}
        active_results = [
            result
            for result in stage1_results
            if (
                result.active
                and result.winner_selection_eligible
                and result.compressed_answer.strip()
                and self.answer_validator.is_valid(result.compressed_answer)
            )
        ]
        if not active_results:
            self._last_winner_selection_trace = {
                "strategy": "clustered_self_consistency",
                "status": "no_active_valid_candidates",
                "clusters": [],
            }
            return None

        verifier_by_agent = {
            result.target_agent_id: float(result.verifier_score or 0.0)
            for result in verifier_results or []
        }
        evidence_text = self._winner_evidence_text(evidence)
        direct_tool_answers = self._direct_tool_answers(evidence)
        clusters: dict[str, dict[str, Any]] = {}

        for result in active_results:
            key = self._winner_answer_key(result.compressed_answer)
            if not key:
                continue
            cluster = clusters.setdefault(
                key,
                {
                    "cluster_key": key,
                    "answer": result.compressed_answer,
                    "agent_ids": [],
                    "candidates": [],
                    "evidence_supported": False,
                    "direct_tool_supported": False,
                    "max_reward": 0.0,
                    "max_confidence": 0.0,
                    "max_total_score": float("-inf"),
                },
            )
            config = next((agent for agent in self.agents if agent.agent_id == result.agent_id), None)
            reward = verifier_by_agent.get(
                result.agent_id,
                config.avg_verifier_score if config else 0.0,
            )
            total_score = config.total_score if config else float("-inf")
            evidence_supported = self._answer_supported_by_evidence(result.compressed_answer, evidence_text)
            direct_tool_supported = self._answer_matches_direct_tool(result.compressed_answer, direct_tool_answers)
            cluster["agent_ids"].append(result.agent_id)
            cluster["candidates"].append(
                {
                    "agent_id": result.agent_id,
                    "answer": result.compressed_answer,
                    "confidence": result.confidence_score,
                    "versa_reward": reward,
                    "total_score": total_score,
                    "evidence_supported": evidence_supported,
                    "direct_tool_supported": direct_tool_supported,
                }
            )
            cluster["evidence_supported"] = bool(cluster["evidence_supported"] or evidence_supported)
            cluster["direct_tool_supported"] = bool(cluster["direct_tool_supported"] or direct_tool_supported)
            cluster["max_reward"] = max(float(cluster["max_reward"]), float(reward or 0.0))
            cluster["max_confidence"] = max(float(cluster["max_confidence"]), float(result.confidence_score or 0.0))
            cluster["max_total_score"] = max(float(cluster["max_total_score"]), float(total_score))

        cluster_list = list(clusters.values())
        if not cluster_list:
            self._last_winner_selection_trace = {
                "strategy": "clustered_self_consistency",
                "status": "no_answer_clusters",
                "clusters": [],
            }
            return None

        selected_cluster, selection_reason = self._rank_answer_clusters(cluster_list)
        if not self._cluster_has_selection_signal(selected_cluster):
            self._last_winner_selection_trace = {
                "strategy": "clustered_self_consistency",
                "status": "no_cluster_signal_fallback_to_total_score",
                "clusters": [
                    self._cluster_metadata(cluster)
                    for cluster in sorted(
                        cluster_list,
                        key=lambda item: self._cluster_rank_tuple(item),
                        reverse=True,
                    )
                ],
            }
            return None
        selected_candidate = max(
            selected_cluster["candidates"],
            key=lambda item: (
                bool(item.get("direct_tool_supported")),
                bool(item.get("evidence_supported")),
                float(item.get("versa_reward") or 0.0),
                float(item.get("confidence") or 0.0),
                float(
                    item.get("total_score")
                    if item.get("total_score") is not None
                    else float("-inf")
                ),
            ),
        )
        winner = result_by_agent.get(str(selected_candidate.get("agent_id") or ""))
        self._last_winner_selection_trace = {
            "strategy": "clustered_self_consistency",
            "status": "answerable" if winner else "selected_agent_missing",
            "selection_reason": selection_reason,
            "selected_answer": selected_cluster.get("answer", ""),
            "selected_cluster_key": selected_cluster.get("cluster_key", ""),
            "selected_cluster_size": len(set(selected_cluster.get("agent_ids") or [])),
            "selected_agents": sorted(set(selected_cluster.get("agent_ids") or [])),
            "selected_candidate": selected_candidate,
            "clusters": [
                self._cluster_metadata(cluster)
                for cluster in sorted(
                    cluster_list,
                    key=lambda item: self._cluster_rank_tuple(item),
                    reverse=True,
                )
            ],
        }
        return winner

    def _rank_answer_clusters(self, clusters: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        selected = max(clusters, key=self._cluster_rank_tuple)
        vote_count = len(set(selected.get("agent_ids") or []))
        reason = "fallback_highest_cluster_rank"
        if selected.get("direct_tool_supported") and vote_count >= 2:
            reason = "direct_tool_supported_answer"
        elif vote_count >= 2 and selected.get("evidence_supported"):
            reason = "multi_agent_consensus_with_evidence_support"
        elif vote_count >= 2 and float(selected.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD:
            reason = "multi_agent_consensus_with_high_versa_reward"
        elif vote_count >= 2:
            reason = "multi_agent_consensus"
        elif selected.get("evidence_supported") and float(selected.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD:
            reason = "single_agent_evidence_supported_high_versa_reward"
        elif float(selected.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD:
            reason = "single_agent_high_versa_reward"
        return selected, reason

    def _cluster_rank_tuple(self, cluster: dict[str, Any]) -> tuple:
        vote_count = len(set(cluster.get("agent_ids") or []))
        return (
            bool(cluster.get("direct_tool_supported")) and vote_count >= 2,
            vote_count >= 2 and bool(cluster.get("evidence_supported")),
            vote_count >= 2 and float(cluster.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD,
            vote_count >= 2,
            bool(cluster.get("evidence_supported")),
            float(cluster.get("max_reward") or 0.0),
            vote_count,
            float(cluster.get("max_confidence") or 0.0),
            float(
                cluster.get("max_total_score")
                if cluster.get("max_total_score") is not None
                else float("-inf")
            ),
        )

    def _cluster_has_selection_signal(self, cluster: dict[str, Any]) -> bool:
        vote_count = len(set(cluster.get("agent_ids") or []))
        return bool(
            vote_count >= 2
            or float(cluster.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD
            or (
                cluster.get("evidence_supported")
                and float(cluster.get("max_reward") or 0.0) >= self.EARLY_STOP_VERIFIER_THRESHOLD
            )
        )

    def _cluster_metadata(self, cluster: dict[str, Any]) -> dict[str, Any]:
        return {
            "cluster_key": cluster.get("cluster_key", ""),
            "answer": cluster.get("answer", ""),
            "agent_ids": sorted(set(cluster.get("agent_ids") or [])),
            "agent_count": len(set(cluster.get("agent_ids") or [])),
            "evidence_supported": bool(cluster.get("evidence_supported")),
            "direct_tool_supported": bool(cluster.get("direct_tool_supported")),
            "max_reward": round(float(cluster.get("max_reward") or 0.0), 6),
            "max_confidence": round(float(cluster.get("max_confidence") or 0.0), 6),
            "candidate_count": len(cluster.get("candidates") or []),
        }

    def _winner_answer_key(self, answer: str) -> str:
        cleaned = self.answer_validator.clean(answer)
        if not cleaned or not self.answer_validator.is_valid(cleaned):
            return ""
        normalized = normalize_for_exact(cleaned).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        if "," in normalized:
            parts = [
                re.sub(r"\s+", " ", part.strip())
                for part in normalized.split(",")
                if part.strip()
            ]
            if len(parts) > 1:
                normalized = ",".join(sorted(parts))
        return normalized

    def _winner_evidence_text(self, evidence: dict[str, Any]) -> str:
        parts = [
            str(evidence.get("search_result") or ""),
            str(evidence.get("attachment_result") or ""),
            str(evidence.get("solver_result") or ""),
        ]
        return "\n".join(part for part in parts if part.strip())

    def _answer_supported_by_evidence(self, answer: str, evidence_text: str) -> bool:
        answer_key = self._winner_answer_key(answer)
        evidence_key = normalize_for_exact(evidence_text).strip().lower()
        if not answer_key or not evidence_key:
            return False
        if answer_key in evidence_key:
            return True
        if "," in answer_key:
            parts = [part.strip() for part in answer_key.split(",") if part.strip()]
            if parts:
                matched = sum(1 for part in parts if part and part in evidence_key)
                return matched >= max(1, (len(parts) + 1) // 2)
        return False

    def _direct_tool_answers(self, evidence: dict[str, Any]) -> list[str]:
        answers: list[str] = []
        for item in evidence.get("tool_usage", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("tool_name") != "deterministic_handler_router":
                continue
            if item.get("ok") and item.get("output_type") == "final_answer":
                raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
                trust = item.get("handler_trust") if isinstance(item.get("handler_trust"), dict) else {}
                answer = raw.get("answer") or trust.get("answer", "")
                if str(answer or "").strip():
                    answers.append(str(answer))
        return answers

    def _answer_matches_direct_tool(self, answer: str, direct_tool_answers: list[str]) -> bool:
        answer_key = self._winner_answer_key(answer)
        if not answer_key:
            return False
        return any(answer_key == self._winner_answer_key(tool_answer) for tool_answer in direct_tool_answers)

    def _winner_selection_metadata(
        self,
        stage1_results: list[AgentReasoningSummary],
        winner: AgentReasoningSummary | None,
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
        if winner is not None:
            status = "answerable"
        elif stage1_results and len(abstained) == len(stage1_results):
            status = "all_agents_abstained"
        elif stage1_results:
            status = "all_agents_abstained_or_invalid"
        else:
            status = "no_stage1_results"

        return {
            "status": status,
            "winner_agent_id": winner.agent_id if winner else "",
            "winner_answer": winner.compressed_answer if winner else "",
            "selection_trace": dict(self._last_winner_selection_trace or {}),
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
