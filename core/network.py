from __future__ import annotations

from threading import Lock
import time
from typing import Any

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    JudgeScoreByReasoning,
    NetworkSummary,
)
from core.evidence_runner import EvidenceRunner
from core.slm_agent import SLM_Agent
from core.stage1_runner import Stage1Runner
from core.stage2_runner import Stage2Runner
from score import AnswerValidator, PenaltyCalculator, ScoreCalculator
from utils.network_utils import normalize_for_exact


class Network:
    """
    主控一次多 Agent 推理任務，協調 evidence 準備、Stage1 候選答案生成、
    early-stop 判定、Stage2 cross-agent judging、分數計算與最終答案選擇。

    Args:
        - question: 使用者輸入的問題。
        - agents: 參與推理與評分的 AgentConfig 清單。
        - attachment: 題目附檔資訊或已解析內容。
        - tool_manager: 可執行 search、calculator 等工具的管理器。
        - stage1_runs_per_agent: 每個 Agent 在 Stage1 要重複 reasoning 的次數。
        - max_stage1_workers: Stage1 平行 worker 數量上限。
        - max_stage2_workers: Stage2 judge pair 平行 worker 數量上限。
        - stage2_max_tokens: Stage2 judge 單次回覆的最大 token 數。
        - enable_stage1_early_stop: 是否啟用 Stage1 early-stop。
        - enable_stage1_tool_use: 是否允許 Stage1 Agent 在 reasoning 中使用工具。
        - max_stage1_tool_turns: Stage1 tool-use 模式下每個 run 最多工具回合數。
        - previous_best_agent_id: 前一題表現最佳的 Agent id，用於 early-stop judge。
        - stage1_early_stop_max_retries: early-stop 條件不通過時最多重新執行 Stage1 的次數。
        - search_result: 外部預先提供的 search evidence。
        - attachment_result: 外部預先提供的 attachment evidence。

    Returns:
        - NetworkSummary: 包含 final answer、winner agent、Stage1 結果、Stage2 judge 結果、
          Agent 分數、response time、token usage 與 tool usage metadata。
    """

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
        enable_stage1_early_stop: bool = False,
        enable_stage1_tool_use: bool = False,
        max_stage1_tool_turns: int = 2,
        previous_best_agent_id: str | None = None,
        stage1_early_stop_max_retries: int = 1,
        enable_compact_search_evidence: bool = False,
        enable_evidence_driven_search: bool = True,
        search_result: str = "",
        attachment_result: str = "",
    ) -> None:
        self.question = question
        self.agents = agents
        self.attachment = attachment or {}
        self.tool_manager = tool_manager
        self.stage1_runs_per_agent = stage1_runs_per_agent
        self.max_stage1_workers = max_stage1_workers
        self.max_stage2_workers = max_stage2_workers
        self.stage2_max_tokens = stage2_max_tokens
        self.enable_stage1_early_stop = enable_stage1_early_stop
        self.enable_stage1_tool_use = enable_stage1_tool_use
        self.max_stage1_tool_turns = max(0, max_stage1_tool_turns)
        self.previous_best_agent_id = previous_best_agent_id
        self.stage1_early_stop_max_retries = max(0, stage1_early_stop_max_retries)
        self.enable_compact_search_evidence = enable_compact_search_evidence
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.search_result = search_result
        self.attachment_result = attachment_result

        self._slm_agents: dict[str, SLM_Agent] = {}
        self._slm_agents_lock = Lock()
        self._token_usage_lock = Lock()
        self._token_usage: dict[str, dict[str, int]] = {}

        self.score_calculator = ScoreCalculator()
        self.penalty_calculator = PenaltyCalculator()
        self.answer_validator = AnswerValidator()
        self.evidence_runner = EvidenceRunner(
            question=self.question,
            attachment=self.attachment,
            tool_manager=self.tool_manager,
            search_result=self.search_result,
            attachment_result=self.attachment_result,
            compact_search_evidence=self.enable_compact_search_evidence,
            enable_evidence_driven_search=self.enable_evidence_driven_search,
        )
        self.stage1_runner = Stage1Runner(
            question=self.question,
            agents=self.agents,
            get_agent=self._get_slm_agent,
            record_token_usage=self._record_token_usage,
            stage1_runs_per_agent=self.stage1_runs_per_agent,
            max_workers=self.max_stage1_workers,
            enable_tool_use=self.enable_stage1_tool_use,
            max_tool_turns=self.max_stage1_tool_turns,
            tool_manager=self.tool_manager,
        )
        self.stage2_runner = Stage2Runner(
            question=self.question,
            agents=self.agents,
            get_agent=self._get_slm_agent,
            record_token_usage=self._record_token_usage,
            max_workers=self.max_stage2_workers,
            max_tokens=self.stage2_max_tokens,
        )

    def run(self) -> NetworkSummary:
        """
        執行完整 Network 任務流程，包含 evidence、Stage1、early-stop、Stage2、
        penalty、score calculation 與 winner selection。

        Args:
            - 無。

        Returns:
            - NetworkSummary: 本次任務的最終答案、各階段結果、分數與 metadata。
        """
        response_started_at = time.perf_counter()
        self._reset_token_usage()
        evidence = self.evidence_runner.run()

        stage1_attempts = 0
        early_stop_reason = ""
        early_stop_judge_results: list[JudgeScoreByReasoning] = []
        direct_consensus_winner: AgentReasoningSummary | None = None
        direct_consensus_supporting_agents: list[str] = []
        while True:
            stage1_attempts += 1
            stage1_results = self.stage1_runner.run(evidence)
            (
                direct_consensus_winner,
                direct_consensus_supporting_agents,
            ) = self._confidence_one_answer_consensus(stage1_results)
            if direct_consensus_winner is not None:
                early_stop_winner = direct_consensus_winner
                early_stop_judge_results = []
                early_stop_reason = "cross_agent_confidence_1.0_answer_consensus"
            else:
                early_stop_winner, early_stop_judge_results, early_stop_reason = (
                    self._stage1_early_stop_decision(stage1_results)
                )
            should_retry_stage1 = (
                self.enable_stage1_early_stop
                and early_stop_reason == "confidence_0.67_judge_score_not_positive"
                and stage1_attempts <= self.stage1_early_stop_max_retries
            )
            if not should_retry_stage1:
                break

        active_results = [result for result in stage1_results if result.active]
        stage2_skipped = early_stop_winner is not None
        judge_results = (
            early_stop_judge_results
            if stage2_skipped
            else self.stage2_runner.run(active_results)
        )
        if direct_consensus_winner is not None:
            penalty_results = []
            self._write_direct_consensus_scores(stage1_results)
        else:
            penalty_results = self.penalty_calculator.calculate(
                stage1_results,
                question=self.question,
            )
            self.score_calculator.write_scores_to_agent_config(
                self.agents,
                stage1_results,
                judge_results,
                penalty_results,
            )
        winner = early_stop_winner or self._select_winner(stage1_results)
        response_time_seconds = time.perf_counter() - response_started_at

        return NetworkSummary(
            question=self.question,
            final_answer=winner.compressed_answer if winner else "",
            winner_agent_id=winner.agent_id if winner else "",
            stage1_results=stage1_results,
            judge_results=judge_results,
            agent_scores=self.agents,
            metadata={
                "stage1_runs_per_agent": self.stage1_runs_per_agent,
                "response_time_seconds": response_time_seconds,
                "response_time_ms": round(response_time_seconds * 1000, 3),
                "token_usage": self._token_usage_snapshot(),
                "max_stage1_workers": self.stage1_runner.worker_count(),
                "max_stage2_workers": self.stage2_runner.worker_count(active_results),
                "stage2_max_tokens": self.stage2_max_tokens,
                "enable_stage1_tool_use": self.enable_stage1_tool_use,
                "enable_compact_search_evidence": self.enable_compact_search_evidence,
                "query_planner": "signal",
                "enable_evidence_driven_search": self.enable_evidence_driven_search,
                "max_stage1_tool_turns": self.max_stage1_tool_turns,
                "enable_stage1_early_stop": self.enable_stage1_early_stop,
                "previous_best_agent_id": self.previous_best_agent_id or "",
                "stage1_early_stop_max_retries": self.stage1_early_stop_max_retries,
                "stage1_attempts": stage1_attempts,
                "stage1_early_stop": stage2_skipped,
                "stage1_early_stop_reason": early_stop_reason if stage2_skipped else "",
                "stage2_skipped": stage2_skipped,
                "cross_agent_consensus_used": direct_consensus_winner is not None,
                "cross_agent_consensus_supporting_agents": direct_consensus_supporting_agents,
                "cross_agent_consensus_answer": (
                    direct_consensus_winner.compressed_answer
                    if direct_consensus_winner is not None
                    else ""
                ),
                "active_agent_count": len(active_results),
                "search_used": bool(evidence["search_result"].strip()),
                "attachment_used": bool(evidence["attachment_result"].strip()),
                "solver_used": bool(evidence["solver_result"].strip()),
                "routing": evidence.get("routing", {}),
                "tool_usage": evidence.get("tool_usage", []),
                "penalty_results": penalty_results,
            },
        )

    def _confidence_one_answer_consensus(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> tuple[AgentReasoningSummary | None, list[str]]:
        """
        找出多個 Agent 同時達到 confidence=1.0 且 normalized answer 相同的跨 Agent 共識。
        Args:
            - stage1_results: Stage1Runner 回傳的每個 Agent 推理摘要。
        Returns:
            - AgentReasoningSummary | None: 若存在共識，回傳代表該答案的 winner。
            - list[str]: 支持該共識答案的 Agent id 清單。
        """
        confident_results = [
            result
            for result in stage1_results
            if (
                result.active
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
        判斷兩個跨 Agent 候選答案是否為相同答案。
        Args:
            - answer_a: 第一個候選答案。
            - answer_b: 第二個候選答案。
        Returns:
            - bool: 兩個答案經 exact normalization 後是否相同。
        """
        return normalize_for_exact(answer_a) == normalize_for_exact(answer_b)

    def _write_direct_consensus_scores(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> None:
        """
        跨 Agent 共識直接輸出時，重設 AgentConfig 的非必要評分欄位。
        Args:
            - stage1_results: Stage1Runner 回傳的每個 Agent 推理摘要。
        Returns:
            - None。
        """
        result_by_agent = {result.agent_id: result for result in stage1_results}
        for config in self.agents:
            result = result_by_agent.get(config.agent_id)
            config.confidence_score = result.confidence_score if result else 0.0
            config.judge_scores = []
            config.avg_judge_score = 0.0
            config.penalty_score = 0.0
            config.penalty_reasons = []
            config.total_score = (
                config.confidence_score
                if result is not None and result.active
                else float("-inf")
            )

    def _stage1_early_stop_decision(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> tuple[AgentReasoningSummary | None, list[JudgeScoreByReasoning], str]:
        """
        根據 Stage1 confidence 與 previous-best judge 結果判斷是否提前停止。

        Args:
            - stage1_results: Stage1Runner 產生的各 Agent 候選結果。

        Returns:
            - AgentReasoningSummary | None: 若 early-stop 成立，回傳勝出候選。
            - list[JudgeScoreByReasoning]: early-stop 過程產生的 judge 結果。
            - str: early-stop 判斷原因。
        """
        if not self.enable_stage1_early_stop:
            return None, [], ""

        active_results = [
            result
            for result in stage1_results
            if result.active and result.compressed_answer.strip()
        ]
        if not active_results:
            return None, [], "no_active_stage1_result"

        confident_results = [
            result
            for result in active_results
            if result.confidence_score >= 1.0
        ]
        if confident_results:
            return confident_results[0], [], "confidence_1.0"

        max_confidence = max(result.confidence_score for result in active_results)
        if max_confidence != 0.67:
            return None, [], "max_confidence_not_0.67"

        candidates = [
            result
            for result in active_results
            if result.confidence_score == max_confidence
        ]
        judge_config = self._early_stop_judge_config(candidates)
        if judge_config is None:
            return None, [], "no_early_stop_judge_agent"

        judge_results = [
            self.stage2_runner.judge_reasoning(judge_config, candidate)
            for candidate in candidates
        ]
        positive_results = [
            result for result in judge_results if result.judge_score > 0
        ]
        if not positive_results:
            return None, judge_results, "confidence_0.67_judge_score_not_positive"

        best_judge_result = max(positive_results, key=lambda result: result.judge_score)
        winner = next(
            result
            for result in candidates
            if result.agent_id == best_judge_result.target_agent_id
        )
        return winner, judge_results, "confidence_0.67_positive_previous_best_judge"

    def _early_stop_judge_config(
        self,
        candidates: list[AgentReasoningSummary],
    ) -> AgentConfig | None:
        """
        選擇 early-stop 模式下用來評分候選答案的 judge agent。

        Args:
            - candidates: confidence score 最高的 Stage1 候選結果。

        Returns:
            - AgentConfig | None: 可用的 judge agent 設定；若沒有 agent 則回傳 None。
        """
        if self.previous_best_agent_id:
            for config in self.agents:
                if config.agent_id == self.previous_best_agent_id:
                    return config

        candidate_ids = {candidate.agent_id for candidate in candidates}
        for config in self.agents:
            if config.agent_id not in candidate_ids:
                return config
        return self.agents[0] if self.agents else None

    def _select_winner(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> AgentReasoningSummary | None:
        """
        根據 AgentConfig 中的 total_score、confidence_score 與 avg_judge_score 選出最終 winner。

        Args:
            - stage1_results: Stage1Runner 產生的各 Agent 候選結果。

        Returns:
            - AgentReasoningSummary | None: 最終勝出的候選結果；若沒有 active agent 則回傳 None。
        """
        result_by_agent = {result.agent_id: result for result in stage1_results}
        active_agents = [
            config
            for config in self.agents
            if result_by_agent.get(config.agent_id) and result_by_agent[config.agent_id].active
        ]
        if not active_agents:
            return None
        winner_config = max(
            active_agents,
            key=lambda config: (
                config.total_score,
                config.confidence_score,
                config.avg_judge_score,
            ),
        )
        return result_by_agent[winner_config.agent_id]

    def _get_slm_agent(self, config: AgentConfig) -> SLM_Agent:
        """
        從任務內快取取得 SLM_Agent，若尚未建立則依 AgentConfig 建立。

        Args:
            - config: 指定 agent_id、model_name 與 temperature 的 AgentConfig。

        Returns:
            - SLM_Agent: 可重複使用的模型呼叫物件。
        """
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
        重置本次任務的 Stage1、Stage2 與 total token usage 統計。

        Args:
            - 無。

        Returns:
            - None。
        """
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
        累加指定階段的 prompt token、completion token 與 total token。

        Args:
            - stage: token usage 所屬階段，例如 stage1 或 stage2。
            - prompt_tokens: 本次呼叫消耗的 prompt token 數。
            - completion_tokens: 本次呼叫產生的 completion token 數。

        Returns:
            - None。
        """
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
        建立目前 token usage 的安全快照，供 NetworkSummary metadata 使用。

        Args:
            - 無。

        Returns:
            - dict[str, dict[str, int]]: 各階段與 total 的 token 統計。
        """
        with self._token_usage_lock:
            return {
                stage: dict(values)
                for stage, values in self._token_usage.items()
            }


__all__ = ["Network"]
