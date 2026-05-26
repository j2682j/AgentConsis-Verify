from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from context.stage2_context import Stage2ContextBuilder
from core.config import AgentConfig, AgentReasoningSummary, JudgeScoreByReasoning
from core.slm_agent import SLM_Agent
from parsers import Stage2JudgeParser
from parsers.reasoning_parser import format_reasoning_steps


class Stage2Runner:
    """
    執行 Stage2 cross-agent judging，讓 active agents 彼此評分候選 reasoning steps。

    Args:
        - question: 使用者輸入的問題。
        - agents: 可作為 judge 的 AgentConfig 清單。
        - get_agent: 根據 AgentConfig 取得或建立 SLM_Agent 的函式。
        - record_token_usage: 紀錄 Stage2 prompt_tokens 與 completion_tokens 的 callback。
        - max_workers: Stage2 judge pair 平行 worker 數量上限。
        - max_tokens: 單次 judge 回覆的最大 token 數。
        - context_builder: 建立 Stage2 judge prompt 的 Stage2ContextBuilder。
        - judge_parser: 解析 judge 回覆 step scores 的 Stage2JudgeParser。

    Returns:
        - list[JudgeScoreByReasoning]: 每個 judge-target pair 的 step scores 與平均 judge score。
        - []: 沒有 active judge pair 時回傳空清單。
    """

    def __init__(
        self,
        *,
        question: str,
        agents: list[AgentConfig],
        get_agent: Callable[[AgentConfig], SLM_Agent],
        record_token_usage: Callable[..., None],
        max_workers: int | None = None,
        max_tokens: int = 512,
        context_builder: Stage2ContextBuilder | None = None,
        judge_parser: Stage2JudgeParser | None = None,
    ) -> None:
        self.question = question
        self.agents = agents
        self.get_agent = get_agent
        self.record_token_usage = record_token_usage
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.context_builder = context_builder or Stage2ContextBuilder()
        self.judge_parser = judge_parser or Stage2JudgeParser()

    def run(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> list[JudgeScoreByReasoning]:
        """
        平行執行所有 active target 與 active judge 的 Stage2 評分配對。

        Args:
            - stage1_results: Stage1Runner 產生且已聚合的候選結果。

        Returns:
            - list[JudgeScoreByReasoning]: 依 target-judge pair 排列的評分結果。
            - []: 沒有可評分配對時回傳空清單。
        """
        pairs = self._judge_pairs(stage1_results)
        if not pairs:
            return []

        results_by_pair: dict[tuple[str, str], JudgeScoreByReasoning] = {}
        with ThreadPoolExecutor(max_workers=self.worker_count(stage1_results)) as executor:
            future_to_pair = {
                executor.submit(self.judge_reasoning, judge_config, target): (target, judge_config)
                for target, judge_config in pairs
            }
            for future in as_completed(future_to_pair):
                target, judge_config = future_to_pair[future]
                try:
                    result = future.result()
                except Exception:
                    result = JudgeScoreByReasoning(
                        judge_agent_id=judge_config.agent_id,
                        target_agent_id=target.agent_id,
                        judge_score=0.0,
                        step_scores=[],
                        raw_reply="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    )
                results_by_pair[(target.agent_id, judge_config.agent_id)] = result

        return [
            results_by_pair[(target.agent_id, judge_config.agent_id)]
            for target, judge_config in pairs
            if (target.agent_id, judge_config.agent_id) in results_by_pair
        ]

    def judge_reasoning(
        self,
        judge_config: AgentConfig,
        target: AgentReasoningSummary,
    ) -> JudgeScoreByReasoning:
        """
        使用單一 judge agent 對單一 target agent 的 compressed reasoning 進行 step scoring。

        Args:
            - judge_config: 負責評分的 AgentConfig。
            - target: 被評分的 AgentReasoningSummary。

        Returns:
            - JudgeScoreByReasoning: 此 judge-target pair 的 step scores、平均分數與 token usage。
        """
        messages = self._build_messages(target)
        raw_reply = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            raw_reply, prompt_tokens, completion_tokens = self.get_agent(judge_config).invoke_with_usage(
                messages,
                max_tokens=self.max_tokens,
            )
            self.record_token_usage(
                stage="stage2",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            step_scores, judge_score = self.judge_parser.parse(
                raw_reply,
                target.compressed_reasoning,
            )
        except Exception:
            step_scores = []
            judge_score = 0.0

        return JudgeScoreByReasoning(
            judge_agent_id=judge_config.agent_id,
            target_agent_id=target.agent_id,
            judge_score=judge_score,
            step_scores=step_scores,
            raw_reply=raw_reply,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def worker_count(self, stage1_results: list[AgentReasoningSummary]) -> int:
        """
        計算 Stage2 實際使用的平行 worker 數量。

        Args:
            - stage1_results: Stage1Runner 產生的候選結果，用於計算 active pair 數。

        Returns:
            - int: 依 active pair 數與 max_workers 限制後的 worker 數。
        """
        active_count = sum(1 for result in stage1_results if result.active)
        total_pairs = max(1, active_count * max(0, active_count - 1))
        if self.max_workers is None:
            return total_pairs
        return max(1, min(self.max_workers, total_pairs))

    def _judge_pairs(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> list[tuple[AgentReasoningSummary, AgentConfig]]:
        """
        建立 Stage2 judge-target 配對，排除 agent 自評自己的候選結果。

        Args:
            - stage1_results: Stage1Runner 產生的候選結果。

        Returns:
            - list[tuple[AgentReasoningSummary, AgentConfig]]: 待執行的 target 與 judge 配對。
        """
        active_agent_ids = {result.agent_id for result in stage1_results if result.active}
        return [
            (target, judge_config)
            for target in stage1_results
            for judge_config in self.agents
            if judge_config.agent_id != target.agent_id
            and judge_config.agent_id in active_agent_ids
        ]

    def _build_messages(self, target: AgentReasoningSummary) -> list[dict[str, str]]:
        """
        建立傳給 judge agent 的 Stage2 prompt messages。

        Args:
            - target: 被評分的 AgentReasoningSummary。

        Returns:
            - list[dict[str, str]]: OpenAI-compatible chat messages。
        """
        target_tool_evidence = self._build_target_tool_evidence(target)

        return self.context_builder.build(
            question=self.question,
            target_answer=target.compressed_answer,
            target_reasoning=format_reasoning_steps(target.compressed_reasoning or "None"),
            target_tool_evidence=target_tool_evidence,
        )

    def _build_target_tool_evidence(self, target: AgentReasoningSummary) -> list[dict]:
        """
        從 target runs 中整理 Stage1 tool calls 與 tool results，供 Stage2 judge 評分使用。

        Args:
            - target: 被評分的 AgentReasoningSummary。

        Returns:
            - list[dict]: 每次工具使用的 run_index、tool_name、tool_args、reasoning_step 與 result_summary。
        """
        evidence: list[dict] = []

        for run in target.runs:
            if not run.parse_completed:
                continue

            for index, tool_call in enumerate(run.tool_calls):
                tool_result = (
                    run.tool_results[index]
                    if index < len(run.tool_results)
                    else {}
                )

                evidence.append(
                    {
                        "run_index": run.run_index,
                        "question": self.question,
                        "tool_name": tool_call.get("tool_name", ""),
                        "tool_args": tool_call.get("tool_args", {}),
                        "reasoning_step": tool_call.get("reasoning_step", ""),
                        "cache_hit": bool(tool_result.get("cache_hit", False)),
                        "result_summary": str(tool_result.get("output_text", "") or "")[:2000],
                    }
                )

        return evidence


__all__ = ["Stage2Runner"]
