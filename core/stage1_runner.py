from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from context.context_builder import ContextPacket
from context.stage1_context import Stage1ContextBuilder
from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply
from core.slm_agent import SLM_Agent
from core.stage1_trajectory_runner import Stage1TrajectoryRunner
from parsers import Stage1ReplyParser
from score import Stage1Aggregator


class Stage1Runner:
    """
    執行 Stage1 多 Agent 自我一致性推理，將每個 Agent 重複執行多次，
    並把每次回覆聚合成每個 Agent 的候選答案、壓縮 reasoning 與 confidence score。

    Args:
        - question: 使用者輸入的問題。
        - agents: 參與 Stage1 推理的 AgentConfig 清單。
        - get_agent: 根據 AgentConfig 取得或建立 SLM_Agent 的函式。
        - record_token_usage: 紀錄 prompt_tokens 與 completion_tokens 的 callback。
        - stage1_runs_per_agent: 每個 Agent 要重複推理的次數。
        - max_workers: Stage1 平行執行 worker 數量上限。
        - enable_tool_use: 是否啟用 Stage1 tool-use trajectory。
        - max_tool_turns: tool-use 模式下每個 run 最多工具回合數。
        - tool_manager: tool-use 模式下執行 search、calculator 等工具的管理器。
        - context_builder: 一般 Stage1 prompt/context builder。
        - parser: 解析一般 Stage1 回覆的 Stage1ReplyParser。
        - aggregator: 聚合多次 Stage1 回覆並計算 confidence score 的 Stage1Aggregator。
        - trajectory_runner: tool-use 模式下執行單次 Agent trajectory 的 runner。

    Returns:
        - list[AgentReasoningSummary]: 每個 Agent 的 Stage1 聚合結果，包含 runs、
          compressed_answer、compressed_reasoning、confidence_score 與 active 狀態。
        - []: 當沒有任何 Agent 或 Stage1 task 時回傳空清單。
    """

    def __init__(
        self,
        *,
        question: str,
        agents: list[AgentConfig],
        get_agent: Callable[[AgentConfig], SLM_Agent],
        record_token_usage: Callable[..., None],
        stage1_runs_per_agent: int = 3,
        max_workers: int | None = None,
        enable_tool_use: bool = False,
        max_tool_turns: int = 2,
        tool_manager: Any | None = None,
        context_builder: Stage1ContextBuilder | None = None,
        parser: Stage1ReplyParser | None = None,
        aggregator: Stage1Aggregator | None = None,
        trajectory_runner: Stage1TrajectoryRunner | None = None,
    ) -> None:
        self.question = question
        self.agents = agents
        self.get_agent = get_agent
        self.record_token_usage = record_token_usage
        self.stage1_runs_per_agent = stage1_runs_per_agent
        self.max_workers = max_workers
        self.enable_tool_use = enable_tool_use
        self.max_tool_turns = max(0, max_tool_turns)
        self.context_builder = context_builder or Stage1ContextBuilder()
        self.parser = parser or Stage1ReplyParser()
        self.aggregator = aggregator or Stage1Aggregator()
        self.trajectory_runner = trajectory_runner or Stage1TrajectoryRunner(
            tool_manager=tool_manager,
            max_tool_turns=self.max_tool_turns,
        )

    def run(self, evidence: dict[str, Any]) -> list[AgentReasoningSummary]:
        """
        平行執行所有 Agent 的 Stage1 runs，並將同一 Agent 的多次結果聚合。

        Args:
            - evidence: EvidenceRunner 產生的 search、attachment、solver evidence。

        Returns:
            - list[AgentReasoningSummary]: 各 Agent 的聚合候選答案與 confidence score。
            - []: 沒有可執行 task 時回傳空清單。
        """
        runs_by_agent: dict[str, list[EachAgentReply]] = {
            config.agent_id: [] for config in self.agents
        }
        if not self.agents or self.stage1_runs_per_agent <= 0:
            return []

        for run_index in range(1, self.stage1_runs_per_agent + 1):
            with ThreadPoolExecutor(max_workers=self.worker_count()) as executor:
                future_to_config = {
                    executor.submit(self._run_single_agent, config, run_index, evidence): config
                    for config in self.agents
                }
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        reply = future.result()
                    except Exception as exc:
                        reply = EachAgentReply(
                            agent_id=config.agent_id,
                            model_name=config.model_name,
                            run_index=run_index,
                            raw_reply=f"[stage1_error] {type(exc).__name__}: {exc}",
                            reasoning="",
                            final_answer="",
                            parse_completed=False,
                            tool_context=self.format_tool_context(evidence),
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                        )
                    runs_by_agent[config.agent_id].append(reply)

        results: list[AgentReasoningSummary] = []
        for config in self.agents:
            runs = sorted(runs_by_agent.get(config.agent_id, []), key=lambda run: run.run_index)
            results.append(self.aggregator.summarize(config, runs))
        return results

    def worker_count(self) -> int:
        """
        計算 Stage1 實際使用的平行 worker 數量。

        Args:
            - 無。

        Returns:
            - int: 依 Agent 數、run 數與 max_workers 限制後的 worker 數。
        """
        total_runs = max(1, len(self.agents))
        if self.max_workers is None:
            return total_runs
        return max(1, min(self.max_workers, total_runs))

    def evidence_to_context_packets(self, evidence: dict[str, Any]) -> list[ContextPacket]:
        """
        將 evidence dict 轉成 Stage1ContextBuilder 可使用的 ContextPacket 清單。

        Args:
            - evidence: 包含 solver_result、attachment_result、search_result 的 evidence dict。

        Returns:
            - list[ContextPacket]: 依優先權標記的 evidence packets。
        """
        packets: list[ContextPacket] = []
        if evidence.get("solver_result"):
            packets.append(
                ContextPacket(
                    packet_type="solver_result",
                    content=evidence["solver_result"],
                    priority=90,
                    metadata={"source": "deterministic_solver"},
                )
            )
        if evidence.get("attachment_result"):
            packets.append(
                ContextPacket(
                    packet_type="attachment_result",
                    content=evidence["attachment_result"],
                    priority=80,
                    metadata={"source": "attachment_reader"},
                )
            )
        if evidence.get("search_result"):
            packets.append(
                ContextPacket(
                    packet_type="search_result",
                    content=evidence["search_result"],
                    priority=70,
                    metadata={"source": "search"},
                )
            )
        return packets

    def format_tool_context(self, evidence: dict[str, Any]) -> str:
        """
        將 evidence dict 格式化成寫入 EachAgentReply.tool_context 的文字。

        Args:
            - evidence: 包含 search、attachment、solver 結果的 evidence dict。

        Returns:
            - str: 合併後的工具上下文文字。
        """
        parts = []
        if evidence["search_result"]:
            parts.append("Search_Result:\n" + evidence["search_result"])
        if evidence["attachment_result"]:
            parts.append("Attachment_Result:\n" + evidence["attachment_result"])
        if evidence["solver_result"]:
            parts.append("Solver_Result:\n" + evidence["solver_result"])
        return "\n\n".join(parts)

    def _run_single_agent(
        self,
        config: AgentConfig,
        run_index: int,
        evidence: dict[str, Any],
    ) -> EachAgentReply:
        """
        執行單一 Agent 的一次 Stage1 run，依設定選擇一般模式或 tool-use 模式。

        Args:
            - config: 目前執行的 AgentConfig。
            - run_index: 此 Agent 的第幾次 Stage1 run。
            - evidence: EvidenceRunner 產生的共享 evidence。

        Returns:
            - EachAgentReply: 單次 run 的 raw reply、reasoning、final answer、token 與 tool trace。
        """
        if self.enable_tool_use:
            reply, prompt_tokens, completion_tokens = self.trajectory_runner.run(
                config=config,
                agent=self.get_agent(config),
                question=self.question,
                evidence_packets=self.evidence_to_context_packets(evidence),
                run_index=run_index,
            )
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return reply

        messages = self.context_builder.build(
            question=self.question,
            evidence_packets=self.evidence_to_context_packets(evidence),
        )
        raw_reply = ""
        reasoning = ""
        final_answer = ""
        parse_completed = False
        prompt_tokens = 0
        completion_tokens = 0
        try:
            raw_reply, prompt_tokens, completion_tokens = self.get_agent(config).invoke_with_usage(messages)
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            parsed = self.parser.parse(raw_reply, expected_weight_count=0)
            reasoning = str(parsed.get("reasoning", "")).strip()
            final_answer = str(parsed.get("final_answer", "")).strip()
            parse_completed = bool(final_answer)
        except Exception as exc:
            raw_reply = raw_reply or f"[stage1_error] {type(exc).__name__}: {exc}"

        return EachAgentReply(
            agent_id=config.agent_id,
            model_name=config.model_name,
            run_index=run_index,
            raw_reply=raw_reply,
            reasoning=reasoning,
            final_answer=final_answer,
            parse_completed=parse_completed,
            tool_context=self.format_tool_context(evidence),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


__all__ = ["Stage1Runner"]
