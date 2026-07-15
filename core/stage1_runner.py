from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from context.context_builder import ContextPacket
from context.stage1_context import Stage1ContextBuilder
from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply
from core.slm_agent import SLM_Agent
from core.stage1_tool_use_runner import Stage1ToolUseRunner
from parsers import SelfReviewParser, Stage1ReplyParser
from score import AgentAnswerAggregator, Stage1Aggregator


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
        attachment: dict[str, Any] | None = None,
        stage1_runs_per_agent: int = 3,
        max_workers: int | None = None,
        enable_tool_use: bool = False,
        max_tool_turns: int = 2,
        tool_manager: Any | None = None,
        context_builder: Stage1ContextBuilder | None = None,
        parser: Stage1ReplyParser | None = None,
        aggregator: Stage1Aggregator | None = None,
        answer_aggregator: AgentAnswerAggregator | None = None,
        self_review_parser: SelfReviewParser | None = None,
        trajectory_runner: Stage1ToolUseRunner | None = None,
        unload_previous_slm_on_switch: bool = True,
    ) -> None:
        self.question = question
        self.agents = agents
        self.get_agent = get_agent
        self.record_token_usage = record_token_usage
        self.attachment = attachment or {}
        self.stage1_runs_per_agent = stage1_runs_per_agent
        self.max_workers = max_workers
        self.enable_tool_use = enable_tool_use
        self.max_tool_turns = max(0, max_tool_turns)
        self.context_builder = context_builder or Stage1ContextBuilder()
        self.parser = parser or Stage1ReplyParser()
        self.aggregator = aggregator or Stage1Aggregator()
        self.answer_aggregator = answer_aggregator or AgentAnswerAggregator()
        self.self_review_parser = self_review_parser or SelfReviewParser()
        self.trajectory_runner = trajectory_runner or Stage1ToolUseRunner(
            tool_manager=tool_manager,
            max_tool_turns=self.max_tool_turns,
        )
        self.tool_manager = tool_manager
        self.unload_previous_slm_on_switch = bool(unload_previous_slm_on_switch)
        self.model_lifecycle_records: list[dict[str, Any]] = []
        self.model_switch_stop_records = self.model_lifecycle_records

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

        self.model_lifecycle_records = []
        self.model_switch_stop_records = self.model_lifecycle_records
        if self.unload_previous_slm_on_switch:
            return self._run_sequential_with_model_switch_unload(evidence)

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
                        context_budget={},
                    )
                    runs_by_agent[config.agent_id].append(reply)

        results: list[AgentReasoningSummary] = []
        for config in self.agents:
            runs = sorted(runs_by_agent.get(config.agent_id, []), key=lambda run: run.run_index)
            results.append(self._summarize_with_aggregation(config, runs))
        return self._apply_self_review_fallback_if_needed(results, runs_by_agent)

    def _run_sequential_with_model_switch_unload(
        self,
        evidence: dict[str, Any],
    ) -> list[AgentReasoningSummary]:
        """
        以單 worker 序列方式執行 Stage1，並在切換到下一個 SLM 前卸載上一個 SLM。

        Args:
            - evidence: EvidenceRunner 輸出的 search、attachment、solver evidence。

        Returns:
            - list[AgentReasoningSummary]: 每個 Agent 的 Stage1 聚合結果。
        """
        runs_by_agent: dict[str, list[EachAgentReply]] = {
            config.agent_id: [] for config in self.agents
        }
        for config in self.agents:
            for run_index in range(1, self.stage1_runs_per_agent + 1):
                unload_after_call = (
                    not self.enable_tool_use
                    and run_index == self.stage1_runs_per_agent
                )
                runs_by_agent[config.agent_id].append(
                    self._safe_run_single_agent(
                        config,
                        run_index,
                        evidence,
                        unload_after_call=unload_after_call,
                    )
                )
            if self.enable_tool_use:
                self._unload_agent_model(
                    config=config,
                    phase="after_agent_stage1_tool_use_complete",
                )
            else:
                self.model_lifecycle_records.append(
                    {
                        "phase": "after_agent_stage1_complete",
                        "agent_id": config.agent_id,
                        "model": config.model_name,
                        "unload_method": "keep_alive_0",
                        "unloaded": True,
                        "trigger": "last_stage1_call",
                        "warning": "",
                    }
                )

        results: list[AgentReasoningSummary] = []
        for config in self.agents:
            runs = sorted(runs_by_agent.get(config.agent_id, []), key=lambda run: run.run_index)
            results.append(self._summarize_with_aggregation(config, runs))
        return self._apply_self_review_fallback_if_needed(results, runs_by_agent)

    def _summarize_with_aggregation(
        self,
        config: AgentConfig,
        runs: list[EachAgentReply],
    ) -> AgentReasoningSummary:
        """
        做單一 Agent 內部答案聚合，但不立刻觸發 self-review。

        Args:
         - config: 目前 Agent 設定。
         - runs: 該 Agent 的 Stage1 多次回答。

        Returns:
         - AgentReasoningSummary: 已寫入 aggregation metadata 的 summary。
        """
        summary = self.aggregator.summarize(config, runs)
        aggregation = self.answer_aggregator.aggregate(runs)
        summary.aggregation_metadata = aggregation.to_dict()
        if aggregation.answer and aggregation.status == "needs_review":
            summary.compressed_answer = aggregation.answer
            summary.confidence_score = aggregation.confidence_score
            config.confidence_score = aggregation.confidence_score
        return summary

    def _apply_self_review_fallback_if_needed(
        self,
        summaries: list[AgentReasoningSummary],
        runs_by_agent: dict[str, list[EachAgentReply]],
    ) -> list[AgentReasoningSummary]:
        """
        只有當所有 Agent 都無法形成 2/3 或 3/3 聚合答案時，才啟用 self-review fallback。

        Args:
         - summaries: 每個 Agent 的內部聚合結果。
         - runs_by_agent: 每個 Agent 的原始 Stage1 runs。

        Returns:
         - list[AgentReasoningSummary]: 必要時已套用 review answer 的 summaries。
        """
        if not summaries:
            return summaries
        if any(not self._summary_needs_review_fallback(summary) for summary in summaries):
            for summary in summaries:
                summary.self_review_metadata = {
                    "applied": False,
                    "skipped": True,
                    "skip_reason": "at_least_one_agent_has_aggregate_answer",
                }
            return summaries

        config_by_agent = {config.agent_id: config for config in self.agents}
        for summary in summaries:
            config = config_by_agent.get(summary.agent_id)
            if config is None:
                continue
            aggregation = summary.aggregation_metadata or {}
            review = self._run_self_review(
                config=config,
                runs=sorted(
                    runs_by_agent.get(summary.agent_id, []),
                    key=lambda run: run.run_index,
                ),
                aggregation=aggregation,
            )
            self._unload_agent_model(
                config=config,
                phase="after_agent_self_review_complete",
            )
            review["fallback_scope"] = "all_agents_need_review"
            summary.self_review_metadata = review
            if review.get("applied") and review.get("answer"):
                answer = str(review.get("answer") or "").strip()
                summary.compressed_answer = answer
                summary.compressed_reasoning = (
                    "step 1. Review the three previous final answers because no Agent "
                    "formed an internal majority.\n"
                    "step 2. Select the best short final answer after self-review."
                )
                summary.confidence_score = 0.67
                summary.active = True
                summary.winner_selection_eligible = True
                summary.winner_selection_status = "answerable"
                config.confidence_score = 0.67
        return summaries

    def _summary_needs_review_fallback(self, summary: AgentReasoningSummary) -> bool:
        aggregation = summary.aggregation_metadata or {}
        if bool(aggregation.get("needs_review")):
            return True
        status = str(aggregation.get("status", "") or "").strip()
        return status in {"needs_review", "no_valid_answers"}

    def _run_self_review(
        self,
        *,
        config: AgentConfig,
        runs: list[EachAgentReply],
        aggregation: Any,
    ) -> dict[str, Any]:
        answers = [str(run.final_answer or "").strip() for run in runs[:3]]
        while len(answers) < 3:
            answers.append("")
        trigger = (
            aggregation.get("reason", "")
            if isinstance(aggregation, dict)
            else getattr(aggregation, "reason", "")
        )
        metadata: dict[str, Any] = {
            "applied": False,
            "trigger": trigger,
            "previous_answers": list(answers),
            "tool_used": False,
            "tool_calls": [],
            "tool_results": [],
            "raw_reply": "",
            "final_raw_reply": "",
            "answer": "",
            "parse_completed": False,
            "error": "",
        }
        messages = self._build_self_review_messages(answers)
        try:
            raw_reply, prompt_tokens, completion_tokens = self.get_agent(config).invoke_with_usage(messages)
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            metadata["error"] = f"self_review_call_failed:{type(exc).__name__}: {exc}"
            return metadata

        metadata["raw_reply"] = raw_reply
        parsed = self.self_review_parser.parse(raw_reply)
        metadata["initial_parse"] = parsed.to_dict()

        if parsed.reply_type == "final_answer" and parsed.answer:
            metadata.update(
                {
                    "applied": True,
                    "answer": parsed.answer,
                    "parse_completed": bool(parsed.parse_completed),
                    "validator_passed": bool(parsed.parse_completed),
                    "error": parsed.error,
                }
            )
            return metadata

        if parsed.reply_type != "tool_request" or not parsed.parse_completed:
            metadata["error"] = parsed.error or "self_review_parse_failed"
            return metadata

        if self.tool_manager is None:
            metadata["error"] = "tool_manager_unavailable"
            return metadata

        tool_name = self._normalize_review_tool_name(parsed.tool_name)
        tool_args = self._normalize_review_tool_args(tool_name, parsed.tool_args)
        tool_call = {"tool_name": tool_name, "tool_args": tool_args}
        metadata["tool_calls"].append(tool_call)
        tool_result = self.tool_manager.execute_tool(
            tool_name,
            tool_args,
            agent_id=config.agent_id,
            stage="stage1_self_review",
        )
        metadata["tool_used"] = True
        metadata["tool_results"].append(tool_result)

        final_messages = self._build_self_review_after_tool_messages(
            answers=answers,
            tool_result=tool_result,
        )
        try:
            final_raw, prompt_tokens, completion_tokens = self.get_agent(config).invoke_with_usage(final_messages)
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            metadata["error"] = f"self_review_final_call_failed:{type(exc).__name__}: {exc}"
            return metadata

        metadata["final_raw_reply"] = final_raw
        final_parsed = self.self_review_parser.parse(final_raw)
        metadata["final_parse"] = final_parsed.to_dict()
        if final_parsed.reply_type == "final_answer" and final_parsed.answer:
            metadata.update(
                {
                    "applied": True,
                    "answer": final_parsed.answer,
                    "parse_completed": bool(final_parsed.parse_completed),
                    "validator_passed": bool(final_parsed.parse_completed),
                    "error": final_parsed.error,
                }
            )
            return metadata
        metadata["error"] = final_parsed.error or "self_review_final_parse_failed"
        return metadata

    def _build_self_review_messages(self, answers: list[str]) -> list[dict[str, str]]:
        content = (
            "You are reviewing your own previous final answers.\n\n"
            f"Question:\n{self.question}\n\n"
            "Your previous final answers:\n"
            f"<answer>\n{answers[0]}\n</answer>\n"
            f"<answer>\n{answers[1]}\n</answer>\n"
            f"<answer>\n{answers[2]}\n</answer>\n\n"
            "Task:\n"
            "Decide the best final answer. If the previous answers are inconsistent, choose the answer that best satisfies the question. "
            "If you cannot decide from the previous answers alone, request one tool call.\n\n"
            "Rules:\n"
            "- Use only the question, previous final answers, and tool result if a tool is used.\n"
            "- Do not invent new facts.\n"
            "- Prefer a short final answer.\n"
            "- If the question asks for a number, output only the number.\n"
            "- If the question asks for a name, title, place, or yes/no answer, output only that answer.\n"
            "- Do not answer with a candidate number, candidate letter, or index.\n"
            "- You may request at most one tool call.\n"
            "- Output JSON only.\n\n"
            "If you can answer now:\n"
            "{\"type\":\"final_answer\",\"answer\":\"...\"}\n\n"
            "If you need a tool:\n"
            "{\"type\":\"tool_request\",\"tool_name\":\"...\",\"arguments\":{}}\n"
        )
        return [{"role": "user", "content": content}]

    def _build_self_review_after_tool_messages(
        self,
        *,
        answers: list[str],
        tool_result: dict[str, Any],
    ) -> list[dict[str, str]]:
        content = (
            "You requested a tool during self-review.\n\n"
            f"Question:\n{self.question}\n\n"
            "Your previous final answers:\n"
            f"<answer>\n{answers[0]}\n</answer>\n"
            f"<answer>\n{answers[1]}\n</answer>\n"
            f"<answer>\n{answers[2]}\n</answer>\n\n"
            f"Tool result:\n{self._format_review_tool_result(tool_result)}\n\n"
            "Task:\n"
            "Use the tool result to decide the final answer.\n\n"
            "Rules:\n"
            "- Use only the question, previous final answers, and tool result.\n"
            "- Do not request another tool.\n"
            "- Do not invent new facts.\n"
            "- Prefer a short final answer.\n"
            "- Do not answer with a candidate number, candidate letter, or index.\n"
            "- Output JSON only.\n\n"
            "Return:\n"
            "{\"type\":\"final_answer\",\"answer\":\"...\"}\n"
        )
        return [{"role": "user", "content": content}]

    def _normalize_review_tool_name(self, tool_name: str) -> str:
        name = str(tool_name or "").strip()
        if name in {"calculator", "python"}:
            return "python_calculator"
        if name in {"deterministic", "solver"}:
            return "deterministic_solver"
        if name in {"attachment", "file_reader", "reader"}:
            return "attachment_reader"
        if name in {"wikipedia_search", "web_search", "internet_search", "google_search"}:
            return "search"
        return name

    def _normalize_review_tool_args(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        args = dict(tool_args or {})
        if tool_name == "search":
            if "input" not in args and "query" in args:
                args["input"] = args["query"]
            args.setdefault("input", self.question)
            args.setdefault("mode", "text")
        elif tool_name == "python_calculator":
            if "input" not in args and "expression" in args:
                args["input"] = args["expression"]
        elif tool_name == "deterministic_solver":
            args.setdefault("input", args.get("question") or args.get("query") or self.question)
        elif tool_name == "attachment_reader":
            args.setdefault("question", args.get("input") or args.get("query") or self.question)
            if isinstance(self.attachment, dict) and self.attachment:
                args.setdefault("attachment", self.attachment)
                file_path = self.attachment.get("file_path") or self.attachment.get("path")
                if file_path:
                    args.setdefault("file_path", file_path)
        return args

    def _format_review_tool_result(self, tool_result: dict[str, Any]) -> str:
        output = str(tool_result.get("output_text", "") or "").strip()
        if len(output) > 4000:
            output = output[:4000] + "\n...[truncated]"
        return (
            f"tool_name={tool_result.get('tool_name', '')}\n"
            f"status={tool_result.get('status', '')}\n"
            f"ok={tool_result.get('ok', False)}\n"
            f"output={output}"
        )

    def _safe_run_single_agent(
        self,
        config: AgentConfig,
        run_index: int,
        evidence: dict[str, Any],
        *,
        unload_after_call: bool = False,
    ) -> EachAgentReply:
        """
        執行單一 Agent run，並將例外轉成可寫入報告的 Stage1 reply。

        Args:
            - config: 本次執行的 Agent 設定。
            - run_index: Stage1 run 編號。
            - evidence: EvidenceRunner 輸出的 evidence。

        Returns:
            - EachAgentReply: 正常或錯誤狀態的 Agent 回覆。
        """
        try:
            return self._run_single_agent(
                config,
                run_index,
                evidence,
                unload_after_call=unload_after_call,
            )
        except Exception as exc:
            return EachAgentReply(
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
                context_budget={},
            )

    def _unload_agent_model(
        self,
        *,
        config: AgentConfig,
        phase: str,
    ) -> None:
        """
        在下一個 SLM 開始回答前，透過 Ollama CLI 卸載上一個 SLM。

        Args:
            - previous_config: 剛完成回答的 Agent 設定。
            - next_config: 即將開始回答的 Agent 設定。
            - run_index: 即將執行的 Stage1 run 編號。

        Returns:
            - None
        """
        unload = getattr(self.get_agent(config), "unload", None)
        if callable(unload):
            record = dict(unload())
        else:
            record = {
                "model": config.model_name,
                "provider": "",
                "unload_method": "none",
                "unloaded": False,
                "warning": "agent_has_no_unload_method",
            }
        record.update(
            {
                "phase": phase,
                "agent_id": config.agent_id,
                "model": config.model_name,
            }
        )
        self.model_lifecycle_records.append(record)

    def worker_count(self) -> int:
        """
        計算 Stage1 實際使用的平行 worker 數量。

        Args:
            - 無。

        Returns:
            - int: 依 Agent 數、run 數與 max_workers 限制後的 worker 數。
        """
        if self.unload_previous_slm_on_switch:
            return 1
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
        *,
        unload_after_call: bool = False,
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
                attachment=self.attachment,
                unload_after_run=unload_after_call,
            )
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return reply

        messages, context_budget = self.context_builder.build_with_diagnostics(
            question=self.question,
            evidence_packets=self.evidence_to_context_packets(evidence),
        )
        raw_reply = ""
        reasoning = ""
        final_answer = ""
        parse_completed = False
        structured_output: dict[str, Any] = {}
        schema_valid = False
        schema_errors: list[str] = []
        repair_applied = False
        repair_actions: list[str] = []
        eligible_for_winner = False
        validity_labels: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        try:
            raw_reply, prompt_tokens, completion_tokens = self.get_agent(config).invoke_with_usage(
                messages,
                unload_after_call=unload_after_call,
            )
            self.record_token_usage(
                stage="stage1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            parsed = self.parser.parse(raw_reply, expected_weight_count=0)
            reasoning = str(parsed.get("reasoning", "")).strip()
            final_answer = str(parsed.get("final_answer", "")).strip()
            parse_completed = bool(parsed.get("parse_completed"))
            structured_output = dict(parsed.get("structured_output") or {})
            schema_valid = bool(parsed.get("schema_valid"))
            schema_errors = list(parsed.get("schema_errors") or [])
            repair_applied = bool(parsed.get("repair_applied"))
            repair_actions = list(parsed.get("repair_actions") or [])
            eligible_for_winner = bool(parsed.get("eligible_for_winner"))
            validity_labels = list(parsed.get("validity_labels") or [])
        except Exception as exc:
            raw_reply = raw_reply or f"[stage1_error] {type(exc).__name__}: {exc}"
            schema_errors = [type(exc).__name__]
            validity_labels = ["parse_exception"]

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
            structured_output=structured_output,
            schema_valid=schema_valid,
            schema_errors=schema_errors,
            repair_applied=repair_applied,
            repair_actions=repair_actions,
            eligible_for_winner=eligible_for_winner,
            validity_labels=validity_labels,
            context_budget=context_budget,
        )


__all__ = ["Stage1Runner"]
