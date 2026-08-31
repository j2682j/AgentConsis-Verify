from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from context.stage1_final_answer_repair_context import Stage1FinalAnswerRepairContextBuilder
from context.stage1_tool_context import Stage1ToolContextBuilder
from core.config import AgentConfig, EachAgentReply
from core.sampling_seed import sampling_overrides
from core.slm_agent import SLM_Agent
from core.stage1_search_gate import Stage1SearchAccessState
from core.tool_turn_policy import AdaptiveToolTurnPolicy
from parsers.tool_request_parser import ToolRequestParser
from parsers.reasoning_parser import prepare_reasoning_for_verifier
from tools.tool_cache import ToolCache
from tools.attachment_workspace import AttachmentWorkspace
from tools.evidence.fact_extraction import (
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactCollector,
    TaskFactStore,
)


class Stage1ToolUseRunner:
    """
    執行單次 Stage1 tool-use trajectory，讓 Agent 可在回答前多回合請求工具。

    Args:
        - context_builder: 建立 Stage1 tool-use prompt 的 Stage1ToolContextBuilder。
        - parser: 解析 tool_request 與 final_answer JSON 的 ToolRequestParser。
        - tool_cache: 快取相同 tool args 的 ToolCache。
        - tool_manager: 實際執行 search、python_calculator 等工具的管理器。
        - max_tool_turns: 每次 run 的初始工具回合預算。
        - hard_max_tool_turns: 工具有持續進展時可延長到的硬上限。
        - no_progress_limit: 連續無有效新證據多少次後停止工具使用。

    Returns:
        - tuple[EachAgentReply, int, int]: 單次 run 結果、累計 prompt tokens、累計 completion tokens。
        - EachAgentReply: 包含 trajectory、tool_calls、tool_results 與 final answer。
    """

    def __init__(
        self,
        *,
        context_builder: Stage1ToolContextBuilder | None = None,
        repair_context_builder: Stage1FinalAnswerRepairContextBuilder | None = None,
        parser: ToolRequestParser | None = None,
        tool_cache: ToolCache | None = None,
        tool_manager: Any | None = None,
        max_tool_turns: int = 2,
        hard_max_tool_turns: int = 4,
        no_progress_limit: int = 2,
        semantic_fact_extractor: SemanticFactExtractor | None = None,
        fact_collector: TaskFactCollector | None = None,
    ) -> None:
        self.context_builder = context_builder or Stage1ToolContextBuilder()
        self.repair_context_builder = (
            repair_context_builder or Stage1FinalAnswerRepairContextBuilder()
        )
        self.parser = parser or ToolRequestParser()
        self.tool_cache = tool_cache or ToolCache()
        self.tool_manager = tool_manager
        self.max_tool_turns = max(0, max_tool_turns)
        self.hard_max_tool_turns = max(
            self.max_tool_turns,
            max(0, hard_max_tool_turns),
        )
        self.no_progress_limit = max(1, no_progress_limit)
        self.semantic_fact_extractor = semantic_fact_extractor
        self.fact_collector = fact_collector or TaskFactCollector()

    def run(
        self,
        *,
        config: AgentConfig,
        agent: SLM_Agent,
        question: str,
        evidence_packets: list[Any],
        run_index: int,
        attachment: dict[str, Any] | None = None,
        unload_after_run: bool = False,
        search_access_state: Stage1SearchAccessState | None = None,
        attachment_workspace: AttachmentWorkspace | None = None,
        fact_store: TaskFactStore | None = None,
    ) -> tuple[EachAgentReply, int, int]:
        """
        執行單一 Agent 的 tool-use reasoning 回合，直到產生 final answer 或達到工具上限。

        Args:
            - config: 目前執行的 AgentConfig。
            - agent: 已建立的 SLM_Agent。
            - question: 使用者輸入的問題。
            - evidence_packets: Stage1 可用的 evidence packets。
            - run_index: 此 Agent 的第幾次 Stage1 run。

        Returns:
            - EachAgentReply: 單次 trajectory 的解析結果。
            - int: 累計 prompt token 數。
            - int: 累計 completion token 數。
        """
        trajectory: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        reasoning_steps: list[str] = []
        raw_reply = ""
        available_tools = self._available_tools()
        tool_gap = self._tool_gap(question, attachment)
        policy = AdaptiveToolTurnPolicy(
            base_budget=self.max_tool_turns,
            hard_limit=self.hard_max_tool_turns,
            no_progress_limit=self.no_progress_limit,
        )
        finalization_prompt_sent = False
        repair_attempted = False
        repair_reason = ""
        turn_index = 0
        blocked_search_requests = 0
        evidence_packets = list(evidence_packets)
        if search_access_state is not None:
            evidence_packets.extend(search_access_state.supplemental_packets())

        while turn_index < policy.hard_limit + 2:
            turn_index += 1
            current_turn_is_repair = bool(policy.force_final and finalization_prompt_sent)
            if current_turn_is_repair:
                messages, context_budget = self.repair_context_builder.build_with_diagnostics(
                    question=question,
                    evidence_packets=evidence_packets,
                    tool_trace=self._format_tool_trace(tool_results),
                    previous_reply=raw_reply,
                    repair_reason=repair_reason or policy.stop_reason,
                    attachment=attachment,
                )
            else:
                messages, context_budget = self.context_builder.build_with_diagnostics(
                    question=question,
                    evidence_packets=evidence_packets,
                    tool_trace=self._format_tool_trace(tool_results),
                    attachment=attachment,
                    available_tools=available_tools,
                    tool_gap=tool_gap,
                    tool_turn_policy=policy.format_prompt(),
                    search_access=(
                        search_access_state.format_prompt()
                        if search_access_state is not None
                        else "Mode: normal search access."
                    ),
                    attachment_access=(
                        self._attachment_access_prompt(attachment_workspace)
                    ),
                )
            raw_reply, prompt_tokens, completion_tokens = agent.invoke_with_usage(
                messages,
                **sampling_overrides(
                    agent_id=config.agent_id,
                    run_index=run_index,
                    turn=turn_index,
                ),
            )
            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            parsed = self.parser.parse(raw_reply)
            trajectory.append(
                {
                    "turn": turn_index,
                    "type": parsed.get("type", "invalid"),
                    "raw_reply": raw_reply,
                    "parsed": parsed,
                    "tool_turn_policy": policy.snapshot(),
                    "repair_turn": current_turn_is_repair,
                    "context_budget": context_budget,
                }
            )

            if parsed["type"] == "tool_request":
                tool_name = self._normalize_tool_name(parsed.get("tool_name", ""))
                tool_args = self._normalize_tool_args(
                    tool_name,
                    parsed.get("tool_args", {}),
                    question=question,
                    attachment=attachment,
                    reasoning_step=str(parsed.get("reasoning_step", "") or "").strip(),
                )
                tool_name, tool_args = self._reroute_local_media_tool(
                    tool_name,
                    tool_args,
                    question=question,
                    attachment=attachment,
                )
                missing_information = str(
                    tool_args.pop("missing_information", "") or ""
                ).strip()
                reasoning_step = str(parsed.get("reasoning_step", "") or "").strip()
                if reasoning_step:
                    reasoning_steps.append(reasoning_step)
                tool_call = {
                    "turn": turn_index,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "reasoning_step": reasoning_step,
                }
                if missing_information:
                    tool_call["missing_information"] = missing_information
                tool_calls.append(tool_call)

                if (
                    tool_name == "search"
                    and search_access_state is not None
                    and policy.can_execute()
                ):
                    search_query = str(
                        tool_args.get("input") or tool_args.get("query") or ""
                    ).strip()
                    gate_decision = search_access_state.authorize(
                        query=search_query,
                        missing_information=missing_information,
                        agent_id=config.agent_id,
                    )
                    tool_call["search_gate"] = {
                        "allowed": gate_decision.allowed,
                        "reason": gate_decision.reason,
                        "use_existing_evidence": gate_decision.use_existing_evidence,
                    }
                    if not gate_decision.allowed:
                        blocked_search_requests += 1
                        tool_result = search_access_state.blocked_result(gate_decision)
                        tool_result["stage1_search_gate"] = search_access_state.snapshot()
                        tool_results.append(tool_result)
                        if blocked_search_requests >= 2:
                            policy.request_final_answer("repeated_search_request_blocked")
                            finalization_prompt_sent = True
                            repair_attempted = True
                            repair_reason = "repeated_search_request_blocked"
                        trajectory.append(
                            {
                                "turn": turn_index,
                                "type": "tool_result",
                                "tool_call": tool_call,
                                "tool_result": tool_result,
                                "tool_turn_policy": policy.snapshot(),
                                "stage1_search_gate": search_access_state.snapshot(),
                            }
                        )
                        continue

                if not policy.can_execute():
                    tool_result = policy.block_result(tool_name)
                    if finalization_prompt_sent:
                        tool_result["final_answer_repair_failed"] = True
                        tool_results.append(tool_result)
                        break
                    finalization_prompt_sent = True
                    repair_attempted = True
                    repair_reason = policy.stop_reason or "tool_turn_budget_exhausted"
                else:
                    tool_result = self.tool_cache.get_or_execute(
                        tool_manager=self.tool_manager,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_id=config.agent_id,
                        stage=f"stage1_tool_turn_{policy.turns_used + 1}",
                        runtime_context=(
                            {
                                "attachment_workspace": attachment_workspace,
                                "question": question,
                                "search_context": self._search_context(evidence_packets),
                            }
                            if tool_name == "attachment_reader"
                            and attachment_workspace is not None
                            else None
                        ),
                    )
                    if (
                        tool_name == "attachment_reader"
                        and attachment_workspace is not None
                        and tool_result.get("cache_hit")
                    ):
                        attachment_workspace.record_tool_cache_hit()
                    if tool_name == "search" and search_access_state is not None:
                        search_access_state.complete(
                            query=str(tool_args.get("input") or tool_args.get("query") or ""),
                            result=tool_result,
                        )
                        tool_result["stage1_search_gate"] = search_access_state.snapshot()
                    self._attach_semantic_facts(
                        question=question,
                        tool_name=tool_name,
                        tool_result=tool_result,
                    )
                    if fact_store is not None:
                        self.fact_collector.collect_item(
                            fact_store,
                            tool_result,
                            question=question,
                            source_scope="stage1_tool_use",
                        )
                    progress = policy.observe(tool_result)
                    tool_result["adaptive_progress"] = progress
                    tool_result["tool_turn_policy"] = policy.snapshot()
                    if policy.force_final:
                        finalization_prompt_sent = True
                        repair_attempted = True
                        repair_reason = policy.stop_reason
                tool_results.append(tool_result)
                trajectory.append(
                    {
                        "turn": turn_index,
                        "type": "tool_result",
                        "tool_call": tool_call,
                        "tool_result": tool_result,
                        "tool_turn_policy": policy.snapshot(),
                    }
                )
                continue

            if parsed["type"] == "final_answer":
                if (
                    not bool(parsed.get("eligible_for_winner"))
                    and not repair_attempted
                    and self._should_repair_final_answer(parsed)
                ):
                    repair_attempted = True
                    repair_reason = self._repair_reason_from_parsed(parsed)
                    policy.request_final_answer(repair_reason)
                    finalization_prompt_sent = True
                    trajectory[-1]["retry_final_answer"] = True
                    trajectory[-1]["repair_reason"] = repair_reason
                    continue

                reasoning = str(parsed.get("reasoning", "") or "").strip()
                if not reasoning and reasoning_steps:
                    reasoning = "\n".join(
                        f"step {index}. {self._strip_step_marker(step)}"
                        for index, step in enumerate(reasoning_steps, start=1)
                        if self._strip_step_marker(step)
                    )
                final_answer = str(parsed.get("final_answer", "") or "").strip()
                return (
                    self._make_reply(
                        config=config,
                        run_index=run_index,
                        raw_reply=raw_reply,
                        reasoning=reasoning,
                        final_answer=final_answer,
                        parsed=parsed,
                        tool_results=tool_results,
                        tool_calls=tool_calls,
                        trajectory=trajectory,
                        prompt_tokens_total=prompt_tokens_total,
                        completion_tokens_total=completion_tokens_total,
                        repair_attempted=repair_attempted,
                        repair_reason=repair_reason,
                        final_answer_source=(
                            "repair_turn" if current_turn_is_repair else "original"
                        ),
                    ),
                    prompt_tokens_total,
                    completion_tokens_total,
                )

            if parsed["type"] == "invalid" and not finalization_prompt_sent:
                repair_attempted = True
                repair_reason = "invalid_reply_requires_final_answer"
                policy.request_final_answer("invalid_reply_requires_final_answer")
                finalization_prompt_sent = True
                trajectory[-1]["retry_final_answer"] = True
                trajectory[-1]["repair_reason"] = repair_reason
                continue

            break

        return (
            self._make_no_final_reply(
                config=config,
                run_index=run_index,
                raw_reply=raw_reply,
                reasoning="\n".join(reasoning_steps),
                tool_results=tool_results,
                tool_calls=tool_calls,
                trajectory=trajectory,
                prompt_tokens_total=prompt_tokens_total,
                completion_tokens_total=completion_tokens_total,
                repair_attempted=repair_attempted,
                repair_reason=repair_reason or policy.stop_reason,
            ),
            prompt_tokens_total,
            completion_tokens_total,
        )

    def _make_reply(
        self,
        *,
        config: AgentConfig,
        run_index: int,
        raw_reply: str,
        reasoning: str,
        final_answer: str,
        parsed: dict[str, Any],
        tool_results: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        prompt_tokens_total: int,
        completion_tokens_total: int,
        repair_attempted: bool,
        repair_reason: str,
        final_answer_source: str,
    ) -> EachAgentReply:
        repair_actions = list(parsed.get("repair_actions") or [])
        if repair_attempted and "final_answer_repair_turn" not in repair_actions:
            repair_actions.append("final_answer_repair_turn")
        repair_metadata = {
            "attempted": bool(repair_attempted),
            "reason": str(repair_reason or ""),
            "source": final_answer_source,
            "success": bool(parsed.get("eligible_for_winner")),
        }
        reasoning_parse = prepare_reasoning_for_verifier(
            reasoning,
            final_answer=final_answer,
        ).to_dict()
        return EachAgentReply(
            agent_id=config.agent_id,
            model_name=config.model_name,
            run_index=run_index,
            raw_reply=raw_reply,
            reasoning=reasoning,
            final_answer=final_answer,
            parse_completed=bool(parsed.get("eligible_for_winner")),
            tool_context=self._format_tool_trace(tool_results),
            context_source="runtime_tool_trace",
            runtime_tool_trace_chars=len(
                str(self._format_tool_trace(tool_results) or "")
            ),
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
            total_tokens=prompt_tokens_total + completion_tokens_total,
            tool_calls=tool_calls,
            tool_results=tool_results,
            trajectory=trajectory,
            structured_output=dict(parsed.get("structured_output") or {}),
            schema_valid=bool(parsed.get("schema_valid")),
            schema_errors=list(parsed.get("schema_errors") or []),
            repair_applied=bool(parsed.get("repair_applied")) or bool(repair_attempted),
            repair_actions=repair_actions,
            eligible_for_winner=bool(parsed.get("eligible_for_winner")),
            validity_labels=list(parsed.get("validity_labels") or []),
            final_answer_source=final_answer_source,
            repair_metadata=repair_metadata,
            context_budget=self._context_budget_summary(trajectory),
            reasoning_parse_quality=str(
                reasoning_parse.get("quality_status") or "unreliable"
            ),
            reasoning_versa_eligible=bool(reasoning_parse.get("versa_eligible")),
            reasoning_parse_diagnostics=dict(
                reasoning_parse.get("diagnostics") or {}
            ),
            reasoning_steps=[
                (int(item[0]), str(item[1]))
                for item in list(reasoning_parse.get("steps") or [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ],
        )

    @staticmethod
    def _strip_step_marker(value: str) -> str:
        """Remove an existing step prefix before trajectory steps are renumbered."""

        import re

        return re.sub(
            r"^\s*(?:step\s*)?\d{1,3}\s*[.):\-]\s*",
            "",
            str(value or "").strip(),
            flags=re.IGNORECASE,
        ).strip()

    def _make_no_final_reply(
        self,
        *,
        config: AgentConfig,
        run_index: int,
        raw_reply: str,
        reasoning: str,
        tool_results: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        prompt_tokens_total: int,
        completion_tokens_total: int,
        repair_attempted: bool,
        repair_reason: str,
    ) -> EachAgentReply:
        schema_errors = ["tool_trajectory_no_final_answer"]
        validity_labels = ["tool_trajectory_no_final_answer"]
        if repair_attempted:
            schema_errors.append("final_answer_repair_failed")
            validity_labels.append("final_answer_repair_failed")
        return EachAgentReply(
            agent_id=config.agent_id,
            model_name=config.model_name,
            run_index=run_index,
            raw_reply=raw_reply,
            reasoning=reasoning,
            final_answer="",
            parse_completed=False,
            tool_context=self._format_tool_trace(tool_results),
            context_source="runtime_tool_trace",
            runtime_tool_trace_chars=len(
                str(self._format_tool_trace(tool_results) or "")
            ),
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
            total_tokens=prompt_tokens_total + completion_tokens_total,
            tool_calls=tool_calls,
            tool_results=tool_results,
            trajectory=trajectory,
            structured_output={},
            schema_valid=False,
            schema_errors=schema_errors,
            repair_applied=bool(repair_attempted),
            repair_actions=(
                ["final_answer_repair_turn"] if repair_attempted else []
            ),
            eligible_for_winner=False,
            validity_labels=validity_labels,
            final_answer_source="none",
            repair_metadata={
                "attempted": bool(repair_attempted),
                "reason": str(repair_reason or ""),
                "source": "none",
                "success": False,
            },
            context_budget=self._context_budget_summary(trajectory),
        )

    def _context_budget_summary(self, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        """Fold the per-turn budgets into one record, without losing the turns.

        Summing chars across turns and dropping everything else made the tool-use
        path report less than the non-tool path: the prepared and rendered search
        context, the only fields that say whether evidence survived budgeting,
        never reached the run record at all.

        Hashes are kept per turn and never combined. A hash of concatenated turns
        identifies nothing that can be compared against anything, whereas the
        first turn's hash is exactly what a later replay needs to check.
        """

        budgets = [
            dict(item.get("context_budget") or {})
            for item in trajectory
            if isinstance(item.get("context_budget"), dict)
        ]
        if not budgets:
            return {}

        turns = [
            {
                "turn": index,
                "prepared_search_context_chars": int(
                    item.get("prepared_search_context_chars", 0) or 0
                ),
                "prepared_search_context_hash": str(
                    item.get("prepared_search_context_hash", "") or ""
                ),
                "rendered_search_context_chars": int(
                    item.get("rendered_search_context_chars", 0) or 0
                ),
                "rendered_search_context_hash": str(
                    item.get("rendered_search_context_hash", "") or ""
                ),
                "search_result_truncated": bool(item.get("search_result_truncated")),
                "section_chars": dict(item.get("section_chars") or {}),
            }
            for index, item in enumerate(budgets, 1)
        ]
        first = turns[0]
        return {
            "turn_count": len(budgets),
            "original_chars": sum(int(item.get("original_chars", 0) or 0) for item in budgets),
            "final_chars": sum(int(item.get("final_chars", 0) or 0) for item in budgets),
            "truncation_applied": any(bool(item.get("truncation_applied")) for item in budgets),
            # Kept, but not evidence of anything on its own: reference tails are
            # trimmed without incrementing it, by design and by test. Loss shows
            # up in `search_result_truncated` and the rendered chars instead.
            "dropped_evidence_count": sum(
                int(item.get("dropped_evidence_count", 0) or 0) for item in budgets
            ),
            "truncated_sections": sorted(
                {
                    str(section)
                    for item in budgets
                    for section in list(item.get("truncated_sections") or [])
                }
            ),
            "turns": turns,
            "first_turn_prepared_search_context_chars": first["prepared_search_context_chars"],
            "first_turn_prepared_search_context_hash": first["prepared_search_context_hash"],
            "first_turn_rendered_search_context_chars": first["rendered_search_context_chars"],
            "first_turn_rendered_search_context_hash": first["rendered_search_context_hash"],
            "search_result_truncated_turn_count": sum(
                1 for turn in turns if turn["search_result_truncated"]
            ),
        }

    def _should_repair_final_answer(self, parsed: dict[str, Any]) -> bool:
        labels = {
            str(label or "").strip()
            for label in parsed.get("validity_labels", []) or []
            if str(label or "").strip()
        }
        schema_errors = {
            str(error or "").strip()
            for error in parsed.get("schema_errors", []) or []
            if str(error or "").strip()
        }
        repairable_labels = {
            "empty_final_answer",
            "invalid_final_answer",
            "refusal_like_final_answer",
            "too_verbose_final_answer",
            "uncertain_final_answer",
            "tool_call_as_final_answer",
        }
        repairable_schema_errors = {
            "malformed_reasoning_steps",
            "unknown_answer_type",
            "invalid_evidence_id",
            "confidence_out_of_range",
        }
        return bool(labels & repairable_labels or schema_errors & repairable_schema_errors)

    def _repair_reason_from_parsed(self, parsed: dict[str, Any]) -> str:
        labels = [str(label) for label in parsed.get("validity_labels", []) or [] if label]
        errors = [str(error) for error in parsed.get("schema_errors", []) or [] if error]
        reasons = labels + errors
        return ",".join(reasons) if reasons else "invalid_final_answer_requires_repair"

    def _normalize_tool_name(self, tool_name: str) -> str:
        """
        正規化 Agent 回傳的工具名稱，將常見別名轉成系統支援名稱。

        Args:
            - tool_name: Agent 回傳的工具名稱。

        Returns:
            - str: 正規化後的工具名稱。
        """
        name = str(tool_name or "").strip()
        if name in {"calculator", "python"}:
            return "python_calculator"
        if name in {"deterministic", "solver"}:
            return "deterministic_solver"
        if name in {"attachment", "file_reader", "reader"}:
            return "attachment_reader"
        return name

    def _normalize_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        question: str = "",
        attachment: dict[str, Any] | None = None,
        reasoning_step: str = "",
    ) -> dict[str, Any]:
        """
        正規化工具參數，補齊 search 預設欄位並處理 query/input 別名。

        Args:
            - tool_name: 正規化後的工具名稱。
            - tool_args: Agent 回傳的工具參數。
            - reasoning_step: 本回合工具請求的理由，用於補齊 missing_information。

        Returns:
            - dict[str, Any]: 可傳給 ToolManager 的工具參數。
        """
        args = dict(tool_args or {})
        if tool_name == "search":
            if "input" not in args and "query" in args:
                args["input"] = args["query"]
            args.setdefault("mode", "text")
            if not str(args.get("missing_information", "") or "").strip():
                derived = self._missing_information_from_reasoning(reasoning_step)
                if derived:
                    args["missing_information"] = derived
        elif tool_name == "python_calculator":
            if "input" not in args and "expression" in args:
                args["input"] = args["expression"]
        elif tool_name == "deterministic_solver":
            if "input" not in args:
                args["input"] = args.get("question") or args.get("query") or question
        elif tool_name == "attachment_reader":
            if "question" not in args:
                args["question"] = args.get("input") or args.get("query") or question
            if "file_path" not in args:
                for key in ("path", "attachment_path"):
                    if key in args:
                        args["file_path"] = args[key]
                        break
            if "file_path" not in args and isinstance(attachment, dict):
                file_path = attachment.get("file_path") or attachment.get("path")
                if file_path:
                    args["file_path"] = file_path
            if "attachment" not in args and isinstance(attachment, dict) and attachment:
                args["attachment"] = attachment
            args.setdefault(
                "information_need",
                args.get("input") or args.get("query") or args.get("question") or question,
            )
        return args

    _STEP_PREFIX_RE = re.compile(r"^\s*step\s*\d+\s*[.:)-]\s*", re.IGNORECASE)
    _NEED_PREFIX_RE = re.compile(
        r"^\s*(?:i\s+need\s+to\s+|i\s+must\s+|i\s+should\s+|need\s+to\s+)", re.IGNORECASE
    )
    _MISSING_INFORMATION_MAX_CHARS = 200
    # Words that only restate that something is missing. A step built from these
    # alone names nothing, so it cannot stand in for the gate's required field.
    _GENERIC_NEED_WORDS = frozenset(
        {
            "a", "an", "and", "answer", "any", "be", "data", "detail", "details",
            "fact", "facts", "find", "for", "from", "get", "identify", "in",
            "information", "is", "it", "its", "locate", "missing", "more",
            "needed", "obtain", "of", "on", "one", "out", "required", "retrieve",
            "search", "specific", "that", "the", "this", "to", "tool", "value",
            "values", "what", "which", "why", "with",
        }
    )
    _MISSING_INFORMATION_MIN_CONTENT_WORDS = 2

    @classmethod
    def _missing_information_from_reasoning(cls, reasoning_step: str) -> str:
        """Recover the search gate's required field from the adjacent one.

        The gate blocks a search whose tool_args omit `missing_information`, and
        the block costs the run one of its few tool turns. But the agent has
        usually already stated the need in `reasoning_step` -- across
        level1_final_06, _07 and _08 all 142 requests blocked this way carried
        one, reading like "step 1. I need to find the minimum perigee distance
        between Earth and the Moon". The field is misplaced, not absent.

        Recovery is deliberately narrow, because reading any step across would
        make the gate's requirement vacuous: the prompt asks for a
        `reasoning_step` on every tool request, so every request would satisfy
        it. The step has to actually name something -- "Obtain the one missing
        fact" restates the need without stating it, and still blocks, while the
        142 real steps name a distance, a nominator, a season.
        """

        text = str(reasoning_step or "").strip()
        if not text:
            return ""
        text = cls._STEP_PREFIX_RE.sub("", text).strip()
        text = cls._NEED_PREFIX_RE.sub("", text).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return ""
        content = [
            word
            for word in re.findall(r"[a-z0-9']+", text.casefold())
            if word not in cls._GENERIC_NEED_WORDS
        ]
        if len(content) < cls._MISSING_INFORMATION_MIN_CONTENT_WORDS:
            return ""
        if len(text) > cls._MISSING_INFORMATION_MAX_CHARS:
            text = text[: cls._MISSING_INFORMATION_MAX_CHARS].rstrip()
        return text

    def _attachment_access_prompt(
        self,
        workspace: AttachmentWorkspace | None,
    ) -> str:
        if workspace is None:
            return "Prepared attachment: unavailable."
        state = workspace.snapshot()
        return (
            f"Prepared attachment: {state.get('prepared_available', False)}\n"
            f"Parse status: {state.get('parse_status', 'not_prepared')}\n"
            f"Available data: {state.get('available_inputs', []) or ['none']}\n"
            f"Eligible handlers: {state.get('eligible_handlers', []) or ['none']}\n"
            "Instruction: request attachment_reader only for one specific missing fact."
        )

    @staticmethod
    def _search_context(evidence_packets: list[Any]) -> str:
        return "\n\n".join(
            str(getattr(packet, "content", "") or "").strip()
            for packet in evidence_packets
            if getattr(packet, "packet_type", "") == "search_result"
            and str(getattr(packet, "content", "") or "").strip()
        )

    _LOCAL_MEDIA_EXTENSIONS = {
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
    }

    def _reroute_local_media_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        question: str,
        attachment: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if tool_name != "video_transcript":
            return tool_name, tool_args

        args = dict(tool_args or {})
        if self._is_remote_video_request(args):
            return tool_name, args

        if self._attachment_is_local_media(attachment):
            routed = dict(args)
            routed.setdefault("question", question)
            routed.setdefault("file_path", attachment.get("file_path") or attachment.get("path"))
            routed.setdefault("attachment", dict(attachment or {}))
            routed["rerouted_from"] = "video_transcript"
            return "attachment_reader", self._normalize_tool_args(
                "attachment_reader",
                routed,
                question=question,
                attachment=attachment,
            )

        candidate = str(
            args.get("file_path")
            or args.get("path")
            or args.get("url")
            or args.get("input")
            or args.get("query")
            or ""
        ).strip()
        if self._looks_like_local_media_path(candidate):
            routed = dict(args)
            routed.setdefault("question", question)
            routed.setdefault("file_path", candidate)
            routed["rerouted_from"] = "video_transcript"
            return "attachment_reader", self._normalize_tool_args(
                "attachment_reader",
                routed,
                question=question,
                attachment=attachment,
            )

        return tool_name, args

    def _is_remote_video_request(self, args: dict[str, Any]) -> bool:
        value = str(args.get("url") or args.get("input") or args.get("query") or "").strip().lower()
        return value.startswith(("http://", "https://"))

    def _attachment_is_local_media(self, attachment: dict[str, Any] | None) -> bool:
        if not isinstance(attachment, dict) or not attachment:
            return False
        extension = str(attachment.get("extension", "") or "").strip().lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if not extension:
            extension = Path(str(attachment.get("file_path", "") or attachment.get("path", "") or "")).suffix.lower()
        return extension in self._LOCAL_MEDIA_EXTENSIONS

    def _looks_like_local_media_path(self, value: str) -> bool:
        candidate = str(value or "").strip()
        if not candidate:
            return False
        if candidate.lower().startswith(("http://", "https://")):
            return False
        return Path(candidate).suffix.lower() in self._LOCAL_MEDIA_EXTENSIONS

    def _available_tools(self) -> str:
        if self.tool_manager is None:
            return "No enabled tools."
        formatter = getattr(self.tool_manager, "describe_enabled_tools", None)
        if callable(formatter):
            return str(formatter())
        return "No enabled tools."

    def _tool_gap(
        self,
        question: str,
        attachment: dict[str, Any] | None,
    ) -> str:
        if self.tool_manager is None:
            return "Tool manager unavailable."
        formatter = getattr(self.tool_manager, "format_tool_gap", None)
        if not callable(formatter):
            return "Capability analysis unavailable."
        extension = ""
        if isinstance(attachment, dict):
            extension = str(attachment.get("extension", "") or "").lstrip(".")
        return str(formatter(question, attachment_type=extension or None))

    def _format_tool_trace(self, tool_results: list[dict[str, Any]]) -> str:
        """
        將已取得的工具結果格式化成下一回合 prompt 的 Tool_Trace。

        Args:
            - tool_results: 目前 trajectory 中已執行的工具結果。

        Returns:
            - str: 給 Agent 讀取的工具結果摘要。
        """
        if not tool_results:
            return "None"
        lines = []
        for index, result in enumerate(tool_results, 1):
            lines.append(
                f"Tool result {index}: {result.get('tool_name', '')} "
                f"status={result.get('status', '')} ok={result.get('ok', False)} "
                f"evidence_valid={result.get('evidence_valid', False)} "
                f"cache_hit={result.get('cache_hit', False)} "
                f"duplicate_request={result.get('duplicate_request', False)}"
            )
            output = str(result.get("output_text", "") or "").strip()
            if output:
                lines.append(output[:4000])
            error = result.get("error")
            if error:
                lines.append(f"Error: {error}")
            retry_hint = str(result.get("retry_hint", "") or "").strip()
            if retry_hint:
                lines.append(f"Next action: {retry_hint}")
        return "\n".join(lines)

    def _attach_semantic_facts(
        self,
        *,
        question: str,
        tool_name: str,
        tool_result: dict[str, Any],
    ) -> None:
        if tool_name not in {
            "attachment_reader",
            "video_evidence",
            "video_transcript",
        }:
            return
        if not tool_result.get("ok") or not tool_result.get("evidence_valid", False):
            return
        raw = tool_result.get("raw_result")
        if not isinstance(raw, dict):
            raw = {}
            tool_result["raw_result"] = raw
        if raw.get("semantic_facts"):
            return
        output_text = str(tool_result.get("output_text") or "").strip()
        if not output_text:
            return
        if self.semantic_fact_extractor is None:
            self.semantic_fact_extractor = SemanticFactExtractor(max_units_per_call=1)
        result = self.semantic_fact_extractor.extract_batch(
            question=question,
            answer_requirement=question,
            current_goal=question,
            units=[
                SemanticSourceUnit(
                    unit_id="T1",
                    text=output_text,
                    source_id=f"stage1:{tool_name}",
                    source_type=tool_name,
                    source_title=tool_name.replace("_", " ").title(),
                )
            ],
        )
        raw["semantic_facts"] = [fact.to_dict() for fact in result.facts]
        raw["semantic_fact_diagnostics"] = dict(result.diagnostics)


__all__ = ["Stage1ToolUseRunner"]
