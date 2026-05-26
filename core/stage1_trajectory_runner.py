from __future__ import annotations

from typing import Any

from context.stage1_tool_context import Stage1ToolContextBuilder
from core.config import AgentConfig, EachAgentReply
from core.slm_agent import SLM_Agent
from parsers.tool_request_parser import ToolRequestParser
from tools.tool_cache import ToolCache


class Stage1TrajectoryRunner:
    """
    執行單次 Stage1 tool-use trajectory，讓 Agent 可在回答前多回合請求工具。

    Args:
        - context_builder: 建立 Stage1 tool-use prompt 的 Stage1ToolContextBuilder。
        - parser: 解析 tool_request 與 final_answer JSON 的 ToolRequestParser。
        - tool_cache: 快取相同 tool args 的 ToolCache。
        - tool_manager: 實際執行 search、python_calculator 等工具的管理器。
        - max_tool_turns: 每次 run 最多允許的工具請求回合數。

    Returns:
        - tuple[EachAgentReply, int, int]: 單次 run 結果、累計 prompt tokens、累計 completion tokens。
        - EachAgentReply: 包含 trajectory、tool_calls、tool_results 與 final answer。
    """

    ALLOWED_TOOLS = {"search", "python_calculator"}

    def __init__(
        self,
        *,
        context_builder: Stage1ToolContextBuilder | None = None,
        parser: ToolRequestParser | None = None,
        tool_cache: ToolCache | None = None,
        tool_manager: Any | None = None,
        max_tool_turns: int = 2,
    ) -> None:
        self.context_builder = context_builder or Stage1ToolContextBuilder()
        self.parser = parser or ToolRequestParser()
        self.tool_cache = tool_cache or ToolCache()
        self.tool_manager = tool_manager
        self.max_tool_turns = max(0, max_tool_turns)

    def run(
        self,
        *,
        config: AgentConfig,
        agent: SLM_Agent,
        question: str,
        evidence_packets: list[Any],
        run_index: int,
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

        for turn_index in range(1, self.max_tool_turns + 2):
            messages = self.context_builder.build(
                question=question,
                evidence_packets=evidence_packets,
                tool_trace=self._format_tool_trace(tool_results),
            )
            raw_reply, prompt_tokens, completion_tokens = agent.invoke_with_usage(messages)
            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            parsed = self.parser.parse(raw_reply)
            trajectory.append(
                {
                    "turn": turn_index,
                    "type": parsed.get("type", "invalid"),
                    "raw_reply": raw_reply,
                    "parsed": parsed,
                }
            )

            if parsed["type"] == "tool_request" and turn_index <= self.max_tool_turns:
                tool_name = self._normalize_tool_name(parsed.get("tool_name", ""))
                tool_args = self._normalize_tool_args(tool_name, parsed.get("tool_args", {}))
                reasoning_step = str(parsed.get("reasoning_step", "") or "").strip()
                if reasoning_step:
                    reasoning_steps.append(reasoning_step)
                tool_call = {
                    "turn": turn_index,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "reasoning_step": reasoning_step,
                }
                tool_calls.append(tool_call)

                if tool_name not in self.ALLOWED_TOOLS:
                    tool_result = {
                        "ok": False,
                        "tool_name": tool_name,
                        "output_text": "",
                        "raw_result": None,
                        "error": f"tool '{tool_name}' is not allowed in stage1 tool mode",
                        "cache_hit": False,
                    }
                else:
                    tool_result = self.tool_cache.get_or_execute(
                        tool_manager=self.tool_manager,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_id=config.agent_id,
                        stage=f"stage1_tool_turn_{turn_index}",
                    )
                tool_results.append(tool_result)
                trajectory.append(
                    {
                        "turn": turn_index,
                        "type": "tool_result",
                        "tool_call": tool_call,
                        "tool_result": tool_result,
                    }
                )
                continue

            if parsed["type"] == "final_answer":
                reasoning = str(parsed.get("reasoning", "") or "").strip()
                if not reasoning and reasoning_steps:
                    reasoning = "\n".join(reasoning_steps)
                final_answer = str(parsed.get("final_answer", "") or "").strip()
                return (
                    EachAgentReply(
                        agent_id=config.agent_id,
                        model_name=config.model_name,
                        run_index=run_index,
                        raw_reply=raw_reply,
                        reasoning=reasoning,
                        final_answer=final_answer,
                        parse_completed=bool(final_answer),
                        tool_context=self._format_tool_trace(tool_results),
                        prompt_tokens=prompt_tokens_total,
                        completion_tokens=completion_tokens_total,
                        total_tokens=prompt_tokens_total + completion_tokens_total,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        trajectory=trajectory,
                    ),
                    prompt_tokens_total,
                    completion_tokens_total,
                )

            break

        return (
            EachAgentReply(
                agent_id=config.agent_id,
                model_name=config.model_name,
                run_index=run_index,
                raw_reply=raw_reply,
                reasoning="\n".join(reasoning_steps),
                final_answer="",
                parse_completed=False,
                tool_context=self._format_tool_trace(tool_results),
                prompt_tokens=prompt_tokens_total,
                completion_tokens=completion_tokens_total,
                total_tokens=prompt_tokens_total + completion_tokens_total,
                tool_calls=tool_calls,
                tool_results=tool_results,
                trajectory=trajectory,
            ),
            prompt_tokens_total,
            completion_tokens_total,
        )

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
        return name

    def _normalize_tool_args(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """
        正規化工具參數，補齊 search 預設欄位並處理 query/input 別名。

        Args:
            - tool_name: 正規化後的工具名稱。
            - tool_args: Agent 回傳的工具參數。

        Returns:
            - dict[str, Any]: 可傳給 ToolManager 的工具參數。
        """
        args = dict(tool_args or {})
        if tool_name == "search":
            if "input" not in args and "query" in args:
                args["input"] = args["query"]
            args.setdefault("mode", "text")
        return args

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
                f"ok={result.get('ok', False)} cache_hit={result.get('cache_hit', False)}"
            )
            output = str(result.get("output_text", "") or "").strip()
            if output:
                lines.append(output[:4000])
            error = result.get("error")
            if error:
                lines.append(f"Error: {error}")
        return "\n".join(lines)


__all__ = ["Stage1TrajectoryRunner"]
