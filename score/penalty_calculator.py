from __future__ import annotations

from typing import Any

from core.config import AgentReasoningSummary
from score.answer_validator import AnswerValidator


class PenaltyCalculator:
    """
    根據格式錯誤與 tool failure 等 deterministic 規則計算 candidate-level penalty。

    Args:
        - answer_validator: 用於檢查 final answer 格式與有效性的 AnswerValidator。

    Returns:
        - PenaltyCalculator: 可產生 agent-level penalty 結果的計算器。
    """

    MIN_PENALTY = -1.0
    MAX_PENALTY = 0.0

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def calculate(
        self,
        stage1_results: list[AgentReasoningSummary],
        question: str = "",
    ) -> list[dict[str, Any]]:
        """
        對所有 Stage1 候選結果計算 rule-based penalty。

        Args:
            - stage1_results: Stage1Runner 產生的各 Agent 候選結果。

        Returns:
            - list[dict[str, Any]]: 每個 Agent 的 penalty、reasons 與 agent_id。
        """
        return [
            self.calculate_for_agent(result, question=question)
            for result in stage1_results
        ]

    def calculate_for_agent(
        self,
        result: AgentReasoningSummary,
        question: str = "",
    ) -> dict[str, Any]:
        """
        對單一 AgentReasoningSummary 計算格式與工具相關 penalty。

        Args:
            - result: 單一 Agent 的 Stage1 聚合結果。

        Returns:
            - dict[str, Any]: 包含 agent_id、penalty 與 reasons。
        """
        penalties: list[float] = []
        reasons: list[str] = []

        self._add_format_penalties(result, penalties, reasons, question=question)
        self._add_tool_penalties(result, penalties, reasons)

        penalty = self._clamp(sum(penalties))
        return {
            "agent_id": result.agent_id,
            "penalty": penalty,
            "reasons": reasons,
        }

    def _add_format_penalties(
        self,
        result: AgentReasoningSummary,
        penalties: list[float],
        reasons: list[str],
        question: str = "",
    ) -> None:
        """
        根據 compressed answer 的有效性加入格式相關 penalty。

        Args:
            - result: 單一 Agent 的 Stage1 聚合結果。
            - penalties: 累積 penalty 數值的清單。
            - reasons: 累積 penalty reason 的清單。

        Returns:
            - None。
        """
        answer = self.answer_validator.clean(result.compressed_answer)

        if not result.active or not answer:
            self._add(penalties, reasons, -1.0, "inactive_or_no_answer")
            return

        if self.answer_validator.is_tool_call_like(answer):
            self._add(penalties, reasons, -1.0, "tool_call_as_final_answer")
            return

        is_refusal_like = self.answer_validator.is_refusal_like(answer)
        refusal_allowed = self.answer_validator.question_allow_refusal(question)

        if is_refusal_like and not refusal_allowed:
            self._add(penalties, reasons, -1.0, "refusal_like_final_answer")

        if self.answer_validator.is_uncertain(answer):
            self._add(penalties, reasons, -0.3, "uncertain_final_answer")

        if self.answer_validator.is_too_verbose(answer):
            self._add(penalties, reasons, -1.0, "too_verbose_final_answer")

        if not self.answer_validator.is_valid(answer) and not (
            is_refusal_like and refusal_allowed
        ):
            self._add(penalties, reasons, -0.75, "invalid_final_answer")

    def _add_tool_penalties(
        self,
        result: AgentReasoningSummary,
        penalties: list[float],
        reasons: list[str],
    ) -> None:
        """
        根據 tool calls 與 tool results 加入工具失敗相關 penalty。

        Args:
            - result: 單一 Agent 的 Stage1 聚合結果。
            - penalties: 累積 penalty 數值的清單。
            - reasons: 累積 penalty reason 的清單。

        Returns:
            - None。
        """
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for run in result.runs:
            tool_calls.extend(run.tool_calls or [])
            tool_results.extend(run.tool_results or [])

        if not tool_calls and not tool_results:
            return

        if any(not str(call.get("tool_name", "") or "").strip() for call in tool_calls):
            self._add(penalties, reasons, -0.25, "missing_tool_name")

        failed_results = [
            tool_result
            for tool_result in tool_results
            if not bool(tool_result.get("ok", False))
        ]
        if failed_results:
            if len(failed_results) == len(tool_results):
                self._add(penalties, reasons, -0.5, "all_tool_calls_failed")
            else:
                self._add(penalties, reasons, -0.25, "tool_failure")

        if any(self._result_text_indicates_failure(tool_result) for tool_result in tool_results):
            self._add(penalties, reasons, -0.25, "tool_result_indicates_failure")

        if self._final_answer_after_failed_tool(result, failed_results):
            self._add(penalties, reasons, -0.5, "tool_failure_before_final_answer")

    def _final_answer_after_failed_tool(
        self,
        result: AgentReasoningSummary,
        failed_results: list[dict[str, Any]],
    ) -> bool:
        """
        判斷 Agent 是否在工具失敗後仍產生 final answer。

        Args:
            - result: 單一 Agent 的 Stage1 聚合結果。
            - failed_results: ok=False 的工具結果清單。

        Returns:
            - bool: 若有成功解析答案的 run 依賴失敗工具結果則回傳 True。
        """
        if not failed_results:
            return False
        for run in result.runs:
            if run.parse_completed and run.final_answer.strip() and run.tool_results:
                if any(not bool(tool_result.get("ok", False)) for tool_result in run.tool_results):
                    return True
        return False

    def _result_text_indicates_failure(self, tool_result: dict[str, Any]) -> bool:
        """
        從工具輸出文字與 error 欄位判斷是否包含失敗訊號。

        Args:
            - tool_result: 單次工具執行結果。

        Returns:
            - bool: 若文字包含 no result、failed、timeout 等失敗訊號則回傳 True。
        """
        text = " ".join(
            str(tool_result.get(key, "") or "").lower()
            for key in ("output_text", "error")
        )
        failure_markers = (
            "no result",
            "not found",
            "failed",
            "error",
            "timeout",
            "exception",
            "unavailable",
        )
        return any(marker in text for marker in failure_markers)

    def _add(
        self,
        penalties: list[float],
        reasons: list[str],
        penalty: float,
        reason: str,
    ) -> None:
        """
        加入一筆 penalty 數值與去重後的 reason。

        Args:
            - penalties: 累積 penalty 數值的清單。
            - reasons: 累積 penalty reason 的清單。
            - penalty: 本次要加入的 penalty 數值。
            - reason: 本次 penalty 的原因代碼。

        Returns:
            - None。
        """
        penalties.append(penalty)
        if reason not in reasons:
            reasons.append(reason)

    def _clamp(self, penalty: float) -> float:
        """
        將 penalty 限制在系統允許的分數範圍內。

        Args:
            - penalty: 原始 penalty 加總值。

        Returns:
            - float: clamp 到 MIN_PENALTY 與 MAX_PENALTY 之間的 penalty。
        """
        return max(self.MIN_PENALTY, min(self.MAX_PENALTY, penalty))


__all__ = ["PenaltyCalculator"]
