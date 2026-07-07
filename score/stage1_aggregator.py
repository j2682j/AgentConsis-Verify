from __future__ import annotations

from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply
from parsers.reasoning_parser import compress_reasoning
from utils.network_utils import answer_equivalence, normalize_for_exact


class Stage1Aggregator:
    """
    聚合同一 Agent 的多次 Stage1 回覆，選出最一致的答案並計算 confidence score。

    Args:
        - 無。

    Returns:
        - Stage1Aggregator: 負責 Stage1 self-consistency 聚合的物件。
    """

    ABSTENTION_LABELS = {
        "empty_final_answer",
        "refusal_like_final_answer",
        "uncertain_final_answer",
        "tool_request_pending",
    }
    INVALID_LABELS = {
        "invalid_final_answer",
        "invalid_tool_reply",
        "parse_exception",
        "schema_invalid",
        "tool_call_as_final_answer",
        "tool_trajectory_no_final_answer",
        "too_verbose_final_answer",
    }

    def summarize(
        self,
        config: AgentConfig,
        runs: list[EachAgentReply],
    ) -> AgentReasoningSummary:
        """
        根據等價答案分組，產生單一 Agent 的 Stage1 聚合摘要。

        Args:
            - config: 要聚合的 AgentConfig。
            - runs: 該 Agent 的多次 Stage1 raw replies。

        Returns:
            - AgentReasoningSummary: 包含最佳答案群、壓縮 reasoning、confidence 與 active 狀態。
        """
        validity = self.summarize_run_validity(runs)
        valid_runs = [
            run
            for run in runs
            if run.parse_completed
            and run.final_answer.strip()
            and getattr(run, "eligible_for_winner", True)
        ]
        if not valid_runs:
            config.confidence_score = 0.0
            return AgentReasoningSummary(
                agent_id=config.agent_id,
                model_name=config.model_name,
                runs=runs,
                compressed_answer="",
                compressed_reasoning="",
                confidence_score=0.0,
                active=False,
                **validity,
            )

        groups = self.group_runs_by_equivalent_answer(valid_runs)
        best_group = max(groups, key=len)
        confidence_score = self.confidence_from_match_count(len(best_group))

        config.confidence_score = confidence_score
        return AgentReasoningSummary(
            agent_id=config.agent_id,
            model_name=config.model_name,
            runs=runs,
            compressed_answer=best_group[0].final_answer,
            compressed_reasoning=compress_reasoning(best_group),
            confidence_score=confidence_score,
            active=True,
            **validity,
        )

    def summarize_run_validity(self, runs: list[EachAgentReply]) -> dict:
        """
        Summarize Stage1 run validity without assigning numeric scores.
        """
        labels = self._run_validity_labels(runs)
        eligible_runs = [
            run
            for run in runs
            if run.parse_completed
            and run.final_answer.strip()
            and getattr(run, "eligible_for_winner", True)
        ]
        abstention_runs = [
            run
            for run in runs
            if self._has_any_label(run, self.ABSTENTION_LABELS)
        ]
        invalid_runs = [
            run
            for run in runs
            if (
                not getattr(run, "schema_valid", True)
                or self._has_any_label(run, self.INVALID_LABELS)
            )
            and not self._has_any_label(run, self.ABSTENTION_LABELS)
        ]

        eligible_count = len(eligible_runs)
        abstention_count = len(abstention_runs)
        invalid_count = len(invalid_runs)
        valid_count = eligible_count
        winner_status = self._winner_selection_status(
            total_run_count=len(runs),
            eligible_run_count=eligible_count,
            abstention_run_count=abstention_count,
            invalid_run_count=invalid_count,
        )
        return {
            "valid_run_count": valid_count,
            "invalid_run_count": invalid_count,
            "abstention_run_count": abstention_count,
            "eligible_run_count": eligible_count,
            "run_validity_labels": labels,
            "winner_selection_eligible": eligible_count >= 1,
            "winner_selection_status": winner_status,
        }

    def _winner_selection_status(
        self,
        *,
        total_run_count: int,
        eligible_run_count: int,
        abstention_run_count: int,
        invalid_run_count: int,
    ) -> str:
        if eligible_run_count >= 2:
            return "answerable"
        if eligible_run_count == 1:
            return "mixed_low_coverage" if total_run_count > 1 else "answerable"
        if total_run_count <= 0:
            return "no_stage1_runs"
        if abstention_run_count >= total_run_count:
            return "all_runs_abstained"
        if invalid_run_count >= total_run_count:
            return "all_runs_invalid"
        return "no_final_answer"

    def _run_validity_labels(self, runs: list[EachAgentReply]) -> list[str]:
        seen: set[str] = set()
        labels: list[str] = []
        for run in runs:
            for label in getattr(run, "validity_labels", []) or []:
                label = str(label or "").strip()
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)
            for error in getattr(run, "schema_errors", []) or []:
                if error and "schema_invalid" not in seen:
                    seen.add("schema_invalid")
                    labels.append("schema_invalid")
        return labels

    def _has_any_label(self, run: EachAgentReply, labels: set[str]) -> bool:
        run_labels = {
            str(label or "").strip()
            for label in getattr(run, "validity_labels", []) or []
            if str(label or "").strip()
        }
        if not getattr(run, "schema_valid", True):
            run_labels.add("schema_invalid")
        return bool(run_labels & labels)

    def group_runs_by_equivalent_answer(
        self,
        runs: list[EachAgentReply],
    ) -> list[list[EachAgentReply]]:
        """
        將多次 Stage1 回覆依答案等價性分組。

        Args:
            - runs: 已成功解析 final answer 的 Stage1 replies。

        Returns:
            - list[list[EachAgentReply]]: 每個子清單代表一組等價答案。
        """
        groups: list[list[EachAgentReply]] = []
        for run in runs:
            for group in groups:
                if self.answers_equivalent(run.final_answer, group[0].final_answer):
                    group.append(run)
                    break
            else:
                groups.append([run])
        return groups

    def answers_equivalent(self, answer_a: str, answer_b: str) -> bool:
        """
        判斷兩個答案是否可視為等價，優先使用 exact normalization，再使用語意等價判斷。

        Args:
            - answer_a: 第一個候選答案。
            - answer_b: 第二個候選答案。

        Returns:
            - bool: 若兩個答案等價則回傳 True。
        """
        if normalize_for_exact(answer_a) == normalize_for_exact(answer_b):
            return True
        try:
            return answer_equivalence(answer_a, answer_b)
        except Exception:
            return False

    def confidence_from_match_count(self, match_count: int) -> float:
        """
        將同答案群的 run 數轉成 Stage1 confidence score。

        Args:
            - match_count: 最佳答案群中的 run 數量。

        Returns:
            - float: 3 次以上為 1.0，2 次為 0.67，其餘為 0.33。
        """
        if match_count >= 3:
            return 1.0
        if match_count == 2:
            return 0.67
        return 0.33


__all__ = ["Stage1Aggregator"]
