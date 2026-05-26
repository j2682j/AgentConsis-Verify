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
        valid_runs = [run for run in runs if run.parse_completed and run.final_answer.strip()]
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
        )

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
