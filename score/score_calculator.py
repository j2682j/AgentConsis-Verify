from __future__ import annotations

from collections import defaultdict

from core.config import AgentConfig, AgentReasoningSummary, JudgeScoreByReasoning


class ScoreCalculator:
    """
    將 Stage1 confidence、Stage2 judge score 與 rule-based penalty 寫回 AgentConfig。

    Args:
        - 無。

    Returns:
        - ScoreCalculator: 負責更新 AgentConfig 分數欄位的計算器。
    """

    def write_scores_to_agent_config(
        self,
        agents: list[AgentConfig],
        stage1_results: list[AgentReasoningSummary],
        judge_results: list[JudgeScoreByReasoning],
        penalty_results: list[dict] | None = None,
    ) -> None:
        """
        彙整 confidence、judge scores 與 penalty，計算每個 Agent 的 total_score。

        Args:
            - agents: 需要寫回分數的 AgentConfig 清單。
            - stage1_results: Stage1 聚合後的候選結果。
            - judge_results: Stage2 judge pair 的評分結果。
            - penalty_results: PenaltyCalculator 產生的 agent-level penalty 結果。

        Returns:
            - None。
        """
        confidence_by_agent = {
            result.agent_id: result.confidence_score for result in stage1_results
        }
        active_by_agent = {result.agent_id: result.active for result in stage1_results}
        scores_by_target: dict[str, list[float]] = defaultdict(list)
        for result in judge_results:
            scores_by_target[result.target_agent_id].append(result.judge_score)
        penalty_by_agent = {
            str(result.get("agent_id", "")): float(result.get("penalty", 0.0) or 0.0)
            for result in penalty_results or []
        }
        penalty_reasons_by_agent = {
            str(result.get("agent_id", "")): list(result.get("reasons", []) or [])
            for result in penalty_results or []
        }

        for config in agents:
            config.confidence_score = confidence_by_agent.get(config.agent_id, 0.0)
            config.judge_scores = list(scores_by_target.get(config.agent_id, []))
            config.avg_judge_score = (
                sum(config.judge_scores) / len(config.judge_scores)
                if config.judge_scores
                else 0.0
            )
            config.penalty_score = penalty_by_agent.get(config.agent_id, 0.0)
            config.penalty_reasons = penalty_reasons_by_agent.get(config.agent_id, [])
            config.total_score = (
                config.confidence_score + config.avg_judge_score + config.penalty_score
                if active_by_agent.get(config.agent_id, False)
                else float("-inf")
            )


__all__ = ["ScoreCalculator"]
