from __future__ import annotations

import json

from core.config import AgentConfig, AgentReasoningSummary, VerifierScoreByReasoning
from parsers.reasoning_parser import extract_reasoning_steps
from score.versa_prm_scorer import (
    DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    DEFAULT_VERSA_PRM_MODEL_ID,
    VersaPRMScorer,
)


class Stage2Runner:
    """
    使用 VersaPRM 作為 Stage2 process verifier，評估每個 Agent reasoning step。

    Args:
     - question: 原始任務問題。
     - agents: Network 中的 AgentConfig 清單，用於 metadata 相容。
     - verifier_mode: Stage2 verifier 模式；目前只支援 versa。
     - versa_prm_model: VersaPRM 模型 id。
     - versa_prm_base_model: VersaPRM base model id。
     - versa_prm_device: auto、cuda 或 cpu。
     - versa_prm_dtype: auto、float16、bfloat16 或 float32。
     - versa_prm_local_files_only: 是否只使用 Hugging Face 本地 cache。
     - versa_scorer: 可注入的 VersaPRMScorer 實例，方便測試。

    Returns:
     - Stage2Runner: 只使用 VersaPRM 產生 step reward probabilities 的 runner。
    """

    def __init__(
        self,
        *,
        question: str,
        agents: list[AgentConfig],
        verifier_mode: str = "versa",
        versa_prm_model: str = DEFAULT_VERSA_PRM_MODEL_ID,
        versa_prm_base_model: str = DEFAULT_VERSA_PRM_BASE_MODEL_ID,
        versa_prm_device: str = "auto",
        versa_prm_dtype: str = "auto",
        versa_prm_local_files_only: bool = False,
        versa_scorer: VersaPRMScorer | None = None,
    ) -> None:
        if verifier_mode != "versa":
            raise ValueError("Only verifier_mode='versa' is supported.")

        self.question = question
        self.agents = agents
        self.verifier_mode = verifier_mode
        self.versa_prm_model = versa_prm_model
        self.versa_prm_base_model = versa_prm_base_model
        self.versa_prm_device = versa_prm_device
        self.versa_prm_dtype = versa_prm_dtype
        self.versa_prm_local_files_only = versa_prm_local_files_only
        self.versa_scorer = versa_scorer or VersaPRMScorer(
            model_id=self.versa_prm_model,
            base_model_id=self.versa_prm_base_model,
            device=self.versa_prm_device,
            dtype=self.versa_prm_dtype,
            local_files_only=self.versa_prm_local_files_only,
        )

    def run(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> list[VerifierScoreByReasoning]:
        """
        對所有 active Stage1 candidates 執行 VersaPRM step-level verification。

        Args:
         - stage1_results: Stage1Runner 輸出的 Agent reasoning summaries。

        Returns:
         - list[VerifierScoreByReasoning]: 每個 active Agent 一筆 VersaPRM 評估結果。
        """
        return self.run_versa_prm(stage1_results)

    def run_versa_prm(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> list[VerifierScoreByReasoning]:
        """
        使用 VersaPRM 評估 active Agent 的 compressed reasoning。

        Args:
         - stage1_results: Stage1Runner 輸出的 Agent reasoning summaries。

        Returns:
         - list[VerifierScoreByReasoning]: target Agent 對應的 VersaPRM reward 結果。
        """
        results: list[VerifierScoreByReasoning] = []
        for target in stage1_results:
            if target.active:
                results.append(self.score_candidate(target))
        return results

    def score_candidate(
        self,
        target: AgentReasoningSummary,
    ) -> VerifierScoreByReasoning:
        """
        使用 VersaPRM 評估單一 Agent reasoning summary。

        Args:
         - target: 要被評估的 AgentReasoningSummary。

        Returns:
         - VerifierScoreByReasoning: 包含平均 reward probability 與每步 reward。
        """
        reasoning_steps = self._reasoning_steps(target.compressed_reasoning)
        score_result = self.versa_scorer.score_steps(
            question=self.question,
            reasoning_steps=reasoning_steps,
        )
        step_scores = [
            {
                "step": item.step_index,
                "step_index": item.step_index,
                "step_text": item.step_text,
                "reward_probability": item.reward_probability,
            }
            for item in score_result.step_scores
        ]
        return VerifierScoreByReasoning(
            verifier_id="versa_prm",
            target_agent_id=target.agent_id,
            verifier_score=score_result.avg_reward_probability,
            step_scores=step_scores,
            raw_reply=json.dumps(score_result.to_dict(), ensure_ascii=False),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def worker_count(self, stage1_results: list[AgentReasoningSummary]) -> int:
        """
        回傳 Stage2 active candidate 數量，用於 metadata 相容。

        Args:
         - stage1_results: Stage1Runner 輸出的 Agent reasoning summaries。

        Returns:
         - int: active Agent 數量；至少為 1。
        """
        active_count = sum(1 for result in stage1_results if result.active)
        return max(1, active_count)

    def _reasoning_steps(self, reasoning: str) -> list[tuple[int, str]]:
        steps = extract_reasoning_steps(reasoning)
        if steps:
            return steps
        text = " ".join(str(reasoning or "").strip().split())
        return [(1, text)] if text else []


__all__ = ["Stage2Runner"]
