from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any

from core.config import AgentConfig, AgentReasoningSummary, VerifierScoreByReasoning
from parsers.reasoning_parser import extract_reasoning_steps
from score.evidence_support_checker import EvidenceSupportChecker
from score.answer_validator import AnswerValidator
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
        versa_prm_local_files_only: bool = True,
        versa_scorer: VersaPRMScorer | None = None,
        evidence_support_checker: EvidenceSupportChecker | None = None,
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
        self.evidence_support_checker = (
            evidence_support_checker or EvidenceSupportChecker()
        )
        self.answer_validator = AnswerValidator()

    def run(
        self,
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any] | None = None,
    ) -> list[VerifierScoreByReasoning]:
        """
        對所有 active Stage1 candidates 執行 VersaPRM step-level verification。

        Args:
         - stage1_results: Stage1Runner 輸出的 Agent reasoning summaries。

        Returns:
         - list[VerifierScoreByReasoning]: 每個 active Agent 一筆 VersaPRM 評估結果。
        """
        return self.run_versa_prm(stage1_results, evidence=evidence)

    def run_versa_prm(
        self,
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any] | None = None,
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
                results.append(self.score_candidate(target, evidence=evidence))
        return results

    def run_candidate_paths(
        self,
        stage1_results: list[AgentReasoningSummary],
        *,
        candidate_key_builder: Any,
        evidence: dict[str, Any] | None = None,
    ) -> list[VerifierScoreByReasoning]:
        """
        對每個有效 Stage1 run 的候選答案與推理路徑分別執行 Versa 評分。

        Args:
         - stage1_results: 保留原始 runs 的 Stage1 Agent summaries。
         - candidate_key_builder: 將 final answer 轉成候選分群鍵值的函式。
         - evidence: Evidence Prepare 與 Stage1 tools 的證據集合。

        Returns:
         - list[VerifierScoreByReasoning]: 帶有 candidate key 與 run index 的評分。
        """
        results: list[VerifierScoreByReasoning] = []
        for target in stage1_results:
            if not target.active:
                continue
            for run in target.runs:
                answer = self.answer_validator.clean(run.final_answer)
                candidate_key = candidate_key_builder(answer)
                if not (
                    candidate_key
                    and run.parse_completed
                    and getattr(run, "eligible_for_winner", True)
                    and self.answer_validator.is_valid(answer)
                ):
                    continue
                candidate_target = replace(
                    target,
                    runs=[run],
                    compressed_answer=answer,
                    compressed_reasoning=str(run.reasoning or "").strip(),
                )
                results.append(
                    self.score_candidate(
                        candidate_target,
                        evidence=evidence,
                        candidate_key=candidate_key,
                        target_run_index=int(run.run_index),
                    )
                )
        return results

    def score_candidate(
        self,
        target: AgentReasoningSummary,
        evidence: dict[str, Any] | None = None,
        *,
        candidate_key: str = "",
        target_run_index: int = 0,
    ) -> VerifierScoreByReasoning:
        """
        使用 VersaPRM 評估單一 Agent reasoning summary。

        Args:
         - target: 要被評估的 AgentReasoningSummary。

        Returns:
         - VerifierScoreByReasoning: 包含平均 reward probability 與每步 reward。
        """
        reasoning_steps = self._reasoning_steps(target.compressed_reasoning)
        support_summary = self.evidence_support_checker.check_agent(
            target=target,
            reasoning_steps=reasoning_steps,
            evidence=evidence or {},
            question=self.question,
        )
        support_by_step = {
            item.step_index: item for item in support_summary.step_results
        }
        score_result = self.versa_scorer.score_steps(
            question=self.question,
            reasoning_steps=reasoning_steps,
        )
        step_scores = []
        for item in score_result.step_scores:
            support = support_by_step.get(item.step_index)
            step_scores.append(
                {
                    "step": item.step_index,
                    "step_index": item.step_index,
                    "step_text": item.step_text,
                    "reward_probability": item.reward_probability,
                    "support_status": (
                        support.status if support is not None else "unsupported"
                    ),
                    "support_reason": support.reason if support is not None else "",
                    "support_source_tools": (
                        list(support.source_tools) if support is not None else []
                    ),
                    "matched_tool_values": (
                        list(support.matched_tool_values)
                        if support is not None
                        else []
                    ),
                    "support_metadata": (
                        dict(support.metadata) if support is not None else {}
                    ),
                }
            )
        process_verification = self._process_verification_summary(
            step_scores=step_scores,
            final_answer=target.compressed_answer,
        )
        support_payload = self.evidence_support_checker.summary_to_dict(
            support_summary
        )
        return VerifierScoreByReasoning(
            verifier_id="versa_prm",
            target_agent_id=target.agent_id,
            verifier_score=score_result.avg_reward_probability,
            step_scores=step_scores,
            raw_reply=json.dumps(
                {
                    "versa_prm": score_result.to_dict(),
                    "evidence_support": support_payload,
                    "process_verification": process_verification,
                },
                ensure_ascii=False,
            ),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            metadata={
                "evidence_support": support_payload,
                "process_verification": process_verification,
                "candidate_key": candidate_key,
                "target_run_index": int(target_run_index or 0),
                "target_answer": target.compressed_answer,
            },
        )

    def _process_verification_summary(
        self,
        *,
        step_scores: list[dict[str, Any]],
        final_answer: str,
    ) -> dict[str, Any]:
        """
        從 Versa step probabilities 建立關鍵步驟最低值與幾何平均。

        Args:
         - step_scores: 已結合 evidence support 狀態的逐步評分。
         - final_answer: 此推理路徑對應的候選答案。

        Returns:
         - dict[str, Any]: 關鍵步驟索引、最低值、幾何平均與整體平均。
        """
        if not step_scores:
            return {
                "critical_step_indices": [],
                "critical_step_floor": 0.0,
                "critical_step_geometric_mean": 0.0,
                "average_probability": 0.0,
            }
        answer = self.answer_validator.clean(final_answer).casefold()
        critical_indices: set[int] = {int(step_scores[-1].get("step_index") or 0)}
        for item in step_scores:
            step_index = int(item.get("step_index") or 0)
            step_text = self.answer_validator.clean(item.get("step_text", "")).casefold()
            if item.get("support_status") in {"supported", "contradicted"}:
                critical_indices.add(step_index)
            if answer and answer in step_text:
                critical_indices.add(step_index)
        critical_probabilities = [
            max(0.0, min(1.0, float(item.get("reward_probability") or 0.0)))
            for item in step_scores
            if int(item.get("step_index") or 0) in critical_indices
        ]
        all_probabilities = [
            max(0.0, min(1.0, float(item.get("reward_probability") or 0.0)))
            for item in step_scores
        ]
        floor = min(critical_probabilities) if critical_probabilities else 0.0
        geometric_mean = (
            math.exp(
                sum(math.log(max(probability, 1e-12)) for probability in critical_probabilities)
                / len(critical_probabilities)
            )
            if critical_probabilities
            else 0.0
        )
        return {
            "critical_step_indices": sorted(critical_indices),
            "critical_step_floor": floor,
            "critical_step_geometric_mean": geometric_mean,
            "average_probability": (
                sum(all_probabilities) / len(all_probabilities)
                if all_probabilities
                else 0.0
            ),
        }

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

    def unload(self) -> dict:
        """
        Release the lazy-loaded VersaPRM scorer after a task finishes scoring.

        Args:
         - None.

        Returns:
         - dict: VersaPRM unload status.
        """
        unload = getattr(self.versa_scorer, "unload", None)
        if not callable(unload):
            return {"was_loaded": False, "warning": "versa_scorer has no unload()"}
        return dict(unload())

    def _reasoning_steps(self, reasoning: str) -> list[tuple[int, str]]:
        steps = extract_reasoning_steps(reasoning)
        if steps:
            return steps
        text = " ".join(str(reasoning or "").strip().split())
        return [(1, text)] if text else []


__all__ = ["Stage2Runner"]
