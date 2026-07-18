from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidateRun,
)
from parsers.reasoning_parser import prepare_reasoning_for_verifier
from score.answer_validator import AnswerValidator
from utils.network_utils import normalize_for_exact


class AnswerCandidateClusterer:
    """
    從所有有效 Stage1 runs 建立跨 Agent 的等價答案候選群組。

    Args:
     - answer_validator: 清理並驗證 final answer 的 AnswerValidator。

    Returns:
     - AnswerCandidateClusterer: 提供候選分群與代表推理路徑重建功能。
    """

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def cluster(
        self,
        stage1_results: list[AgentReasoningSummary],
    ) -> list[AnswerCandidate]:
        """
        收集全部有效 run，並以保守答案正規化結果進行跨 Agent 分群。

        Args:
         - stage1_results: 每個 Agent 的 Stage1 聚合結果與原始 runs。

        Returns:
         - list[AnswerCandidate]: 依首次出現順序排列的候選答案群組。
        """
        raw_members: list[CandidateRun] = []
        per_agent_keys: dict[str, list[str]] = {}
        eligible_counts: dict[str, int] = {}

        for summary in stage1_results:
            if not summary.active and not summary.runs:
                continue
            valid_runs = [run for run in summary.runs if self._valid_run(run)]
            eligible_counts[summary.agent_id] = len(valid_runs)
            per_agent_keys[summary.agent_id] = [
                self.candidate_key(
                    run.final_answer,
                    answer_type=self._answer_type(run),
                )
                for run in valid_runs
            ]
            for run in valid_runs:
                answer = self.answer_validator.clean(run.final_answer)
                answer_type = self._answer_type(run)
                key = self.candidate_key(answer, answer_type=answer_type)
                if not key:
                    continue
                saved_steps = list(getattr(run, "reasoning_steps", []) or [])
                reasoning_parse = None
                if not saved_steps:
                    structured_steps = (
                        run.structured_output.get("reasoning_steps")
                        if isinstance(run.structured_output, dict)
                        else None
                    )
                    reasoning_parse = prepare_reasoning_for_verifier(
                        str(run.reasoning or ""),
                        final_answer=answer,
                        structured_steps=structured_steps,
                    )
                    saved_steps = list(reasoning_parse.steps)
                raw_members.append(
                    CandidateRun(
                        agent_id=summary.agent_id,
                        model_name=summary.model_name,
                        run_index=int(run.run_index),
                        answer=answer,
                        normalized_answer=key,
                        reasoning=str(run.reasoning or "").strip(),
                        agent_confidence=float(summary.confidence_score or 0.0),
                        answer_type=answer_type,
                        schema_valid=bool(run.schema_valid),
                        parse_completed=bool(run.parse_completed),
                        eligible_for_winner=bool(run.eligible_for_winner),
                        validity_labels=list(run.validity_labels),
                        reasoning_steps=saved_steps,
                        reasoning_parse_quality=str(
                            run.reasoning_parse_quality
                            or (
                                reasoning_parse.quality.value
                                if reasoning_parse is not None
                                else "unreliable"
                            )
                        ),
                        reasoning_versa_eligible=bool(
                            run.reasoning_versa_eligible
                            and (
                                reasoning_parse.versa_eligible
                                if reasoning_parse is not None
                                else bool(saved_steps)
                            )
                        ),
                    )
                )

        counts_by_agent = {
            agent_id: Counter(keys)
            for agent_id, keys in per_agent_keys.items()
        }
        groups: dict[str, AnswerCandidate] = {}
        for member in raw_members:
            member.agent_answer_frequency = counts_by_agent[member.agent_id][
                member.normalized_answer
            ]
            member.eligible_run_count = max(1, eligible_counts.get(member.agent_id, 1))
            candidate = groups.get(member.normalized_answer)
            if candidate is None:
                candidate = AnswerCandidate(
                    candidate_key=member.normalized_answer,
                    representative_answer=member.answer,
                )
                groups[member.normalized_answer] = candidate
            candidate.members.append(member)

        for candidate in groups.values():
            representative = max(
                candidate.members,
                key=lambda member: (
                    member.agent_answer_frequency,
                    -member.run_index,
                    -len(member.answer),
                ),
            )
            candidate.representative_answer = representative.answer
        return list(groups.values())

    def candidate_key(self, answer: str, *, answer_type: str = "") -> str:
        """
        產生保守且可重現的候選答案分群鍵值。

        Args:
         - answer: 原始或已清理的 final answer。

        Returns:
         - str: 正規化答案；無效答案回傳空字串。
        """
        cleaned = self.answer_validator.clean(answer)
        if not cleaned or not self.answer_validator.is_valid(
            cleaned,
            answer_type=answer_type,
        ):
            return ""
        normalized = normalize_for_exact(cleaned).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        if "," in normalized:
            parts = [
                re.sub(r"\s+", " ", part.strip())
                for part in normalized.split(",")
                if part.strip()
            ]
            if len(parts) > 1:
                normalized = ",".join(sorted(parts))
        return normalized

    def summary_for_member(
        self,
        stage1_results: list[AgentReasoningSummary],
        member: CandidateRun,
    ) -> AgentReasoningSummary:
        """
        將候選成員還原成可交給既有 Stage2 與 evidence checker 的 summary。

        Args:
         - stage1_results: 原始 Agent summaries。
         - member: 要還原的候選 run。

        Returns:
         - AgentReasoningSummary: 僅代表指定答案與推理路徑的 summary。
        """
        source = next(
            result for result in stage1_results if result.agent_id == member.agent_id
        )
        selected_run = next(
            (
                run
                for run in source.runs
                if int(run.run_index) == int(member.run_index)
            ),
            None,
        )
        metadata = dict(source.aggregation_metadata or {})
        metadata.update(
            {
                "winner_candidate_key": member.normalized_answer,
                "winner_candidate_run_index": member.run_index,
            }
        )
        return replace(
            source,
            runs=[selected_run] if selected_run is not None else list(source.runs),
            compressed_answer=member.answer,
            compressed_reasoning=member.reasoning,
            confidence_score=self._member_consistency(member),
            aggregation_metadata=metadata,
        )

    def _valid_run(self, run: object) -> bool:
        answer = self.answer_validator.clean(getattr(run, "final_answer", ""))
        return bool(
            answer
            and getattr(run, "parse_completed", False)
            and getattr(run, "schema_valid", False)
            and getattr(run, "eligible_for_winner", True)
        )

    def _answer_type(self, run: object) -> str:
        structured_output = getattr(run, "structured_output", {})
        if not isinstance(structured_output, dict):
            return ""
        return str(structured_output.get("answer_type") or "").strip()

    def _member_consistency(self, member: CandidateRun) -> float:
        ratio = member.agent_answer_frequency / max(1, member.eligible_run_count)
        if ratio >= 1.0:
            return 1.0
        if ratio >= (2 / 3):
            return 0.67
        return 0.33


__all__ = ["AnswerCandidateClusterer"]
