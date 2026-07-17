from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidateEvaluation,
    CandidateRun,
    VerifierScoreByReasoning,
)
from parsers.reasoning_parser import extract_reasoning_steps
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.answer_validator import AnswerValidator
from score.evidence_support_checker import EvidenceSupportChecker


@dataclass
class FinalWinnerSelection:
    """保存 final winner 與完整候選比較軌跡。"""

    winner: AgentReasoningSummary | None
    evaluation: CandidateEvaluation | None
    evaluations: list[CandidateEvaluation]
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": "candidate_centric_hierarchical",
            "status": self.status,
            "selection_reason": self.reason,
            "selected_answer": self.evaluation.answer if self.evaluation else "",
            "selected_candidate_key": (
                self.evaluation.candidate_key if self.evaluation else ""
            ),
            "selected_agent_id": (
                self.evaluation.selected_agent_id if self.evaluation else ""
            ),
            "selected_run_index": (
                self.evaluation.selected_run_index if self.evaluation else 0
            ),
            "candidates": [asdict(item) for item in self.evaluations],
        }


class FinalWinnerSelector:
    """
    以資格、證據層級、跨 Agent 共識與 Versa 關鍵步驟依序選擇答案。

    Args:
     - clusterer: 建立及還原候選答案的 AnswerCandidateClusterer。
     - evidence_support_checker: 驗證工具、附件與搜尋證據的 checker。

    Returns:
     - FinalWinnerSelector: 不使用 weighted score 的分層 winner selector。
    """

    SUPPORT_TIERS = {
        "contradicted": -1,
        "tool_failed_model_only": 0,
        "no_support": 1,
        "tool_intermediate_supported": 2,
        "search_evidence_supported": 3,
        "attachment_evidence_supported": 3,
        "tool_final_supported": 4,
    }
    DIRECT_SUPPORT_STATUSES = {
        "search_evidence_supported",
        "attachment_evidence_supported",
        "tool_final_supported",
    }

    def __init__(
        self,
        *,
        clusterer: AnswerCandidateClusterer | None = None,
        answer_validator: AnswerValidator | None = None,
        evidence_support_checker: EvidenceSupportChecker | None = None,
    ) -> None:
        self.answer_validator = answer_validator or AnswerValidator()
        self.clusterer = clusterer or AnswerCandidateClusterer(self.answer_validator)
        self.evidence_support_checker = (
            evidence_support_checker or EvidenceSupportChecker(self.answer_validator)
        )

    def select(
        self,
        *,
        stage1_results: list[AgentReasoningSummary],
        candidates: list[AnswerCandidate],
        verifier_results: list[VerifierScoreByReasoning],
        evidence: dict[str, Any],
    ) -> FinalWinnerSelection:
        """
        評估所有候選並以固定優先序選出最終答案與代表推理路徑。

        Args:
         - stage1_results: 原始 Stage1 Agent summaries。
         - candidates: 由全部有效 runs 建立的答案群組。
         - verifier_results: 帶 candidate/run metadata 的 Versa 結果。
         - evidence: Evidence Prepare 與 Stage1 tool evidence。

        Returns:
         - FinalWinnerSelection: winner、候選評估與選擇原因。
        """
        evaluations = [
            self._evaluate_candidate(
                candidate=candidate,
                stage1_results=stage1_results,
                verifier_results=verifier_results,
                evidence=evidence,
            )
            for candidate in candidates
        ]
        eligible = [item for item in evaluations if item.eligible]
        if not eligible:
            return FinalWinnerSelection(
                winner=None,
                evaluation=None,
                evaluations=evaluations,
                status="no_eligible_candidate",
                reason="all_candidates_invalid_or_contradicted",
            )

        ranked = sorted(eligible, key=self.rank_tuple, reverse=True)
        selected = ranked[0]
        tied = [
            item for item in ranked if self.rank_tuple(item) == self.rank_tuple(selected)
        ]
        if len({item.candidate_key for item in tied}) > 1:
            return FinalWinnerSelection(
                winner=None,
                evaluation=selected,
                evaluations=ranked,
                status="review_required",
                reason="top_candidates_tied_after_hierarchical_selection",
            )
        if self._requires_search_support(evidence) and selected.support_tier < 2:
            return FinalWinnerSelection(
                winner=None,
                evaluation=selected,
                evaluations=sorted(eligible, key=self.rank_tuple, reverse=True),
                status="unresolved",
                reason="factual_search_has_no_supported_candidate",
            )

        selected_member = self._member_by_identity(
            candidates,
            candidate_key=selected.candidate_key,
            agent_id=selected.selected_agent_id,
            run_index=selected.selected_run_index,
        )
        if selected_member is None:
            return FinalWinnerSelection(
                winner=None,
                evaluation=selected,
                evaluations=evaluations,
                status="selected_member_missing",
                reason="candidate_member_could_not_be_restored",
            )
        winner = self.clusterer.summary_for_member(stage1_results, selected_member)
        return FinalWinnerSelection(
            winner=winner,
            evaluation=selected,
            evaluations=sorted(evaluations, key=self.rank_tuple, reverse=True),
            status="answerable",
            reason=self._selection_reason(selected),
        )

    def rank_tuple(self, evaluation: CandidateEvaluation) -> tuple[Any, ...]:
        """回傳不含加權相加的候選字典序比較鍵值。"""
        return (
            int(evaluation.support_tier),
            bool(evaluation.direct_support),
            len(set(evaluation.supporting_agent_ids)),
            int(evaluation.supporting_run_count),
            float(evaluation.critical_step_floor),
            float(evaluation.critical_step_geometric_mean),
            int(evaluation.selected_agent_answer_frequency),
            float(evaluation.selected_agent_confidence),
        )

    def _evaluate_candidate(
        self,
        *,
        candidate: AnswerCandidate,
        stage1_results: list[AgentReasoningSummary],
        verifier_results: list[VerifierScoreByReasoning],
        evidence: dict[str, Any],
    ) -> CandidateEvaluation:
        member_evaluations = [
            self._evaluate_member(
                member=member,
                stage1_results=stage1_results,
                verifier_results=verifier_results,
                evidence=evidence,
            )
            for member in candidate.members
        ]
        valid_members = [item for item in member_evaluations if item["eligible"]]
        if not valid_members:
            contradicted = any(item["contradicted"] for item in member_evaluations)
            return CandidateEvaluation(
                candidate_key=candidate.candidate_key,
                answer=candidate.representative_answer,
                eligible=False,
                rejection_reason=(
                    "candidate_contradicted" if contradicted else "no_valid_candidate_path"
                ),
                support_tier=-1 if contradicted else 0,
                support_status="contradicted" if contradicted else "invalid",
                contradicted=contradicted,
                supporting_agent_ids=candidate.supporting_agent_ids,
                supporting_run_count=candidate.supporting_run_count,
                metadata={"member_evaluations": member_evaluations},
            )

        selected_member = max(valid_members, key=self._member_rank_tuple)
        requirement = self._answer_requirement(evidence)
        requirement_status = "not_available"
        if requirement:
            requirement_status = (
                "supported_by_direct_evidence"
                if selected_member["direct_support"]
                else "not_directly_verified"
            )
        return CandidateEvaluation(
            candidate_key=candidate.candidate_key,
            answer=candidate.representative_answer,
            eligible=True,
            support_tier=int(selected_member["support_tier"]),
            support_status=str(selected_member["support_status"]),
            direct_support=bool(selected_member["direct_support"]),
            contradicted=False,
            requirement_status=requirement_status,
            supporting_agent_ids=candidate.supporting_agent_ids,
            supporting_run_count=candidate.supporting_run_count,
            selected_agent_id=str(selected_member["agent_id"]),
            selected_run_index=int(selected_member["run_index"]),
            selected_reasoning=str(selected_member["reasoning"]),
            selected_agent_confidence=float(selected_member["agent_confidence"]),
            selected_agent_answer_frequency=int(
                selected_member["agent_answer_frequency"]
            ),
            critical_step_floor=float(selected_member["critical_step_floor"]),
            critical_step_geometric_mean=float(
                selected_member["critical_step_geometric_mean"]
            ),
            average_verifier_probability=float(
                selected_member["average_verifier_probability"]
            ),
            metadata={"member_evaluations": member_evaluations},
        )

    def _evaluate_member(
        self,
        *,
        member: CandidateRun,
        stage1_results: list[AgentReasoningSummary],
        verifier_results: list[VerifierScoreByReasoning],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        summary = self.clusterer.summary_for_member(stage1_results, member)
        reasoning_steps = extract_reasoning_steps(member.reasoning)
        if not reasoning_steps and member.reasoning.strip():
            reasoning_steps = [(1, member.reasoning.strip())]
        support = self.evidence_support_checker.check_agent(
            target=summary,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
        )
        verifier = self._find_verifier_result(member, verifier_results)
        process = self._process_metadata(verifier)
        support_status = support.status
        verifier_support = self._verifier_support_metadata(verifier)
        if verifier_support:
            support_status = str(verifier_support.get("status") or support_status)
        support_tier = self.SUPPORT_TIERS.get(support_status, 0)
        contradicted = support_status == "contradicted"
        return {
            "agent_id": member.agent_id,
            "run_index": member.run_index,
            "answer": member.answer,
            "reasoning": member.reasoning,
            "eligible": bool(self.answer_validator.is_valid(member.answer) and not contradicted),
            "contradicted": contradicted,
            "support_tier": support_tier,
            "support_status": support_status,
            "support_priority": int(
                verifier_support.get("priority", support.priority)
                if verifier_support
                else support.priority
            ),
            "direct_support": support_status in self.DIRECT_SUPPORT_STATUSES,
            "agent_confidence": member.agent_confidence,
            "agent_answer_frequency": member.agent_answer_frequency,
            "critical_step_floor": float(process.get("critical_step_floor") or 0.0),
            "critical_step_geometric_mean": float(
                process.get("critical_step_geometric_mean") or 0.0
            ),
            "average_verifier_probability": float(
                process.get("average_probability")
                or (verifier.verifier_score if verifier is not None else 0.0)
            ),
            "critical_step_indices": list(process.get("critical_step_indices") or []),
            "evidence_support": self.evidence_support_checker.summary_to_dict(support),
        }

    def _member_rank_tuple(self, item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(item.get("support_tier") or 0),
            bool(item.get("direct_support")),
            float(item.get("critical_step_floor") or 0.0),
            float(item.get("critical_step_geometric_mean") or 0.0),
            int(item.get("agent_answer_frequency") or 0),
            float(item.get("agent_confidence") or 0.0),
        )

    def _find_verifier_result(
        self,
        member: CandidateRun,
        verifier_results: list[VerifierScoreByReasoning],
    ) -> VerifierScoreByReasoning | None:
        exact: list[VerifierScoreByReasoning] = []
        legacy: list[VerifierScoreByReasoning] = []
        for result in verifier_results:
            if result.target_agent_id != member.agent_id:
                continue
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            candidate_key = str(metadata.get("candidate_key") or "")
            run_index = int(metadata.get("target_run_index") or 0)
            if candidate_key == member.normalized_answer and run_index == member.run_index:
                exact.append(result)
            elif not candidate_key:
                legacy.append(result)
        return exact[0] if exact else (legacy[0] if legacy else None)

    def _process_metadata(
        self,
        result: VerifierScoreByReasoning | None,
    ) -> dict[str, Any]:
        if result is None or not isinstance(result.metadata, dict):
            return {}
        process = result.metadata.get("process_verification")
        return process if isinstance(process, dict) else {}

    def _verifier_support_metadata(
        self,
        result: VerifierScoreByReasoning | None,
    ) -> dict[str, Any]:
        if result is None or not isinstance(result.metadata, dict):
            return {}
        support = result.metadata.get("evidence_support")
        return support if isinstance(support, dict) else {}

    def _member_by_identity(
        self,
        candidates: list[AnswerCandidate],
        *,
        candidate_key: str,
        agent_id: str,
        run_index: int,
    ) -> CandidateRun | None:
        for candidate in candidates:
            if candidate.candidate_key != candidate_key:
                continue
            for member in candidate.members:
                if member.agent_id == agent_id and member.run_index == run_index:
                    return member
        return None

    def _requires_search_support(self, evidence: dict[str, Any]) -> bool:
        routing = evidence.get("routing") if isinstance(evidence.get("routing"), dict) else {}
        route = str(routing.get("primary_route") or "").strip().lower()
        return bool(route == "factual_search" and str(evidence.get("search_result") or "").strip())

    def _answer_requirement(self, evidence: dict[str, Any]) -> str:
        direct = str(evidence.get("answer_requirement") or "").strip()
        if direct:
            return direct
        for item in evidence.get("tool_usage", []) or []:
            if not isinstance(item, dict) or item.get("tool_name") != "search":
                continue
            raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
            diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
            contract = diagnostics.get("evidence_selection_contract")
            if isinstance(contract, dict):
                requirement = str(contract.get("answer_requirement") or "").strip()
                if requirement:
                    return requirement
            for evidence_item in raw.get("evidence_items", []) or []:
                if not isinstance(evidence_item, dict):
                    continue
                requirement = str(evidence_item.get("answer_requirement") or "").strip()
                if requirement:
                    return requirement
        return ""

    def _selection_reason(self, evaluation: CandidateEvaluation) -> str:
        if evaluation.support_tier >= 4:
            return "trusted_tool_final_answer"
        if evaluation.support_tier >= 3 and len(evaluation.supporting_agent_ids) >= 2:
            return "direct_evidence_cross_agent_consensus"
        if evaluation.support_tier >= 3:
            return "direct_evidence_supported_candidate"
        if evaluation.support_tier >= 2:
            return "bridge_evidence_supported_reasoning"
        if len(evaluation.supporting_agent_ids) >= 2:
            return "cross_agent_consensus"
        return "best_valid_model_only_candidate"


__all__ = ["FinalWinnerSelection", "FinalWinnerSelector"]
