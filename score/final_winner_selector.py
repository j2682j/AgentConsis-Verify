from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidateEvaluation,
    CandidateRun,
    VerifierScoreByReasoning,
)
from parsers.reasoning_parser import extract_reasoning_steps
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.answer_requirement_gate import AnswerRequirementGate
from score.answer_validator import AnswerValidator
from score.evidence_support_checker import EvidenceSupportChecker
from score.gate_result import CandidateGateDecision, GateResult


@dataclass
class FinalWinnerSelection:
    """Save the final winner and the complete ordered-gate trace."""

    winner: AgentReasoningSummary | None
    evaluation: CandidateEvaluation | None
    evaluations: list[CandidateEvaluation]
    status: str
    reason: str
    gate_trace: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": "candidate_ordered_gates",
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
            "gate_trace": [item.to_dict() for item in self.gate_trace],
            "candidates": [asdict(item) for item in self.evaluations],
        }


class FinalWinnerSelector:
    """
    Select a candidate through ordered gates without a weighted aggregate score.

    A rejected candidate never re-enters the pipeline. Evidence and contract
    checks therefore dominate consensus, consistency, and Versa probabilities.
    """

    SUPPORT_BUCKETS = {
        "tool_final_supported": "trusted_tool_final",
        "derived_evidence_supported": "verified_derived",
        "search_evidence_supported": "direct_evidence",
        "attachment_evidence_supported": "direct_evidence",
        "tool_intermediate_supported": "bridge_evidence",
        "tool_failed_model_only": "unsupported",
        "no_support": "unsupported",
        "invalid": "unsupported",
        "contradicted": "contradicted",
    }
    SUPPORT_BUCKET_ORDER = (
        "contradicted",
        "unsupported",
        "bridge_evidence",
        "direct_evidence",
        "verified_derived",
        "trusted_tool_final",
    )
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
        answer_requirement_gate: AnswerRequirementGate | None = None,
        question: str = "",
    ) -> None:
        self.question = str(question or "").strip()
        self.answer_validator = answer_validator or AnswerValidator()
        self.clusterer = clusterer or AnswerCandidateClusterer(self.answer_validator)
        self.evidence_support_checker = (
            evidence_support_checker or EvidenceSupportChecker(self.answer_validator)
        )
        self.answer_requirement_gate = (
            answer_requirement_gate or AnswerRequirementGate()
        )

    def select(
        self,
        *,
        stage1_results: list[AgentReasoningSummary],
        candidates: list[AnswerCandidate],
        verifier_results: list[VerifierScoreByReasoning],
        evidence: dict[str, Any],
    ) -> FinalWinnerSelection:
        """Run every candidate through the fixed ordered-gate pipeline."""
        evaluations = [
            self._evaluate_candidate(
                candidate=candidate,
                stage1_results=stage1_results,
                verifier_results=verifier_results,
                evidence=evidence,
            )
            for candidate in candidates
        ]
        survivors = list(evaluations)
        gate_trace: list[GateResult] = []
        gates: tuple[Callable[..., GateResult], ...] = (
            self._apply_validity_gate,
            self._apply_requirement_gate,
            self._apply_contradiction_gate,
            self._apply_evidence_gate,
            self._apply_cross_agent_gate,
            self._apply_self_consistency_gate,
            self._apply_versa_gate,
        )

        for gate in gates:
            result = gate(survivors, evidence=evidence)
            gate_trace.append(result)
            survivors = result.survivors
            if result.terminal_status:
                return FinalWinnerSelection(
                    winner=None,
                    evaluation=None,
                    evaluations=evaluations,
                    status=result.terminal_status,
                    reason=result.terminal_reason,
                    gate_trace=gate_trace,
                )

        if not survivors:
            return FinalWinnerSelection(
                winner=None,
                evaluation=None,
                evaluations=evaluations,
                status="no_eligible_candidate",
                reason="all_candidates_eliminated",
                gate_trace=gate_trace,
            )
        if len(survivors) > 1:
            return FinalWinnerSelection(
                winner=None,
                evaluation=None,
                evaluations=evaluations,
                status="unresolved_exact_tie",
                reason="ordered_gates_could_not_separate_candidates",
                gate_trace=gate_trace,
            )

        selected = survivors[0]
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
                gate_trace=gate_trace,
            )

        winner = self.clusterer.summary_for_member(stage1_results, selected_member)
        return FinalWinnerSelection(
            winner=winner,
            evaluation=selected,
            evaluations=evaluations,
            status="answerable",
            reason=self._selection_reason(selected),
            gate_trace=gate_trace,
        )

    def _apply_validity_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        survivors: list[CandidateEvaluation] = []
        eliminated: list[CandidateGateDecision] = []
        decisions: list[CandidateGateDecision] = []
        for candidate in candidates:
            if candidate.eligible:
                decision = self._decision(candidate, "pass", "candidate_has_valid_run")
                survivors.append(candidate)
            else:
                reason = candidate.rejection_reason or "candidate_has_no_valid_run"
                decision = self._decision(candidate, "reject", reason)
                eliminated.append(decision)
            decisions.append(decision)
        return GateResult(
            gate_name="validity",
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
            terminal_status="no_eligible_candidate" if not survivors else "",
            terminal_reason=(
                "all_candidates_failed_validity_gate" if not survivors else ""
            ),
        )

    def _apply_requirement_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        requirement, role = self._answer_contract(evidence)
        survivors: list[CandidateEvaluation] = []
        eliminated: list[CandidateGateDecision] = []
        decisions: list[CandidateGateDecision] = []
        for candidate in candidates:
            member_results = []
            for member in self._member_evaluations(candidate):
                if not member.get("valid"):
                    continue
                member_results.append(
                    self.answer_requirement_gate.evaluate(
                        answer=candidate.answer,
                        answer_type=str(member.get("answer_type") or ""),
                        answer_requirement=requirement,
                        answer_role=role,
                    )
                )
            outcomes = {item.outcome for item in member_results}
            if "compatible" in outcomes:
                outcome = "pass"
                reason = "answer_requirement_compatible"
                candidate.requirement_status = "compatible"
            elif outcomes and outcomes == {"incompatible"}:
                outcome = "reject"
                reason = "answer_requirement_incompatible"
                candidate.requirement_status = "incompatible"
            else:
                outcome = "unknown"
                reason = "answer_requirement_not_decisive"
                candidate.requirement_status = "unknown"
            details = {
                "answer_requirement": requirement,
                "answer_role": role,
                "member_results": [item.to_dict() for item in member_results],
            }
            candidate.metadata["answer_requirement_gate"] = details
            decision = self._decision(candidate, outcome, reason, details)
            decisions.append(decision)
            if outcome == "reject":
                candidate.eligible = False
                candidate.rejection_reason = reason
                eliminated.append(decision)
            else:
                survivors.append(candidate)
        return GateResult(
            gate_name="answer_requirement",
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
            terminal_status=(
                "unresolved_requirement_conflict" if candidates and not survivors else ""
            ),
            terminal_reason=(
                "all_candidates_conflict_with_answer_requirement"
                if candidates and not survivors
                else ""
            ),
        )

    def _apply_contradiction_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        survivors: list[CandidateEvaluation] = []
        eliminated: list[CandidateGateDecision] = []
        decisions: list[CandidateGateDecision] = []
        for candidate in candidates:
            valid_paths = [
                item
                for item in self._member_evaluations(candidate)
                if item.get("valid") and not item.get("contradicted")
            ]
            if not valid_paths:
                candidate.eligible = False
                candidate.contradicted = True
                candidate.support_status = "contradicted"
                candidate.rejection_reason = "all_candidate_paths_contradicted"
                decision = self._decision(
                    candidate,
                    "reject",
                    "all_candidate_paths_contradicted",
                )
                eliminated.append(decision)
            else:
                candidate.contradicted = False
                self._apply_selected_member(candidate, self._select_member_path(valid_paths))
                decision = self._decision(
                    candidate,
                    "pass",
                    "candidate_has_non_contradicted_path",
                    {"surviving_path_count": len(valid_paths)},
                )
                survivors.append(candidate)
            decisions.append(decision)
        return GateResult(
            gate_name="contradiction",
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
            terminal_status=(
                "no_eligible_candidate" if candidates and not survivors else ""
            ),
            terminal_reason=(
                "all_candidates_contradicted_by_evidence"
                if candidates and not survivors
                else ""
            ),
        )

    def _apply_evidence_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        if not candidates:
            return GateResult(gate_name="evidence_support")
        buckets = {
            item.candidate_key: self._support_bucket(item.support_status)
            for item in candidates
        }
        best_bucket = max(
            buckets.values(),
            key=self.SUPPORT_BUCKET_ORDER.index,
        )
        if best_bucket == "unsupported":
            survivors = list(candidates)
            decisions = [
                self._decision(
                    item,
                    "unknown",
                    "candidate_has_no_verified_support",
                    {"support_bucket": buckets[item.candidate_key]},
                )
                for item in candidates
            ]
            factual = self._is_factual_search(evidence)
            return GateResult(
                gate_name="evidence_support",
                survivors=survivors,
                decisions=decisions,
                terminal_status=(
                    "unresolved_unsupported_factual_conflict"
                    if factual and len(candidates) > 1
                    else "unresolved_factual_without_support"
                    if factual
                    else ""
                ),
                terminal_reason=(
                    "factual_candidates_have_no_verified_evidence_support"
                    if factual
                    else ""
                ),
            )

        survivors = [
            item
            for item in candidates
            if buckets[item.candidate_key] == best_bucket
        ]
        eliminated = [
            self._decision(
                item,
                "reject",
                "lower_evidence_support_bucket",
                {
                    "support_bucket": buckets[item.candidate_key],
                    "selected_bucket": best_bucket,
                },
            )
            for item in candidates
            if item not in survivors
        ]
        decisions = [
            self._decision(
                item,
                "pass" if item in survivors else "reject",
                (
                    "highest_evidence_support_bucket"
                    if item in survivors
                    else "lower_evidence_support_bucket"
                ),
                {
                    "support_bucket": buckets[item.candidate_key],
                    "selected_bucket": best_bucket,
                },
            )
            for item in candidates
        ]
        for item in candidates:
            if item not in survivors:
                item.eligible = False
                item.rejection_reason = "lower_evidence_support_bucket"
        return GateResult(
            gate_name="evidence_support",
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
        )

    def _apply_cross_agent_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        return self._retain_maximum(
            gate_name="cross_agent_consensus",
            candidates=candidates,
            value=lambda item: len(set(item.supporting_agent_ids)),
            pass_reason="maximum_distinct_agent_support",
            reject_reason="fewer_distinct_supporting_agents",
            detail_name="distinct_agent_count",
        )

    def _apply_self_consistency_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        if len(candidates) <= 1:
            return self._pass_through(
                "self_consistency",
                candidates,
                "single_candidate_no_consistency_comparison",
            )
        metrics = {
            item.candidate_key: self._consistency_metrics(item)
            for item in candidates
        }
        survivors = self._filter_max(
            candidates,
            lambda item: metrics[item.candidate_key]["best_class"],
        )
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: metrics[item.candidate_key]["agents_at_best"],
            )
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: item.supporting_run_count,
            )
        return self._gate_from_survivors(
            gate_name="self_consistency",
            candidates=candidates,
            survivors=survivors,
            pass_reason="maximum_agent_internal_consistency",
            reject_reason="lower_agent_internal_consistency",
            details=metrics,
        )

    def _apply_versa_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        if len(candidates) <= 1:
            return self._pass_through(
                "versa_verification",
                candidates,
                "single_candidate_no_versa_comparison",
            )
        availability = {
            item.candidate_key: bool(item.metadata.get("versa_available"))
            for item in candidates
        }
        if not all(availability.values()):
            decisions = [
                self._decision(
                    item,
                    "unknown",
                    "versa_result_missing_for_one_or_more_candidates",
                    {"versa_available": availability[item.candidate_key]},
                )
                for item in candidates
            ]
            return GateResult(
                gate_name="versa_verification",
                survivors=list(candidates),
                decisions=decisions,
                terminal_status="unresolved_missing_verification",
                terminal_reason="not_all_surviving_candidates_have_versa_results",
            )

        survivors = self._filter_max(
            candidates,
            lambda item: item.critical_step_floor,
        )
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: item.critical_step_geometric_mean,
            )
        result = self._gate_from_survivors(
            gate_name="versa_verification",
            candidates=candidates,
            survivors=survivors,
            pass_reason="best_critical_step_verification",
            reject_reason="lower_critical_step_verification",
            details={
                item.candidate_key: {
                    "critical_step_floor": item.critical_step_floor,
                    "critical_step_geometric_mean": item.critical_step_geometric_mean,
                }
                for item in candidates
            },
        )
        if len(survivors) > 1:
            result.terminal_status = "unresolved_exact_tie"
            result.terminal_reason = "versa_could_not_separate_surviving_candidates"
        return result

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
        valid_members = [item for item in member_evaluations if item["valid"]]
        if not valid_members:
            return CandidateEvaluation(
                candidate_key=candidate.candidate_key,
                answer=candidate.representative_answer,
                eligible=False,
                rejection_reason="no_valid_candidate_path",
                support_status="invalid",
                supporting_agent_ids=candidate.supporting_agent_ids,
                supporting_run_count=candidate.supporting_run_count,
                metadata={"member_evaluations": member_evaluations},
            )

        selected_member = self._select_member_path(valid_members)
        evaluation = CandidateEvaluation(
            candidate_key=candidate.candidate_key,
            answer=candidate.representative_answer,
            eligible=True,
            supporting_agent_ids=candidate.supporting_agent_ids,
            supporting_run_count=candidate.supporting_run_count,
            metadata={"member_evaluations": member_evaluations},
        )
        self._apply_selected_member(evaluation, selected_member)
        return evaluation

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
            question=self.question,
        )
        verifier = self._find_verifier_result(member, verifier_results)
        process = self._process_metadata(verifier)
        support_status = support.status
        verifier_support = self._verifier_support_metadata(verifier)
        if verifier_support:
            support_status = str(verifier_support.get("status") or support_status)
        contradicted = support_status == "contradicted"
        valid = bool(
            member.parse_completed
            and member.schema_valid
            and self.answer_validator.is_valid(
                member.answer,
                answer_type=member.answer_type,
            )
        )
        return {
            "agent_id": member.agent_id,
            "run_index": member.run_index,
            "answer": member.answer,
            "answer_type": member.answer_type,
            "reasoning": member.reasoning,
            "valid": valid,
            "contradicted": contradicted,
            "support_status": support_status,
            "support_bucket": self._support_bucket(support_status),
            "direct_support": support_status in self.DIRECT_SUPPORT_STATUSES,
            "agent_confidence": member.agent_confidence,
            "agent_answer_frequency": member.agent_answer_frequency,
            "eligible_run_count": member.eligible_run_count,
            "versa_available": verifier is not None and bool(process),
            "critical_step_floor": float(process.get("critical_step_floor") or 0.0),
            "critical_step_geometric_mean": float(
                process.get("critical_step_geometric_mean") or 0.0
            ),
            "average_verifier_probability": float(
                process.get("average_probability")
                or (verifier.verifier_score if verifier is not None else 0.0)
            ),
            "critical_step_indices": list(process.get("critical_step_indices") or []),
            "evidence_support": (
                dict(verifier_support)
                if verifier_support
                else self.evidence_support_checker.summary_to_dict(support)
            ),
        }

    def _select_member_path(self, members: list[dict[str, Any]]) -> dict[str, Any]:
        survivors = [item for item in members if not item.get("contradicted")]
        if not survivors:
            survivors = list(members)
        best_bucket = max(
            (str(item.get("support_bucket") or "unsupported") for item in survivors),
            key=self.SUPPORT_BUCKET_ORDER.index,
        )
        survivors = [
            item for item in survivors if item.get("support_bucket") == best_bucket
        ]
        available = [item for item in survivors if item.get("versa_available")]
        if available:
            survivors = self._filter_max_dict(
                available,
                lambda item: float(item.get("critical_step_floor") or 0.0),
            )
            survivors = self._filter_max_dict(
                survivors,
                lambda item: float(item.get("critical_step_geometric_mean") or 0.0),
            )
        survivors = self._filter_max_dict(
            survivors,
            lambda item: int(item.get("agent_answer_frequency") or 0),
        )
        survivors = self._filter_max_dict(
            survivors,
            lambda item: float(item.get("agent_confidence") or 0.0),
        )
        return min(
            survivors,
            key=lambda item: (str(item.get("agent_id") or ""), int(item.get("run_index") or 0)),
        )

    def _apply_selected_member(
        self,
        evaluation: CandidateEvaluation,
        member: dict[str, Any],
    ) -> None:
        evaluation.support_status = str(member.get("support_status") or "no_support")
        evaluation.direct_support = bool(member.get("direct_support"))
        evaluation.selected_agent_id = str(member.get("agent_id") or "")
        evaluation.selected_run_index = int(member.get("run_index") or 0)
        evaluation.selected_reasoning = str(member.get("reasoning") or "")
        evaluation.selected_agent_confidence = float(
            member.get("agent_confidence") or 0.0
        )
        evaluation.selected_agent_answer_frequency = int(
            member.get("agent_answer_frequency") or 0
        )
        evaluation.critical_step_floor = float(
            member.get("critical_step_floor") or 0.0
        )
        evaluation.critical_step_geometric_mean = float(
            member.get("critical_step_geometric_mean") or 0.0
        )
        evaluation.average_verifier_probability = float(
            member.get("average_verifier_probability") or 0.0
        )
        evaluation.metadata["versa_available"] = bool(member.get("versa_available"))
        evaluation.metadata["selected_support_bucket"] = str(
            member.get("support_bucket") or "unsupported"
        )

    def _retain_maximum(
        self,
        *,
        gate_name: str,
        candidates: list[CandidateEvaluation],
        value: Callable[[CandidateEvaluation], int | float],
        pass_reason: str,
        reject_reason: str,
        detail_name: str,
    ) -> GateResult:
        if len(candidates) <= 1:
            return self._pass_through(
                gate_name,
                candidates,
                f"single_candidate_no_{gate_name}_comparison",
            )
        survivors = self._filter_max(candidates, value)
        details = {
            item.candidate_key: {detail_name: value(item)} for item in candidates
        }
        return self._gate_from_survivors(
            gate_name=gate_name,
            candidates=candidates,
            survivors=survivors,
            pass_reason=pass_reason,
            reject_reason=reject_reason,
            details=details,
        )

    def _gate_from_survivors(
        self,
        *,
        gate_name: str,
        candidates: list[CandidateEvaluation],
        survivors: list[CandidateEvaluation],
        pass_reason: str,
        reject_reason: str,
        details: dict[str, dict[str, Any]],
    ) -> GateResult:
        eliminated: list[CandidateGateDecision] = []
        decisions: list[CandidateGateDecision] = []
        for item in candidates:
            passed = item in survivors
            decision = self._decision(
                item,
                "pass" if passed else "reject",
                pass_reason if passed else reject_reason,
                details.get(item.candidate_key, {}),
            )
            decisions.append(decision)
            if not passed:
                item.eligible = False
                item.rejection_reason = reject_reason
                eliminated.append(decision)
        return GateResult(
            gate_name=gate_name,
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
        )

    def _pass_through(
        self,
        gate_name: str,
        candidates: list[CandidateEvaluation],
        reason: str,
    ) -> GateResult:
        return GateResult(
            gate_name=gate_name,
            survivors=list(candidates),
            decisions=[self._decision(item, "pass", reason) for item in candidates],
        )

    def _decision(
        self,
        candidate: CandidateEvaluation,
        outcome: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> CandidateGateDecision:
        return CandidateGateDecision(
            candidate_key=candidate.candidate_key,
            outcome=outcome,
            reason=reason,
            details=dict(details or {}),
        )

    def _filter_max(
        self,
        candidates: list[CandidateEvaluation],
        value: Callable[[CandidateEvaluation], int | float],
    ) -> list[CandidateEvaluation]:
        if not candidates:
            return []
        maximum = max(value(item) for item in candidates)
        return [item for item in candidates if value(item) == maximum]

    def _filter_max_dict(
        self,
        candidates: list[dict[str, Any]],
        value: Callable[[dict[str, Any]], int | float],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        maximum = max(value(item) for item in candidates)
        return [item for item in candidates if value(item) == maximum]

    def _consistency_metrics(self, candidate: CandidateEvaluation) -> dict[str, Any]:
        ratios_by_agent: dict[str, float] = {}
        for item in self._member_evaluations(candidate):
            if not item.get("valid") or item.get("contradicted"):
                continue
            denominator = max(1, int(item.get("eligible_run_count") or 1))
            ratio = int(item.get("agent_answer_frequency") or 0) / denominator
            agent_id = str(item.get("agent_id") or "")
            ratios_by_agent[agent_id] = max(ratios_by_agent.get(agent_id, 0.0), ratio)
        best_ratio = max(ratios_by_agent.values(), default=0.0)
        best_class = 3 if best_ratio >= 1.0 else 2 if best_ratio >= (2 / 3) else 1
        return {
            "best_class": best_class,
            "best_ratio": best_ratio,
            "agents_at_best": sum(
                1 for ratio in ratios_by_agent.values() if ratio == best_ratio
            ),
            "ratios_by_agent": ratios_by_agent,
            "supporting_run_count": candidate.supporting_run_count,
        }

    def _member_evaluations(
        self,
        candidate: CandidateEvaluation,
    ) -> list[dict[str, Any]]:
        values = candidate.metadata.get("member_evaluations", [])
        return [item for item in values if isinstance(item, dict)]

    def _support_bucket(self, support_status: str) -> str:
        return self.SUPPORT_BUCKETS.get(str(support_status or ""), "unsupported")

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

    def _is_factual_search(self, evidence: dict[str, Any]) -> bool:
        routing = evidence.get("routing")
        if not isinstance(routing, dict):
            return False
        return str(routing.get("primary_route") or "").strip().lower() == "factual_search"

    def _answer_contract(self, evidence: dict[str, Any]) -> tuple[str, str]:
        requirement = str(evidence.get("answer_requirement") or "").strip()
        role = str(evidence.get("answer_role") or "").strip()
        for item in evidence.get("tool_usage", []) or []:
            if not isinstance(item, dict) or item.get("tool_name") != "search":
                continue
            raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
            diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
            contract = diagnostics.get("evidence_selection_contract")
            if isinstance(contract, dict):
                requirement = requirement or str(contract.get("answer_requirement") or "").strip()
                role = role or str(contract.get("answer_role") or "").strip()
            for evidence_item in raw.get("evidence_items", []) or []:
                if not isinstance(evidence_item, dict):
                    continue
                requirement = requirement or str(evidence_item.get("answer_requirement") or "").strip()
                role = role or str(evidence_item.get("answer_role") or "").strip()
        return requirement, role

    def _selection_reason(self, evaluation: CandidateEvaluation) -> str:
        bucket = self._support_bucket(evaluation.support_status)
        if bucket == "trusted_tool_final":
            return "trusted_tool_final_answer"
        if bucket == "verified_derived":
            return "verified_derived_answer"
        if bucket == "direct_evidence" and len(evaluation.supporting_agent_ids) >= 2:
            return "direct_evidence_cross_agent_consensus"
        if bucket == "direct_evidence":
            return "direct_evidence_supported_candidate"
        if bucket == "bridge_evidence":
            return "bridge_evidence_supported_reasoning"
        if len(evaluation.supporting_agent_ids) >= 2:
            return "cross_agent_consensus"
        return "single_valid_candidate"


__all__ = ["FinalWinnerSelection", "FinalWinnerSelector"]
