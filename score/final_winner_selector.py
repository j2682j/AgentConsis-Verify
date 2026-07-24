from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidateEvaluation,
    CandidatePathEvaluation,
    CandidateRun,
    VerifierScoreByReasoning,
)
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.answer_requirement_gate import AnswerRequirementGate
from score.answer_requirement_contract import TaskAnswerRequirementContract
from score.answer_validator import AnswerValidator
from score.evidence_support_level import (
    EvidenceSupportLevel,
    is_strong_support_level,
    support_level_for_status,
)
from score.evidence_answer_resolver import EvidenceAnswerResolver
from score.gate_result import CandidateGateDecision, GateResult
from score.question_echo import is_question_echo
from utils.network_utils import normalize_text


@dataclass
class FinalWinnerSelection:
    """Save the final winner and the complete ordered-gate trace."""

    winner: AgentReasoningSummary | None
    evaluation: CandidateEvaluation | None
    evaluations: list[CandidateEvaluation]
    status: str
    reason: str
    gate_trace: list[GateResult] = field(default_factory=list)
    resolved_answer: str = ""
    selection_origin: str = "agent_candidate"
    resolution_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        active = [item.candidate_key for item in self.evaluations if item.selection_state == "active"]
        reserve = [item.candidate_key for item in self.evaluations if item.selection_state == "reserve"]
        rejected = [item.candidate_key for item in self.evaluations if item.selection_state == "rejected"]
        return {
            "strategy": "candidate_ordered_gates",
            "status": self.status,
            "selection_reason": self.reason,
            "selected_answer": (
                self.evaluation.answer if self.evaluation else self.resolved_answer
            ),
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
            "selection_origin": self.selection_origin,
            "evidence_only_resolution": dict(self.resolution_metadata),
            "candidates": [asdict(item) for item in self.evaluations],
            "active_candidate_keys": active,
            "reserve_candidate_keys": reserve,
            "rejected_candidate_keys": rejected,
            "evidence_resolution_status": (
                "resolved"
                if self.resolved_answer or any(
                    item.selection_state != "rejected"
                    and support_level_for_status(item.support_status).value
                    != EvidenceSupportLevel.UNSUPPORTED.value
                    for item in self.evaluations
                )
                else "unresolved"
            ),
        }


class FinalWinnerSelector:
    """
    Select a candidate through ordered gates without a weighted aggregate score.

    A rejected candidate never re-enters the pipeline. Evidence and contract
    checks therefore dominate consensus, consistency, and Versa probabilities.
    """

    SUPPORT_BUCKET_ORDER = (
        EvidenceSupportLevel.CONTRADICTED.value,
        EvidenceSupportLevel.UNSUPPORTED.value,
        EvidenceSupportLevel.BRIDGE_EVIDENCE.value,
        EvidenceSupportLevel.DIRECT_EVIDENCE.value,
        EvidenceSupportLevel.VERIFIED_DERIVED.value,
        EvidenceSupportLevel.TRUSTED_TOOL_FINAL.value,
    )

    def __init__(
        self,
        *,
        clusterer: AnswerCandidateClusterer | None = None,
        answer_validator: AnswerValidator | None = None,
        evidence_support_checker: Any | None = None,
        answer_requirement_gate: AnswerRequirementGate | None = None,
        evidence_answer_resolver: EvidenceAnswerResolver | None = None,
        question: str = "",
        enable_fallback_selection: bool = True,
    ) -> None:
        self.question = str(question or "").strip()
        # An empty final answer can never be correct, so an unresolved gate
        # outcome falls back to the best remaining candidate instead of
        # abstaining. Disable only to observe the raw gate verdict.
        self.enable_fallback_selection = bool(enable_fallback_selection)
        self.answer_validator = answer_validator or AnswerValidator()
        self.clusterer = clusterer or AnswerCandidateClusterer(self.answer_validator)
        self.answer_requirement_gate = (
            answer_requirement_gate or AnswerRequirementGate()
        )
        self.evidence_answer_resolver = (
            evidence_answer_resolver or EvidenceAnswerResolver()
        )

    def select(
        self,
        *,
        stage1_results: list[AgentReasoningSummary],
        candidates: list[AnswerCandidate],
        path_evaluations: list[CandidatePathEvaluation] | None = None,
        verifier_results: list[VerifierScoreByReasoning] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> FinalWinnerSelection:
        """Run every candidate through the fixed ordered-gate pipeline."""
        evidence = evidence or {}
        path_index = {
            (
                item.identity.candidate_key,
                item.identity.agent_id,
                item.identity.run_index,
            ): item
            for item in list(path_evaluations or [])
        }
        evaluations = [
            self._evaluate_candidate(
                candidate=candidate,
                path_index=path_index,
                verifier_results=list(verifier_results or []),
            )
            for candidate in candidates
        ]
        resolution = self.resolve_evaluations(evaluations, evidence=evidence)
        if resolution.evaluation is None:
            return resolution
        selected = resolution.evaluation
        selected_member = self._member_by_identity(
            candidates,
            candidate_key=selected.candidate_key,
            agent_id=selected.selected_agent_id,
            run_index=selected.selected_run_index,
        )
        if selected_member is None:
            return replace(
                resolution,
                winner=None,
                status="selected_member_missing",
                reason="candidate_member_could_not_be_restored",
            )
        winner = self.clusterer.summary_for_member(stage1_results, selected_member)
        # Candidate identity is canonical; the selected run only contributes
        # reasoning and provenance. Restoring its raw answer can reintroduce an
        # explicitly repaired ordering or formatting error.
        winner = replace(winner, compressed_answer=selected.answer)
        return replace(resolution, winner=winner)

    def resolve_evaluations(
        self,
        evaluations: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any] | None = None,
    ) -> FinalWinnerSelection:
        """Run pre-built candidate evaluations through the ordered gates.

        Split from select() so saved evaluations (for example from a recorded
        selection trace) can replay the exact decision logic without the
        member-restore step; the returned selection carries no winner summary.
        """
        evidence = evidence or {}
        for evaluation in evaluations:
            evaluation.selection_state = "active"
            evaluation.hard_rejection_reason = ""
            evaluation.soft_deferred_by = []
        survivors = list(evaluations)
        gate_trace: list[GateResult] = []
        gates: tuple[Callable[..., GateResult], ...] = (
            self._apply_validity_gate,
            self._apply_requirement_gate,
            self._apply_contradiction_gate,
            self._apply_evidence_gate,
            # Attestation runs before the vote-based gates on purpose. Agents
            # sharing a context agree on the same invented answer often enough
            # that consensus alone would confirm it; asking whether the fetched
            # pages ever state it is independent of how many runs proposed it.
            self._apply_corpus_attestation_gate,
            self._apply_cross_agent_gate,
            self._apply_self_consistency_gate,
            self._apply_versa_gate,
        )

        selected: CandidateEvaluation | None = None
        fallback_from: tuple[str, str] | None = None
        for gate in gates:
            result = gate(survivors, evidence=evidence)
            gate_trace.append(result)
            survivors = result.survivors
            if result.terminal_status:
                fallback = self._fallback_selection(evaluations, survivors)
                if fallback is None:
                    return FinalWinnerSelection(
                        winner=None,
                        evaluation=None,
                        evaluations=evaluations,
                        status=result.terminal_status,
                        reason=result.terminal_reason,
                        gate_trace=gate_trace,
                    )
                selected = fallback
                fallback_from = (result.terminal_status, result.terminal_reason)
                break
            if (
                result.gate_name == "evidence_support"
                and bool(result.metadata.get("all_candidates_unsupported"))
            ):
                resolution = self.evidence_answer_resolver.resolve(
                    evidence,
                    allowed_candidate_keys={item.candidate_key for item in survivors},
                    # A completed relation plan may legitimately derive an
                    # answer no Agent nominated (for example a multi-hop
                    # relation target). Generic promoted facts may not replace
                    # an existing candidate pool.
                    allow_new_nomination=bool(
                        str(evidence.get("required_relation") or "").strip()
                    ) or not bool(survivors),
                )
                if resolution.resolved:
                    gate_trace.append(
                        GateResult(
                            gate_name="evidence_only_resolution",
                            survivors=[],
                            metadata=resolution.to_dict(),
                        )
                    )
                    return FinalWinnerSelection(
                        winner=None,
                        evaluation=None,
                        evaluations=evaluations,
                        status="answerable",
                        reason="unique_relation_bound_evidence_answer",
                        gate_trace=gate_trace,
                        resolved_answer=resolution.answer,
                        selection_origin="evidence_only_resolution",
                        resolution_metadata=resolution.to_dict(),
                    )
                if resolution.status == "conflict":
                    gate_trace.append(
                        GateResult(
                            gate_name="evidence_only_resolution",
                            survivors=[],
                            terminal_status="unresolved_evidence_conflict",
                            terminal_reason=resolution.reason,
                            metadata=resolution.to_dict(),
                        )
                    )
                    fallback = self._fallback_selection(evaluations, survivors)
                    if fallback is None:
                        return FinalWinnerSelection(
                            winner=None,
                            evaluation=None,
                            evaluations=evaluations,
                            status="unresolved_evidence_conflict",
                            reason=resolution.reason,
                            gate_trace=gate_trace,
                            selection_origin="evidence_only_resolution",
                            resolution_metadata=resolution.to_dict(),
                        )
                    selected = fallback
                    fallback_from = ("unresolved_evidence_conflict", resolution.reason)
                    break

        if selected is None:
            if not survivors:
                selected = self._fallback_selection(evaluations, [])
                if selected is None:
                    return FinalWinnerSelection(
                        winner=None,
                        evaluation=None,
                        evaluations=evaluations,
                        status="no_eligible_candidate",
                        reason="all_candidates_eliminated",
                        gate_trace=gate_trace,
                    )
                fallback_from = ("no_eligible_candidate", "all_candidates_eliminated")
            elif len(survivors) > 1:
                selected = self._fallback_selection(evaluations, survivors)
                if selected is None:
                    return FinalWinnerSelection(
                        winner=None,
                        evaluation=None,
                        evaluations=evaluations,
                        status="unresolved_exact_tie",
                        reason="ordered_gates_could_not_separate_candidates",
                        gate_trace=gate_trace,
                    )
                fallback_from = (
                    "unresolved_exact_tie",
                    "ordered_gates_could_not_separate_candidates",
                )
            else:
                selected = survivors[0]
        if (
            fallback_from is None
            and self._is_factual_search(evidence)
            and self._support_bucket(selected.support_status)
            == EvidenceSupportLevel.UNSUPPORTED.value
            and not bool(selected.metadata.get("versa_available"))
        ):
            if not self.enable_fallback_selection:
                return FinalWinnerSelection(
                    winner=None,
                    evaluation=selected,
                    evaluations=evaluations,
                    status="unresolved_factual_without_support",
                    reason="factual_candidate_lacks_evidence_and_verification",
                    gate_trace=gate_trace,
                )
            fallback_from = (
                "unresolved_factual_without_support",
                "factual_candidate_lacks_evidence_and_verification",
            )
        if fallback_from is not None:
            return FinalWinnerSelection(
                winner=None,
                evaluation=selected,
                evaluations=evaluations,
                status="answerable",
                reason=f"fallback_best_candidate_after_{fallback_from[0]}",
                gate_trace=gate_trace,
                selection_origin="fallback_best_candidate",
                resolution_metadata={
                    "fallback_from_status": fallback_from[0],
                    "fallback_from_reason": fallback_from[1],
                },
            )
        return FinalWinnerSelection(
            winner=None,
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
                candidate.selection_state = "rejected"
                candidate.hard_rejection_reason = reason
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
                candidate.selection_state = "rejected"
                candidate.hard_rejection_reason = reason
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
                candidate.selection_state = "rejected"
                candidate.hard_rejection_reason = "all_candidate_paths_contradicted"
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
                metadata={
                    "factual_search": factual,
                    "all_candidates_unsupported": True,
                },
            )

        top_candidates = [
            item
            for item in candidates
            if buckets[item.candidate_key] == best_bucket
        ]
        echo_only_support = (
            best_bucket
            in {
                EvidenceSupportLevel.DIRECT_EVIDENCE.value,
                EvidenceSupportLevel.VERIFIED_DERIVED.value,
            }
            and all(
                is_question_echo(item.answer, self.question)
                for item in top_candidates
            )
        )
        if not is_strong_support_level(best_bucket) or echo_only_support:
            # A weak/indirect signal (bridge_evidence) is often a single matched
            # tool value from a low-quality source, and support held only by
            # question-echo answers is tautological (pages echo the query).
            # Neither may eliminate rivals; carry the signal forward as a soft
            # hint for the consensus/consistency gates instead. Trusted tool
            # finals stay exempt from the echo demotion: a deterministic
            # handler computing an echoed value is real verification.
            for item in candidates:
                item.metadata["weak_support_bucket"] = buckets[item.candidate_key]
            decisions = [
                self._decision(
                    item,
                    "unknown",
                    (
                        "echo_support_not_eliminating"
                        if echo_only_support
                        else "weak_support_bucket_not_eliminating"
                    ),
                    {"support_bucket": buckets[item.candidate_key]},
                )
                for item in candidates
            ]
            return GateResult(
                gate_name="evidence_support",
                survivors=list(candidates),
                decisions=decisions,
                metadata={
                    "gate_strength": "soft",
                    "best_bucket": best_bucket,
                    "weak_signal_not_eliminating": not echo_only_support,
                    "echo_support_not_eliminating": echo_only_support,
                },
            )

        survivors = [
            item
            for item in candidates
            if buckets[item.candidate_key] == best_bucket
        ]
        deferred = [
            self._decision(
                item,
                "reserve",
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
                "pass" if item in survivors else "reserve",
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
                item.selection_state = "reserve"
                if "evidence_support" not in item.soft_deferred_by:
                    item.soft_deferred_by.append("evidence_support")
        return GateResult(
            gate_name="evidence_support",
            survivors=survivors,
            decisions=decisions,
            metadata={
                "gate_strength": "soft",
                "reserve_candidate_keys": [item.candidate_key for item in candidates if item not in survivors],
                "deferred": [item.to_dict() for item in deferred],
            },
        )

    def _fallback_selection(
        self,
        evaluations: list[CandidateEvaluation],
        survivors: list[CandidateEvaluation],
    ) -> CandidateEvaluation | None:
        """Pick the best remaining candidate when the gates cannot decide.

        An abstention is guaranteed wrong on exact-answer tasks, so any
        deterministic best-effort pick strictly dominates an empty answer.
        Prefers gate survivors, then non-rejected candidates, then anything
        with a non-empty answer; ranks by evidence-support bucket, best
        supporter self-consistency, run count, and distinct agent count.
        Ties keep the first-seen candidate for reproducibility.
        """
        if not self.enable_fallback_selection:
            return None
        pools = (
            list(survivors or []),
            [item for item in evaluations if item.selection_state != "rejected"],
            list(evaluations),
        )
        pool: list[CandidateEvaluation] = []
        for group in pools:
            pool = [item for item in group if str(item.answer or "").strip()]
            if pool:
                break
        if not pool:
            return None
        best: CandidateEvaluation | None = None
        best_rank: tuple[int, float, int, int] | None = None
        for item in pool:
            rank = (
                self.SUPPORT_BUCKET_ORDER.index(
                    self._support_bucket(item.support_status)
                ),
                round(self._max_supporter_confidence(item), 4),
                int(item.supporting_run_count or 0),
                len(set(item.supporting_agent_ids or [])),
            )
            if best_rank is None or rank > best_rank:
                best = item
                best_rank = rank
        return best

    def _max_supporter_confidence(self, candidate: CandidateEvaluation) -> float:
        """Best self-consistency among the candidate's valid supporters.

        Agent confidence here is the fraction of that agent's runs agreeing on
        this answer, so it measures self-consistency rather than calibration.
        """
        best = float(candidate.selected_agent_confidence or 0.0)
        for member in self._member_evaluations(candidate):
            if not member.get("valid") or member.get("contradicted"):
                continue
            best = max(best, float(member.get("agent_confidence") or 0.0))
        return best

    def _consensus_rank(
        self,
        candidate: CandidateEvaluation,
    ) -> tuple[int, int, float]:
        """Order consensus candidates by total votes, then breadth, then conviction.

        Agent confidence is the share of an agent's own runs that repeated an
        answer, so it measures self-consistency, not calibration: an agent that
        is reliably wrong scores 1.0. Ranking by confidence first therefore
        lets one stubborn agent outvote a better-supported answer — measured on
        saved runs it lost tasks where the correct answer existed but came from
        fewer runs of a less repetitive agent.

        Total supporting runs is the more honest primary key: it counts every
        vote the answer actually received, across agents. Distinct agents then
        breaks ties in favour of independent agreement over one agent repeating
        itself, and confidence only separates candidates that are level on both.
        """
        return (
            int(candidate.supporting_run_count or 0),
            len(set(candidate.supporting_agent_ids)),
            round(self._max_supporter_confidence(candidate), 4),
        )

    def _apply_cross_agent_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        # Agents fed the same empty context often agree on a fragment of the
        # question itself; that agreement carries no information. When echo
        # and non-echo candidates coexist, only non-echo candidates compete
        # for consensus. Trusted tool finals stay exempt: a deterministic
        # handler computing an echoed value is real verification.
        echo_candidates = [
            item
            for item in candidates
            if (
                is_question_echo(item.answer, self.question)
                and self._support_bucket(item.support_status)
                != EvidenceSupportLevel.TRUSTED_TOOL_FINAL.value
            )
        ]
        non_echo_candidates = [
            item for item in candidates if item not in echo_candidates
        ]
        if echo_candidates and non_echo_candidates:
            for item in echo_candidates:
                item.selection_state = "reserve"
                if "cross_agent_consensus" not in item.soft_deferred_by:
                    item.soft_deferred_by.append("cross_agent_consensus")
            result = self._retain_maximum(
                gate_name="cross_agent_consensus",
                candidates=non_echo_candidates,
                value=self._consensus_rank,
                pass_reason="maximum_confidence_weighted_agent_support",
                reject_reason="lower_confidence_weighted_agent_support",
                detail_name="consensus_rank",
                hard=False,
            )
            result.metadata["echo_candidates_deferred"] = [
                item.candidate_key for item in echo_candidates
            ]
            result.decisions.extend(
                self._decision(
                    item,
                    "reserve",
                    "question_echo_answer_cannot_win_consensus",
                )
                for item in echo_candidates
            )
            return result
        return self._retain_maximum(
            gate_name="cross_agent_consensus",
            candidates=candidates,
            value=self._consensus_rank,
            pass_reason="maximum_confidence_weighted_agent_support",
            reject_reason="lower_confidence_weighted_agent_support",
            detail_name="consensus_rank",
            hard=False,
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
            hard=False,
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
        available_candidates = [
            item for item in candidates if availability[item.candidate_key]
        ]
        if not available_candidates:
            decisions = [
                self._decision(
                    item,
                    "unknown",
                    "versa_not_available_for_any_candidate",
                    {"versa_available": availability[item.candidate_key]},
                )
                for item in candidates
            ]
            return GateResult(
                gate_name="versa_verification",
                survivors=list(candidates),
                decisions=decisions,
                metadata={"coverage": "none", "available_count": 0},
            )

        if len(available_candidates) != len(candidates):
            best_available = self._filter_max(
                available_candidates,
                lambda item: item.critical_step_floor,
            )
            if len(best_available) > 1:
                best_available = self._filter_max(
                    best_available,
                    lambda item: item.critical_step_geometric_mean,
                )
            unavailable = [
                item for item in candidates if not availability[item.candidate_key]
            ]
            survivors = [*best_available, *unavailable]
            return self._gate_from_survivors(
                gate_name="versa_verification",
                candidates=candidates,
                survivors=survivors,
                pass_reason="best_available_or_unscored_candidate",
                reject_reason="lower_available_versa_verification",
                details={
                    item.candidate_key: {
                        "versa_available": availability[item.candidate_key],
                        "critical_step_floor": item.critical_step_floor,
                        "critical_step_geometric_mean": item.critical_step_geometric_mean,
                        "coverage": "partial",
                    }
                    for item in candidates
                },
                hard=False,
            )

        # Measured on saved GAIA runs, VersaPRM probabilities scored wrong
        # answers higher than correct ones on average, so verifier scores may
        # only break ties that self-consistency and the fetched corpus cannot.
        survivors = self._filter_max(
            candidates,
            lambda item: item.selected_agent_answer_frequency,
        )
        tie_break_depth = 0
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: round(self._max_supporter_confidence(item), 4),
            )
            tie_break_depth = 1
        corpus_mentions: dict[str, int] = {}
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: item.critical_step_floor,
            )
            tie_break_depth = 3
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: item.critical_step_geometric_mean,
            )
            tie_break_depth = 4
        if len(survivors) > 1:
            survivors = self._filter_max(
                survivors,
                lambda item: item.average_verifier_probability,
            )
            tie_break_depth = 5
        result = self._gate_from_survivors(
            gate_name="versa_verification",
            candidates=candidates,
            survivors=survivors,
            pass_reason="best_consistency_then_verification",
            reject_reason="lower_consistency_then_verification",
            details={
                item.candidate_key: {
                    "selected_agent_answer_frequency": item.selected_agent_answer_frequency,
                    "max_supporter_confidence": round(
                        self._max_supporter_confidence(item), 4
                    ),
                    "critical_step_floor": item.critical_step_floor,
                    "critical_step_geometric_mean": item.critical_step_geometric_mean,
                    "average_verifier_probability": item.average_verifier_probability,
                }
                for item in candidates
            },
            hard=False,
        )
        result.metadata["tie_break_depth"] = tie_break_depth
        if corpus_mentions:
            result.metadata["corpus_mention_counts"] = dict(corpus_mentions)
        if len(survivors) > 1:
            result.terminal_status = "unresolved_exact_tie"
            result.terminal_reason = "versa_could_not_separate_surviving_candidates"
        return result

    # A rival must be quoted at least this often before absence counts against
    # a candidate: one stray occurrence is not enough to call a rival invented.
    MIN_ATTESTED_MENTIONS = 3
    # A single character matches far too much unrelated text for its count to
    # mean anything; from two characters up the count is informative, and that
    # matters because short numeric answers ("17") are common here.
    MIN_COUNTABLE_ANSWER_CHARS = 2

    def _apply_corpus_attestation_gate(
        self,
        candidates: list[CandidateEvaluation],
        *,
        evidence: dict[str, Any],
    ) -> GateResult:
        """Defer candidates the fetched pages never state."""
        if len(candidates) <= 1 or not self._corpus_surface_form_is_authoritative(
            evidence
        ):
            return self._pass_through(
                "corpus_attestation",
                candidates,
                "corpus_attestation_not_applicable",
            )
        mentions = self._corpus_mention_counts(evidence, candidates)
        survivors = self._drop_unattested_candidates(candidates, mentions)
        if len(survivors) == len(candidates):
            return self._pass_through(
                "corpus_attestation",
                candidates,
                "every_candidate_attested_or_uncountable",
            )
        return self._gate_from_survivors(
            gate_name="corpus_attestation",
            candidates=candidates,
            survivors=survivors,
            pass_reason="answer_appears_in_fetched_pages",
            reject_reason="answer_absent_from_fetched_pages",
            details={
                item.candidate_key: {"corpus_mentions": mentions.get(item.candidate_key, 0)}
                for item in candidates
            },
            hard=False,
        )

    def _drop_unattested_candidates(
        self,
        candidates: list[CandidateEvaluation],
        corpus_mentions: dict[str, int],
    ) -> list[CandidateEvaluation]:
        """Drop candidates the fetched corpus never states, when a rival is well attested.

        Ranking by mention count does not work: a topic word ("Heaven Sent",
        the episode being asked about) is repeated on every relevant page and
        beats the answer, which is stated once. Absence is the asymmetric
        signal — a candidate quoted nowhere in the corpus, while a rival is
        quoted repeatedly, was invented rather than read.

        Deliberately conservative. A candidate is only dropped when its answer
        is long enough to count reliably, some rival clears the attestation
        floor, and the drop leaves something behind. Measured over saved runs
        this removed a wrong winner twice and never removed a correct answer.
        """
        countable = {
            item.candidate_key: corpus_mentions.get(item.candidate_key, 0)
            for item in candidates
            if len(str(item.answer or "").strip()) >= self.MIN_COUNTABLE_ANSWER_CHARS
        }
        if not countable:
            return candidates
        if max(countable.values()) < self.MIN_ATTESTED_MENTIONS:
            # Nothing is well enough attested for silence to be meaningful.
            return candidates

        survivors = [
            item
            for item in candidates
            if countable.get(item.candidate_key) is None
            or countable.get(item.candidate_key, 0) > 0
        ]
        if not survivors or len(survivors) == len(candidates):
            return candidates
        for item in candidates:
            if item in survivors:
                continue
            item.selection_state = "reserve"
            if "corpus_attestation" not in item.soft_deferred_by:
                item.soft_deferred_by.append("corpus_attestation")
            item.metadata["corpus_mentions"] = countable.get(item.candidate_key, 0)
        return survivors

    def _corpus_surface_form_is_authoritative(self, evidence: dict[str, Any]) -> bool:
        """Do not let source abbreviations override an explicit full-form directive."""

        requirement, _ = self._answer_contract(evidence)
        text = normalize_text(requirement or self.question).casefold()
        return not bool(
            re.search(
                r"\b(?:without|no)\s+(?:any\s+)?abbreviations?\b|"
                r"\b(?:do\s+not|don't)\s+abbreviate\b|\bfull\s+(?:name|form)\b",
                text,
            )
        )

    def _evaluate_candidate(
        self,
        *,
        candidate: AnswerCandidate,
        path_index: dict[tuple[str, str, int], CandidatePathEvaluation],
        verifier_results: list[VerifierScoreByReasoning],
    ) -> CandidateEvaluation:
        member_evaluations = [
            self._evaluate_member(
                member=member,
                path_index=path_index,
                verifier_results=verifier_results,
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
        path_index: dict[tuple[str, str, int], CandidatePathEvaluation],
        verifier_results: list[VerifierScoreByReasoning],
    ) -> dict[str, Any]:
        path = path_index.get(
            (member.normalized_answer, member.agent_id, int(member.run_index))
        )
        if path is not None:
            return {
                "agent_id": member.agent_id,
                "run_index": member.run_index,
                "answer": path.answer,
                "answer_type": path.answer_type,
                "reasoning": path.reasoning,
                "valid": path.valid,
                "contradicted": path.contradicted,
                "support_status": path.evidence_support_status,
                "support_bucket": path.evidence_support_level,
                "direct_support": path.direct_support,
                "agent_confidence": path.agent_confidence,
                "agent_answer_frequency": path.agent_answer_frequency,
                "eligible_run_count": path.eligible_run_count,
                "versa_available": path.versa_available,
                "versa_status": path.versa_status,
                "critical_step_floor": path.critical_step_floor,
                "critical_step_geometric_mean": path.critical_step_geometric_mean,
                "average_verifier_probability": path.average_verifier_probability,
                "critical_step_indices": [],
                "evidence_support": dict(path.evidence_support_metadata),
                "reasoning_parse_quality": path.reasoning_parse_quality,
                "reasoning_versa_eligible": path.reasoning_versa_eligible,
            }

        # Compatibility for saved logs that only contain legacy verifier rows.
        verifier = self._find_verifier_result(member, verifier_results)
        process = self._process_metadata(verifier)
        verifier_support = self._verifier_support_metadata(verifier)
        support_status = str(verifier_support.get("status") or "no_support")
        support_level = str(
            verifier_support.get("support_level")
            or support_level_for_status(support_status).value
        )
        contradicted = support_status == "contradicted"
        valid = bool(
            member.parse_completed
            and member.schema_valid
            and member.eligible_for_winner
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
            "support_bucket": support_level,
            "direct_support": support_level in {
                EvidenceSupportLevel.DIRECT_EVIDENCE.value,
                EvidenceSupportLevel.VERIFIED_DERIVED.value,
                EvidenceSupportLevel.TRUSTED_TOOL_FINAL.value,
            },
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
        hard: bool = True,
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
            hard=hard,
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
        hard: bool = True,
    ) -> GateResult:
        eliminated: list[CandidateGateDecision] = []
        decisions: list[CandidateGateDecision] = []
        for item in candidates:
            passed = item in survivors
            outcome = "pass" if passed else ("reject" if hard else "reserve")
            decision = self._decision(
                item,
                outcome,
                pass_reason if passed else reject_reason,
                details.get(item.candidate_key, {}),
            )
            decisions.append(decision)
            if not passed:
                if hard:
                    item.eligible = False
                    item.rejection_reason = reject_reason
                    item.selection_state = "rejected"
                    item.hard_rejection_reason = reject_reason
                    eliminated.append(decision)
                else:
                    item.selection_state = "reserve"
                    if gate_name not in item.soft_deferred_by:
                        item.soft_deferred_by.append(gate_name)
            elif item.selection_state != "rejected":
                item.selection_state = "active"
        return GateResult(
            gate_name=gate_name,
            survivors=survivors,
            eliminated=eliminated,
            decisions=decisions,
            metadata={
                "gate_strength": "hard" if hard else "soft",
                "reserve_candidate_keys": [
                    item.candidate_key
                    for item in candidates
                    if item.selection_state == "reserve"
                ],
            },
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

    def _corpus_mention_counts(
        self,
        evidence: dict[str, Any],
        candidates: list[CandidateEvaluation],
        *,
        max_corpus_chars: int = 200_000,
    ) -> dict[str, int]:
        """Count exact candidate mentions in already-fetched retrieval text.

        Verification is cheaper than generation: when Versa cannot separate
        surviving candidates, the fetched corpus often can — a candidate that
        actually occurs in retrieved passages beats one that occurs nowhere.
        Uses only text already in the evidence bundle; no new network calls.
        """

        blocks: list[str] = []
        total = 0
        for usage in list(evidence.get("tool_usage") or []):
            if not isinstance(usage, dict):
                continue
            raw = usage.get("raw_result")
            raw = raw if isinstance(raw, dict) else {}
            retrieval = raw.get("retrieval")
            retrieval = retrieval if isinstance(retrieval, dict) else {}
            for round_info in list(retrieval.get("rounds") or []):
                if not isinstance(round_info, dict):
                    continue
                for document in list(round_info.get("documents") or []):
                    if not isinstance(document, dict):
                        continue
                    text = str(document.get("text") or "")
                    if text:
                        blocks.append(text.casefold())
                        total += len(text)
                    if total >= max_corpus_chars:
                        break
            for reference in list(raw.get("unverified_references") or []):
                if isinstance(reference, dict):
                    text = str(reference.get("text") or "")
                    if text:
                        blocks.append(text.casefold())
                        total += len(text)
        if not blocks:
            return {}
        corpus = "\n".join(blocks)[:max_corpus_chars]
        counts: dict[str, int] = {}
        for candidate in candidates:
            answer = str(candidate.answer or "").strip().casefold()
            if not answer:
                counts[candidate.candidate_key] = 0
                continue
            pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(answer)}(?![a-z0-9])"
            )
            counts[candidate.candidate_key] = len(pattern.findall(corpus))
        return counts

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
        return support_level_for_status(support_status).value

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
        raw_contract = evidence.get("task_answer_requirement_contract")
        if isinstance(raw_contract, dict):
            contract = TaskAnswerRequirementContract.from_mapping(
                raw_contract,
                question=self.question,
            )
            if contract.resolved:
                return contract.requirement_text, contract.answer_role
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
        contract = TaskAnswerRequirementContract.build(
            question=self.question,
            answer_requirement=requirement,
            answer_role=role,
        )
        return contract.requirement_text, contract.answer_role

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
