from __future__ import annotations

import json
from typing import Any

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidateEvaluationBundle,
    CandidatePathEvaluation,
    CandidatePathIdentity,
    CandidateRun,
    VerifierScoreByReasoning,
)
from core.stage2_runner import Stage2Runner
from score.answer_candidate_clusterer import AnswerCandidateClusterer
from score.evidence_support_checker import EvidenceSupportChecker
from score.evidence_support_level import EvidenceSupportLevel


class CandidatePathEvaluator:
    """Evaluate evidence support and optional Versa output once per candidate path."""

    def __init__(
        self,
        *,
        question: str,
        clusterer: AnswerCandidateClusterer,
        evidence_support_checker: EvidenceSupportChecker,
        stage2_runner: Stage2Runner,
    ) -> None:
        self.question = str(question or "").strip()
        self.clusterer = clusterer
        self.evidence_support_checker = evidence_support_checker
        self.stage2_runner = stage2_runner
        self._cache: dict[CandidatePathIdentity, CandidatePathEvaluation] = {}

    def clear_cache(self) -> None:
        """Clear task-local evaluations when Stage1 or evidence changes."""

        self._cache.clear()

    def evaluate_candidates(
        self,
        *,
        candidates: list[AnswerCandidate],
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any],
        enable_versa: bool,
        evidence_revision: int = 0,
    ) -> CandidateEvaluationBundle:
        """Evaluate every member path and return report-compatible verifier rows."""

        paths = [member for candidate in candidates for member in candidate.members]
        evaluations, hits, misses, context_metadata = self._evaluate_paths(
            paths=paths,
            stage1_results=stage1_results,
            evidence=evidence,
            enable_versa=enable_versa,
            evidence_revision=evidence_revision,
        )
        verifier_results = [
            self.to_verifier_result(item)
            for item in evaluations
            if item.versa_available
        ]
        return CandidateEvaluationBundle(
            path_evaluations=evaluations,
            verifier_results=verifier_results,
            evidence_revision=int(evidence_revision or 0),
            support_context_metadata=context_metadata,
            cache_hits=hits,
            cache_misses=misses,
        )

    def evaluate_paths(
        self,
        *,
        paths: list[CandidateRun],
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any],
        enable_versa: bool,
        evidence_revision: int = 0,
    ) -> list[CandidatePathEvaluation]:
        """Evaluate selected paths for early-stop using the same task cache."""

        evaluations, _, _, _ = self._evaluate_paths(
            paths=paths,
            stage1_results=stage1_results,
            evidence=evidence,
            enable_versa=enable_versa,
            evidence_revision=evidence_revision,
        )
        return evaluations

    def _evaluate_paths(
        self,
        *,
        paths: list[CandidateRun],
        stage1_results: list[AgentReasoningSummary],
        evidence: dict[str, Any],
        enable_versa: bool,
        evidence_revision: int,
    ) -> tuple[list[CandidatePathEvaluation], int, int, dict[str, Any]]:
        context = self.evidence_support_checker.prepare_context(
            evidence=evidence,
            question=self.question,
            evidence_revision=evidence_revision,
        )
        evaluations: list[CandidatePathEvaluation] = []
        cache_hits = 0
        cache_misses = 0
        for member in paths:
            identity = CandidatePathIdentity(
                candidate_key=member.normalized_answer,
                agent_id=member.agent_id,
                run_index=int(member.run_index),
                evidence_revision=int(evidence_revision or 0),
            )
            cached = self._cache.get(identity)
            if cached is not None and (
                not enable_versa
                or cached.versa_available
                or not cached.reasoning_versa_eligible
            ):
                evaluations.append(cached)
                cache_hits += 1
                continue

            if cached is None:
                summary = self.clusterer.summary_for_member(stage1_results, member)
                support = self.evidence_support_checker.check_path(
                    context=context,
                    target=summary,
                    candidate_answer=member.answer,
                    reasoning_steps=list(member.reasoning_steps),
                    question=self.question,
                )
                support_payload = self.evidence_support_checker.summary_to_dict(support)
                level = support.support_level
                evaluation = CandidatePathEvaluation(
                    identity=identity,
                    answer=member.answer,
                    answer_type=member.answer_type,
                    valid=bool(
                        member.parse_completed
                        and member.schema_valid
                        and member.eligible_for_winner
                    ),
                    eligible_for_winner=member.eligible_for_winner,
                    schema_valid=member.schema_valid,
                    parse_completed=member.parse_completed,
                    validity_labels=list(member.validity_labels),
                    reasoning=member.reasoning,
                    reasoning_steps=list(member.reasoning_steps),
                    reasoning_parse_quality=member.reasoning_parse_quality,
                    reasoning_versa_eligible=bool(
                        member.reasoning_versa_eligible and member.reasoning_steps
                    ),
                    evidence_support_status=support.status,
                    evidence_support_level=level,
                    contradicted=(level == EvidenceSupportLevel.CONTRADICTED.value),
                    direct_support=level in {
                        EvidenceSupportLevel.DIRECT_EVIDENCE.value,
                        EvidenceSupportLevel.VERIFIED_DERIVED.value,
                        EvidenceSupportLevel.TRUSTED_TOOL_FINAL.value,
                    },
                    step_support_results=list(support.step_results),
                    evidence_support_metadata=support_payload,
                    versa_status="pending" if enable_versa else "disabled",
                    agent_answer_frequency=member.agent_answer_frequency,
                    eligible_run_count=member.eligible_run_count,
                    agent_confidence=member.agent_confidence,
                    metadata={
                        "support_checked_once": True,
                        "evidence_revision": int(evidence_revision or 0),
                    },
                )
                cache_misses += 1
            else:
                evaluation = cached
                cache_hits += 1

            if enable_versa and evaluation.reasoning_versa_eligible:
                verifier = self.stage2_runner.score_reasoning_path(
                    target_agent_id=identity.agent_id,
                    candidate_key=identity.candidate_key,
                    target_run_index=identity.run_index,
                    final_answer=evaluation.answer,
                    reasoning_steps=list(evaluation.reasoning_steps),
                    step_support_results=list(evaluation.step_support_results),
                )
                process = verifier.metadata.get("process_verification", {})
                evaluation.versa_available = True
                evaluation.versa_status = "available"
                evaluation.critical_step_floor = float(
                    process.get("critical_step_floor") or 0.0
                )
                evaluation.critical_step_geometric_mean = float(
                    process.get("critical_step_geometric_mean") or 0.0
                )
                evaluation.average_verifier_probability = float(
                    process.get("average_probability") or verifier.verifier_score
                )
                evaluation.versa_step_scores = list(verifier.step_scores)
                evaluation.metadata["process_verification"] = dict(process)
            elif enable_versa:
                evaluation.versa_status = "unavailable_unreliable_reasoning"

            self._cache[identity] = evaluation
            evaluations.append(evaluation)

        return evaluations, cache_hits, cache_misses, dict(context.metadata)

    @staticmethod
    def to_verifier_result(path: CandidatePathEvaluation) -> VerifierScoreByReasoning:
        """Adapt a cached path evaluation to the existing task report schema."""

        process = dict(path.metadata.get("process_verification") or {})
        process.update(
            {
                "critical_step_floor": path.critical_step_floor,
                "critical_step_geometric_mean": path.critical_step_geometric_mean,
                "average_probability": path.average_verifier_probability,
                "versa_status": path.versa_status,
            }
        )
        payload = {
            "evidence_support": dict(path.evidence_support_metadata),
            "process_verification": process,
        }
        return VerifierScoreByReasoning(
            verifier_id="versa_prm",
            target_agent_id=path.identity.agent_id,
            verifier_score=float(path.average_verifier_probability or 0.0),
            step_scores=list(path.versa_step_scores),
            raw_reply=json.dumps(payload, ensure_ascii=False),
            metadata={
                **payload,
                "candidate_key": path.identity.candidate_key,
                "target_run_index": path.identity.run_index,
                "target_answer": path.answer,
                "reasoning_parse_quality": path.reasoning_parse_quality,
                "reasoning_versa_eligible": path.reasoning_versa_eligible,
            },
        )


__all__ = ["CandidatePathEvaluator"]
