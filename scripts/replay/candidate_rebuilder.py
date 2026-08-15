"""Rebuild winner-selection inputs from a recorded run, without its conclusions.

A selection trace stores each candidate after the gates have run, so it holds
both what the gates were given and what they decided. Restoring the second kind
makes a replay carry its own conclusions in as premises, and the failure is
silent: an A/B on a gate change reports "no difference" because the candidate
the change was about was removed before either arm could evaluate it.

That happened. The requirement-gate repair for task 038 measured as a no-op
across five runs, and the reason was that the recorded `eligible=False` reached
the validity gate first and dropped the gold candidate. The repair was reaching
nothing. Only after these fields were dropped did the candidate survive to the
gate the change was about.

So every post-gate field is ignored here and recomputed. `metadata` is kept
because it carries `member_evaluations`, which is a gate *input* -- the
per-member validity and answer types that `_apply_requirement_gate` grades.
"""

from __future__ import annotations

from typing import Any

from core.config import CandidateEvaluation

#: Written by the gates. Restoring any of these makes the replay circular.
POST_GATE_FIELDS = (
    "eligible",
    "selection_state",
    "rejection_reason",
    "hard_rejection_reason",
    "soft_deferred_by",
    "requirement_status",
)


def rebuild_candidate(row: dict[str, Any]) -> CandidateEvaluation:
    """One recorded candidate as the gates first saw it."""

    return CandidateEvaluation(
        candidate_key=str(row.get("candidate_key") or ""),
        answer=str(row.get("answer") or ""),
        # Deliberately not from the row: the gates decide these.
        eligible=True,
        rejection_reason="",
        requirement_status="unknown",
        selection_state="active",
        hard_rejection_reason="",
        soft_deferred_by=[],
        # Evaluation results, computed before any gate runs.
        contradicted=bool(row.get("contradicted")),
        support_status=str(row.get("support_status") or "no_support"),
        direct_support=bool(row.get("direct_support")),
        supporting_agent_ids=list(row.get("supporting_agent_ids") or []),
        supporting_run_count=int(row.get("supporting_run_count") or 0),
        selected_agent_id=str(row.get("selected_agent_id") or ""),
        selected_run_index=int(row.get("selected_run_index") or 0),
        selected_reasoning=str(row.get("selected_reasoning") or ""),
        selected_agent_confidence=float(row.get("selected_agent_confidence") or 0.0),
        selected_agent_answer_frequency=int(
            row.get("selected_agent_answer_frequency") or 0
        ),
        step_score_median=float(row.get("step_score_median") or 0.0),
        critical_step_floor=float(row.get("critical_step_floor") or 0.0),
        critical_step_geometric_mean=float(
            row.get("critical_step_geometric_mean") or 0.0
        ),
        average_verifier_probability=float(
            row.get("average_verifier_probability") or 0.0
        ),
        metadata=dict(row.get("metadata") or {}),
    )


def rebuild_candidates(trace: dict[str, Any]) -> list[CandidateEvaluation]:
    return [rebuild_candidate(row) for row in (trace.get("candidates") or [])]


def rebuildable(trace: dict[str, Any]) -> bool:
    """Whether the trace carries the gate inputs at all.

    `member_evaluations` is the one that cannot be reconstructed from anything
    else, and without it the requirement gate grades nothing.
    """

    rows = trace.get("candidates") or []
    if not rows:
        return False
    return any((row.get("metadata") or {}).get("member_evaluations") for row in rows)


__all__ = [
    "POST_GATE_FIELDS",
    "rebuild_candidate",
    "rebuild_candidates",
    "rebuildable",
]
