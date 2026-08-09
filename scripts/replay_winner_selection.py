"""Offline replay of final winner selection over saved GAIA task JSONs.

Rebuilds each task's recorded CandidateEvaluation list from
``network_summary.metadata.winner_selection.selection_trace.candidates`` and
re-runs the ordered-gate pipeline, so selector changes can be validated
against a finished benchmark run without re-executing agents or retrieval.

Usage:
    python scripts/replay_winner_selection.py outputs/level1_40_system_final \
        [--old-selector PATH]

With ``--old-selector`` (a copy of the previous final_winner_selector.py,
for example from ``git show <rev>:score/final_winner_selector.py``) the
script first replays the old logic and checks it reproduces the recorded
answers, which validates the reconstruction itself.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.gaia.answer_matcher import exact_match, partial_match
from core.config import CandidateEvaluation
from score.final_winner_selector import FinalWinnerSelector


def load_old_selector_class(path: str):
    spec = importlib.util.spec_from_file_location("old_final_winner_selector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FinalWinnerSelector


def rebuild_evaluations(trace: dict) -> list[CandidateEvaluation]:
    field_names = set(CandidateEvaluation.__dataclass_fields__)
    evaluations = []
    for row in trace.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        kwargs = {key: value for key, value in row.items() if key in field_names}
        evaluations.append(CandidateEvaluation(**kwargs))
    return evaluations


def replay_evidence(metadata: dict) -> dict:
    return {
        "routing": metadata.get("routing") or {},
        "tool_usage": metadata.get("tool_usage") or [],
    }


def seed_requirement_gate(selector, trace: dict) -> None:
    """Take the requirement gate's verdict from the record instead of redoing it.

    That gate re-derives the answer contract from the evidence bundle, and the
    bundle's ``task_answer_requirement_contract`` is never written to the task
    JSON, so a replay cannot rebuild it. On level1_final_15 task 050 the
    reconstructed contract rejected every candidate. Replaying just this gate
    from the record takes fidelity from 52/53 to 53/53 on level1_final_13, _15
    and _16, and costs nothing for selector work, because every gate that
    ranks candidates -- consensus, self-consistency, versa -- runs after it.
    """

    recorded = next(
        (
            gate
            for gate in trace.get("gate_trace") or []
            if gate.get("gate_name") == "answer_requirement"
        ),
        None,
    )
    if recorded is None:
        return
    survivor_keys = [str(key) for key in recorded.get("survivors") or []]

    def replayed(candidates, *, evidence):
        from score.final_winner_selector import GateResult

        by_key = {item.candidate_key: item for item in candidates}
        decisions = []
        for decision in recorded.get("decisions") or []:
            candidate = by_key.get(str(decision.get("candidate_key")))
            if candidate is None:
                continue
            outcome = str(decision.get("outcome") or "pass")
            reason = str(decision.get("reason") or "")
            if outcome == "reject":
                candidate.selection_state = "rejected"
                candidate.hard_rejection_reason = reason
            elif outcome == "reserve":
                candidate.selection_state = "reserve"
                if "answer_requirement" not in candidate.soft_deferred_by:
                    candidate.soft_deferred_by.append("answer_requirement")
            decisions.append(selector._decision(candidate, outcome, reason))
        return GateResult(
            gate_name="answer_requirement",
            survivors=[by_key[key] for key in survivor_keys if key in by_key],
            eliminated=[item for item in decisions if item.outcome == "reject"],
            decisions=decisions,
            metadata={"replayed_from_record": True},
        )

    selector._apply_requirement_gate = replayed


def agent_chosen_keys(data: dict) -> list[str]:
    """What `select()` sets before the gates and `resolve_evaluations` does not.

    `_consensus_rank` reads it to rank agents that settled on an answer ahead
    of agents that merely produced it in some run, so leaving it empty changes
    the consensus ordering.
    """

    from utils.network_utils import normalize_text

    return [
        normalize_text(str(agent.get("compressed_answer") or "")).casefold()
        for agent in (data.get("network_summary") or {}).get("stage1_results") or []
        if normalize_text(str(agent.get("compressed_answer") or ""))
    ]


def run_old_gates(selector, evaluations, evidence):
    """Reproduce the pre-change select() decision loop on saved evaluations."""
    for evaluation in evaluations:
        evaluation.selection_state = "active"
        evaluation.hard_rejection_reason = ""
        evaluation.soft_deferred_by = []
    survivors = list(evaluations)
    gates = (
        selector._apply_validity_gate,
        selector._apply_requirement_gate,
        selector._apply_contradiction_gate,
        selector._apply_evidence_gate,
        selector._apply_cross_agent_gate,
        selector._apply_self_consistency_gate,
        selector._apply_versa_gate,
    )
    for gate in gates:
        result = gate(survivors, evidence=evidence)
        survivors = result.survivors
        if result.terminal_status:
            return "", result.terminal_status
        # The recorded runs never reached evidence_only_resolution, and the
        # replay evidence bundle lacks the fact store it would need, so the
        # all-unsupported hook is intentionally skipped here.
    if not survivors:
        return "", "no_eligible_candidate"
    if len(survivors) > 1:
        return "", "unresolved_exact_tie"
    selected = survivors[0]
    from score.evidence_support_level import EvidenceSupportLevel

    if (
        selector._is_factual_search(evidence)
        and selector._support_bucket(selected.support_status)
        == EvidenceSupportLevel.UNSUPPORTED.value
        and not bool(selected.metadata.get("versa_available"))
    ):
        return "", "unresolved_factual_without_support"
    return selected.answer, "answerable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="GAIA report directory containing tasks/*.json")
    parser.add_argument("--old-selector", default="", help="path to the previous final_winner_selector.py for baseline fidelity replay")
    args = parser.parse_args()

    old_selector_cls = (
        load_old_selector_class(args.old_selector) if args.old_selector else None
    )

    task_files = sorted(glob.glob(os.path.join(args.output_dir, "tasks", "*.json")))
    if not task_files:
        print(f"no task JSONs under {args.output_dir}")
        return 1

    rows = []
    fidelity_breaks = []
    for path in task_files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        metadata = (data.get("network_summary") or {}).get("metadata") or {}
        trace = (metadata.get("winner_selection") or {}).get("selection_trace") or {}
        question = str(data.get("question") or "")
        expected = str(data.get("expected") or "")
        recorded = str(data.get("predicted") or "")
        evidence = replay_evidence(metadata)

        chosen_keys = agent_chosen_keys(data)

        old_answer = None
        if old_selector_cls is not None:
            old_selector = old_selector_cls(question=question)
            old_selector._agent_chosen_keys = list(chosen_keys)
            seed_requirement_gate(old_selector, trace)
            old_answer, old_status = run_old_gates(
                old_selector, rebuild_evaluations(trace), evidence
            )
            if old_answer.strip() != recorded.strip():
                fidelity_breaks.append(
                    (os.path.basename(path), recorded, old_answer, old_status)
                )

        new_selector = FinalWinnerSelector(question=question)
        new_selector._agent_chosen_keys = list(chosen_keys)
        seed_requirement_gate(new_selector, trace)
        selection = new_selector.resolve_evaluations(
            rebuild_evaluations(trace), evidence=evidence
        )
        new_answer = (
            selection.evaluation.answer
            if selection.evaluation is not None
            else selection.resolved_answer
        )
        rows.append(
            {
                "task": os.path.basename(path)[:3],
                "task_id": str(data.get("task_id") or ""),
                "expected": expected,
                "recorded": recorded,
                "recorded_exact": bool(data.get("exact_match")),
                "new": new_answer,
                "new_status": selection.status,
                "new_reason": selection.reason,
                "new_exact": exact_match(new_answer, expected),
                "new_partial": partial_match(new_answer, expected),
            }
        )

    if old_selector_cls is not None:
        if fidelity_breaks:
            print(f"FIDELITY: {len(fidelity_breaks)} task(s) where old-logic replay != recorded answer")
            for name, recorded, replayed, status in fidelity_breaks:
                print(f"  {name}: recorded={recorded!r} replayed={replayed!r} ({status})")
        else:
            print(f"FIDELITY: old-logic replay reproduces all {len(task_files)} recorded answers")
        print()

    regressions = [r for r in rows if r["recorded_exact"] and not r["new_exact"]]
    gains = [r for r in rows if not r["recorded_exact"] and r["new_exact"]]
    changed = [r for r in rows if r["new"].strip() != r["recorded"].strip()]
    empties_fixed = [r for r in rows if not r["recorded"].strip() and r["new"].strip()]

    old_exact = sum(1 for r in rows if r["recorded_exact"])
    new_exact = sum(1 for r in rows if r["new_exact"])
    new_partial = sum(1 for r in rows if r["new_partial"])
    old_partial = sum(
        1
        for r, path in zip(rows, task_files)
        if json.load(open(path, encoding="utf-8")).get("partial_match")
    )

    print(f"tasks: {len(rows)}")
    print(f"exact:   {old_exact} -> {new_exact}")
    print(f"partial: {old_partial} -> {new_partial}")
    print(f"changed answers: {len(changed)}, empty answers filled: {len(empties_fixed)}")
    print()
    if regressions:
        print("REGRESSIONS (was exact, now wrong):")
        for r in regressions:
            print(f"  {r['task']}: {r['recorded']!r} -> {r['new']!r} (expected {r['expected']!r})")
    else:
        print("no regressions: every previously exact task stays exact")
    if gains:
        print("GAINS (was wrong, now exact):")
        for r in gains:
            print(f"  {r['task']}: {r['recorded']!r} -> {r['new']!r}")
    if changed:
        print("\nall changed answers:")
        for r in changed:
            mark = "+" if r["new_exact"] else ("~" if r["new_partial"] else " ")
            print(
                f"  [{mark}] {r['task']}: {r['recorded']!r} -> {r['new']!r}"
                f" | expected {r['expected']!r} | {r['new_reason']}"
            )
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
