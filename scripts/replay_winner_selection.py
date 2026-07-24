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

        old_answer = None
        if old_selector_cls is not None:
            old_selector = old_selector_cls(question=question)
            old_answer, old_status = run_old_gates(
                old_selector, rebuild_evaluations(trace), evidence
            )
            if old_answer.strip() != recorded.strip():
                fidelity_breaks.append(
                    (os.path.basename(path), recorded, old_answer, old_status)
                )

        new_selector = FinalWinnerSelector(question=question)
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
