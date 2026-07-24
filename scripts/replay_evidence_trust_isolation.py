"""Offline replay of Plan-13 evidence trust isolation over saved GAIA task JSONs.

Simulates the effect of stages 2 and 4 (relaxed passages lose all support
authority) on winner selection, without re-running agents or retrieval.

In the replayed run every search task recorded ``strict_evidence_count == 0``,
so all search-derived support came from relaxed passages. After the isolation
those candidates can only reach ``no_support``. This script downgrades them
accordingly, re-runs the ordered gates via
``FinalWinnerSelector.resolve_evaluations``, and reports:

- winner changes, split into recoveries / regressions / neutral churn,
- whether previously-correct tasks keep their answer (the acceptance gate),
- how many tasks still win via relaxed-derived support (target: zero).

Usage:
    python scripts/replay_evidence_trust_isolation.py outputs/level1_full_system_final_2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.gaia.answer_matcher import exact_match
from core.config import CandidateEvaluation
from score.final_winner_selector import FinalWinnerSelector


# Support statuses that, in this run, could only have come from search
# evidence — which was relaxed-only (strict_evidence_count == 0 everywhere).
SEARCH_DERIVED_STATUSES = {
    "search_evidence_supported",
    "tool_intermediate_supported",
}
# Support that does not come from the search channel and is therefore
# untouched by the evidence trust contract.
NON_SEARCH_STATUSES = {
    "attachment_evidence_supported",
    "derived_evidence_supported",
    "tool_final_supported",
}


def rebuild_evaluations(trace: dict) -> list[CandidateEvaluation]:
    field_names = set(CandidateEvaluation.__dataclass_fields__)
    out = []
    for row in trace.get("candidates") or []:
        if isinstance(row, dict):
            out.append(
                CandidateEvaluation(
                    **{k: v for k, v in row.items() if k in field_names}
                )
            )
    return out


def isolate(evaluations: list[CandidateEvaluation], *, search_used: bool) -> int:
    """Downgrade relaxed-derived support in place; return how many changed."""
    if not search_used:
        return 0
    changed = 0
    for item in evaluations:
        if str(item.support_status or "") in SEARCH_DERIVED_STATUSES:
            item.support_status = "no_support"
            item.direct_support = False
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "tasks", "*.json")))
    if not files:
        print(f"no task JSONs under {args.output_dir}")
        return 1

    rows = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        meta = (data.get("network_summary") or {}).get("metadata") or {}
        trace = (meta.get("winner_selection") or {}).get("selection_trace") or {}
        question = str(data.get("question") or "")
        expected = str(data.get("expected") or "")
        recorded = str(data.get("predicted") or "")
        search_used = bool(meta.get("search_used"))
        evidence = {
            "routing": meta.get("routing") or {},
            "tool_usage": meta.get("tool_usage") or [],
        }

        # Baseline: replay unchanged, to prove the harness reproduces the run.
        base = FinalWinnerSelector(question=question).resolve_evaluations(
            rebuild_evaluations(trace), evidence=evidence
        )
        base_answer = (
            base.evaluation.answer if base.evaluation is not None else base.resolved_answer
        )

        evaluations = rebuild_evaluations(trace)
        downgraded = isolate(evaluations, search_used=search_used)
        new = FinalWinnerSelector(question=question).resolve_evaluations(
            evaluations, evidence=evidence
        )
        new_answer = (
            new.evaluation.answer if new.evaluation is not None else new.resolved_answer
        )

        rows.append(
            {
                "task": os.path.basename(path)[:3],
                "expected": expected,
                "recorded": recorded,
                "recorded_exact": bool(data.get("exact_match")),
                "base": base_answer,
                "base_matches_recorded": base_answer.strip() == recorded.strip(),
                "new": new_answer,
                "new_exact": exact_match(new_answer, expected),
                "new_reason": new.reason,
                "downgraded": downgraded,
                "search_used": search_used,
            }
        )

    fidelity = [r for r in rows if not r["base_matches_recorded"]]
    if fidelity:
        print(f"FIDELITY WARNING: {len(fidelity)} task(s) where unchanged replay != recorded")
        for r in fidelity[:10]:
            print(f"  {r['task']}: recorded={r['recorded']!r} replayed={r['base']!r}")
    else:
        print(f"FIDELITY: unchanged replay reproduces all {len(rows)} recorded answers")
    print()

    old_exact = sum(1 for r in rows if r["recorded_exact"])
    new_exact = sum(1 for r in rows if r["new_exact"])
    regressions = [r for r in rows if r["recorded_exact"] and not r["new_exact"]]
    gains = [r for r in rows if not r["recorded_exact"] and r["new_exact"]]
    changed = [r for r in rows if r["new"].strip() != r["recorded"].strip()]
    touched = [r for r in rows if r["downgraded"]]

    print(f"tasks: {len(rows)}   support-downgraded tasks: {len(touched)}")
    print(f"exact: {old_exact} -> {new_exact}")
    print(f"answers changed: {len(changed)}")
    print()
    if regressions:
        print(f"!! REGRESSIONS ({len(regressions)}): previously correct, now wrong")
        for r in regressions:
            print(
                f"  {r['task']}: {r['recorded']!r} -> {r['new']!r}"
                f"  (expected {r['expected']!r}, {r['new_reason']})"
            )
    else:
        print("no regressions: every previously exact task keeps its answer")
    if gains:
        print(f"GAINS ({len(gains)}):")
        for r in gains:
            print(f"  {r['task']}: {r['recorded']!r} -> {r['new']!r}")
    if changed:
        print("\nall changed answers:")
        for r in changed:
            mark = "+" if r["new_exact"] else " "
            print(
                f"  [{mark}] {r['task']}: {r['recorded'][:34]!r} -> {r['new'][:34]!r}"
                f" | exp {r['expected'][:28]!r} | {r['new_reason']}"
            )
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
