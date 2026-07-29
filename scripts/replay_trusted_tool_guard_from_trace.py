"""Estimate the trusted-tool majority guard's effect from saved gate traces.

Rebuilding the full winner pipeline offline loses fidelity — replay drops
evidence context and the contradiction gate over-prunes. Instead, look at the
saved `evidence_support` gate result directly and check whether the guard would
have rescued a rival candidate:

  1. best bucket was trusted_tool_final
  2. rival's supporting_run_count > best trusted supporting_run_count * ratio

That is exactly what the code path we changed evaluates, and the gate trace
carries every input it reads. If the rescue would fire, note the rival that
would join the survivor pool and would then feed the downstream consensus /
self-consistency / versa gates.

This does NOT predict the final winner (that depends on downstream gates that
also need live evidence). It answers a narrower, more reliable question:
"which tasks would the guard have opened up for reconsideration, and what
answer would compete with the trusted one".

Usage:
    python scripts/replay_trusted_tool_guard_from_trace.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.gaia.answer_matcher import exact_match


TRUSTED_STATUS = "tool_final_supported"


def guard_effect(task: dict[str, Any], *, ratio: float) -> dict[str, Any] | None:
    ws = ((task.get("network_summary") or {}).get("metadata") or {}).get(
        "winner_selection"
    ) or {}
    trace = ws.get("selection_trace") or {}
    candidates = trace.get("candidates") or []
    if not candidates:
        return None
    trusted = [
        c for c in candidates if c.get("support_status") == TRUSTED_STATUS
    ]
    if not trusted:
        return None
    trusted_top = max(int(c.get("supporting_run_count") or 0) for c in trusted)
    threshold = trusted_top * ratio
    rivals = [
        c
        for c in candidates
        if c.get("support_status") != TRUSTED_STATUS
        and int(c.get("supporting_run_count") or 0) > threshold
    ]
    if not rivals:
        return None
    rivals_sorted = sorted(
        rivals,
        key=lambda c: int(c.get("supporting_run_count") or 0),
        reverse=True,
    )
    return {
        "trusted_key": trusted[0].get("candidate_key"),
        "trusted_runs": trusted_top,
        "rivals": [
            {
                "key": c.get("candidate_key"),
                "answer": c.get("answer"),
                "runs": int(c.get("supporting_run_count") or 0),
                "agents": list(c.get("supporting_agent_ids") or []),
            }
            for c in rivals_sorted
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument("--ratio", type=float, default=2.0)
    args = parser.parse_args()

    tasks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((Path(args.run) / "tasks").glob("*.json"))
    ]

    total_with_trusted = 0
    would_fire = []
    would_fire_correct = []
    would_fire_wrong = []
    already_correct = 0

    for task in tasks:
        expected = str(task.get("expected") or "")
        original_correct = bool(task.get("exact_match"))
        winner = str(task.get("predicted") or "")

        ws = ((task.get("network_summary") or {}).get("metadata") or {}).get(
            "winner_selection"
        ) or {}
        candidates = (ws.get("selection_trace") or {}).get("candidates") or []
        if any(c.get("support_status") == TRUSTED_STATUS for c in candidates):
            total_with_trusted += 1

        effect = guard_effect(task, ratio=args.ratio)
        if effect is None:
            continue

        would_fire.append(
            {
                "task_id": task.get("task_id", "")[:8],
                "expected": expected,
                "current_winner": winner,
                "current_correct": original_correct,
                **effect,
            }
        )
        if original_correct:
            already_correct += 1
        best_rival = effect["rivals"][0]
        if exact_match(best_rival["answer"] or best_rival["key"], expected):
            would_fire_correct.append(would_fire[-1])
        elif not original_correct:
            would_fire_wrong.append(would_fire[-1])

    print(f"tasks: {len(tasks)}")
    print(f"tasks with a trusted_tool_final candidate: {total_with_trusted}")
    print(f"tasks where guard (ratio {args.ratio}) would fire: {len(would_fire)}")
    print()
    print(f"  best rival's answer matches gold  = {len(would_fire_correct)}  (potential wins)")
    print(f"  guard fires but already correct   = {already_correct}  (risk of regression)")
    print(f"  guard fires and would surface a wrong rival = {len(would_fire_wrong)}")
    print()
    print("=== tasks where guard would surface a correct rival ===")
    for row in would_fire_correct:
        rival = row["rivals"][0]
        print(
            f"  {row['task_id']}  exp={row['expected']!r:<20}  "
            f"trusted={row['trusted_key']!r}({row['trusted_runs']}r)  "
            f"rival={rival['key']!r}({rival['runs']}r,{len(rival['agents'])}agents)"
        )
    print()
    print("=== tasks where guard fires but rival is wrong (must lose to trusted downstream) ===")
    for row in would_fire_wrong:
        rival = row["rivals"][0]
        print(
            f"  {row['task_id']}  exp={row['expected']!r:<20}  "
            f"trusted={row['trusted_key']!r}({row['trusted_runs']}r)  "
            f"rival={rival['key']!r}({rival['runs']}r)  "
            f"current_winner={row['current_winner']!r}"
        )
    print()
    print("=== tasks where guard fires but pipeline already correct ===")
    for row in would_fire:
        if not row["current_correct"]:
            continue
        rival = row["rivals"][0]
        print(
            f"  {row['task_id']}  exp={row['expected']!r:<20}  "
            f"trusted={row['trusted_key']!r}({row['trusted_runs']}r)  "
            f"rival={rival['key']!r}({rival['runs']}r)  "
            f"current_winner={row['current_winner']!r}"
        )


if __name__ == "__main__":
    main()
