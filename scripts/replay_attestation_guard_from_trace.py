"""Estimate the corpus-attestation majority guard's effect from saved traces.

For each task's `corpus_attestation` gate result we look at every candidate
that was moved to reserve and ask: would the new guard have rescued it?

Rescue = supporting_run_count >= min_runs AND distinct supporting agents >= min_agents.

Prints, per task where the guard fires:
  - which dropped candidate would come back
  - whether that candidate is the gold answer
  - whether the pipeline was already correct (so a rescue could regress)

This does NOT replay downstream gates: consensus / self-consistency / versa
still run after the rescue. The report is a bounded outlook — the tasks listed
under "correct rival rescued" are the only place the guard can possibly help,
but downstream gates still have to pick the rescued candidate over its rivals.

Usage:
    python scripts/replay_attestation_guard_from_trace.py [--run outputs/level1_final_06]
                                                         [--min-runs 3] [--min-agents 2]
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


def scan(tasks: list[dict[str, Any]], *, min_runs: int, min_agents: int) -> dict[str, Any]:
    fires = 0
    rescue_correct: list[dict[str, Any]] = []
    rescue_wrong: list[dict[str, Any]] = []
    rescue_when_already_correct: list[dict[str, Any]] = []

    for task in tasks:
        md = (task.get("network_summary") or {}).get("metadata") or {}
        trace = (md.get("winner_selection") or {}).get("selection_trace") or {}
        gates = trace.get("gate_trace") or []
        ca = next((g for g in gates if g.get("gate_name") == "corpus_attestation"), None)
        if not ca:
            continue

        cand_map = {c.get("candidate_key"): c for c in trace.get("candidates") or []}
        expected = str(task.get("expected") or "")
        winner_correct = bool(task.get("exact_match"))

        for decision in ca.get("decisions") or []:
            if decision.get("outcome") != "reserve":
                continue
            key = decision.get("candidate_key")
            candidate = cand_map.get(key) or {}
            run_count = int(candidate.get("supporting_run_count") or 0)
            agent_count = len(set(candidate.get("supporting_agent_ids") or []))
            if run_count < min_runs or agent_count < min_agents:
                continue

            fires += 1
            entry = {
                "task": (task.get("task_id") or "")[:8],
                "expected": expected,
                "rescued_key": key,
                "rescued_answer": candidate.get("answer"),
                "run_count": run_count,
                "agent_count": agent_count,
                "current_winner": task.get("predicted"),
                "current_winner_correct": winner_correct,
            }
            if exact_match(candidate.get("answer") or key, expected):
                if winner_correct:
                    rescue_when_already_correct.append(entry)
                else:
                    rescue_correct.append(entry)
            else:
                rescue_wrong.append(entry)

    return {
        "fires": fires,
        "rescue_correct": rescue_correct,
        "rescue_wrong": rescue_wrong,
        "rescue_when_already_correct": rescue_when_already_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-agents", type=int, default=2)
    args = parser.parse_args()

    tasks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / args.run / "tasks").glob("*.json"))
    ]
    print(f"tasks={len(tasks)}  guard=(min_runs={args.min_runs}, min_agents={args.min_agents})")

    summary = scan(tasks, min_runs=args.min_runs, min_agents=args.min_agents)

    print()
    print(f"guard would fire on {summary['fires']} attestation drops")
    print(f"  rescued answer matches gold, pipeline was wrong: {len(summary['rescue_correct'])}  <-- potential wins")
    print(f"  rescued answer is wrong                        : {len(summary['rescue_wrong'])}   <-- returns to survivor pool, downstream must reject")
    print(f"  rescued while pipeline already correct         : {len(summary['rescue_when_already_correct'])}  <-- regression risk")

    for label, rows in (
        ("potential wins", summary["rescue_correct"]),
        ("wrong rescues (downstream must handle)", summary["rescue_wrong"]),
        ("rescued while already correct", summary["rescue_when_already_correct"]),
    ):
        print()
        print(f"=== {label} ({len(rows)}) ===")
        for row in rows:
            print(
                f"  {row['task']}  exp={row['expected']!r:<20}  "
                f"rescued={row['rescued_answer']!r:<24} "
                f"(runs={row['run_count']}, agents={row['agent_count']})  "
                f"pipeline_correct={row['current_winner_correct']}"
            )


if __name__ == "__main__":
    main()
