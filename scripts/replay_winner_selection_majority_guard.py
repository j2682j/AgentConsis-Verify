"""Replay winner selection with the trusted-tool majority guard.

Each task's `winner_selection.selection_trace.candidates` block already carries
the per-candidate evaluation fields the selector's gate pipeline reads —
support_status, supporting_run_count, supporting_agent_ids, answer, etc. We can
rebuild CandidateEvaluation objects from that block and feed them straight into
FinalWinnerSelector.resolve_evaluations(), so the guard's effect can be measured
against the exact per-task inputs the live pipeline saw. No LLM calls, no
retrieval, no benchmark re-run needed.

Compares two arms:
  baseline  ratio 0.0 — reproduces the original behaviour
  guarded   ratio 2.0 — a rival with > 2x the trusted answer's supporting runs
                        stays in the survivor set

Usage:
    python scripts/replay_winner_selection_majority_guard.py [--run outputs/level1_final_06]
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
from core.config import CandidateEvaluation
from score.final_winner_selector import FinalWinnerSelector


def load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "tasks").glob("*.json"))
    ]


def rebuild_evaluations(candidate_records: list[dict[str, Any]]) -> list[CandidateEvaluation]:
    """Turn saved candidate dicts back into CandidateEvaluation objects."""
    evaluations = []
    for record in candidate_records:
        evaluation = CandidateEvaluation(
            candidate_key=str(record.get("candidate_key") or ""),
            answer=str(record.get("answer") or ""),
            eligible=bool(record.get("eligible", True)),
            support_status=str(record.get("support_status") or "no_support"),
            direct_support=bool(record.get("direct_support", False)),
            contradicted=bool(record.get("contradicted", False)),
            requirement_status=str(record.get("requirement_status") or "unknown"),
            supporting_agent_ids=list(record.get("supporting_agent_ids") or []),
            supporting_run_count=int(record.get("supporting_run_count") or 0),
            selected_agent_id=str(record.get("selected_agent_id") or ""),
            selected_run_index=int(record.get("selected_run_index") or 0),
            selected_reasoning=str(record.get("selected_reasoning") or ""),
            selected_agent_confidence=float(record.get("selected_agent_confidence") or 0.0),
            selected_agent_answer_frequency=int(
                record.get("selected_agent_answer_frequency") or 0
            ),
            rejection_reason=str(record.get("rejection_reason") or ""),
        )
        # These are lists that CandidateEvaluation carries but which
        # resolve_evaluations reads through fields set above.
        evaluations.append(evaluation)
    return evaluations


def score_task(task: dict[str, Any], *, ratio: float) -> tuple[str, bool, bool]:
    """Return (replay_answer, replayable, replay_correct)."""
    expected = str(task.get("expected") or "")
    ws = ((task.get("network_summary") or {}).get("metadata") or {}).get(
        "winner_selection"
    ) or {}
    trace = ws.get("selection_trace") or {}
    candidates = trace.get("candidates") or []
    if not candidates:
        return "", False, False

    selector = FinalWinnerSelector(
        question=str(task.get("question") or ""),
        trusted_tool_majority_override_ratio=ratio,
    )
    resolution = selector.resolve_evaluations(
        rebuild_evaluations(candidates), evidence={}
    )
    answer = (
        resolution.resolved_answer
        if resolution.resolved_answer
        else (resolution.evaluation.answer if resolution.evaluation else "")
    )
    return answer, True, bool(answer) and exact_match(answer, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    args = parser.parse_args()

    tasks = load_tasks(ROOT / args.run)
    print(f"tasks={len(tasks)}   (offline replay from saved selection_trace)")

    base_correct = guard_correct = 0
    original_correct = 0
    unreplayable = 0
    flip_to_correct: list[str] = []
    flip_to_wrong: list[str] = []
    both_correct = both_wrong = 0

    for task in tasks:
        original_correct += bool(task.get("exact_match"))
        base_ans, replayable, base_ok = score_task(task, ratio=0.0)
        if not replayable:
            unreplayable += 1
            continue
        guard_ans, _, guard_ok = score_task(task, ratio=2.0)
        base_correct += base_ok
        guard_correct += guard_ok
        if base_ok and guard_ok:
            both_correct += 1
        elif not base_ok and not guard_ok:
            both_wrong += 1
        elif not base_ok and guard_ok:
            flip_to_correct.append(
                f"{task.get('task_id','')[:8]}  {base_ans!r} -> {guard_ans!r}  (exp={task.get('expected')!r})"
            )
        else:
            flip_to_wrong.append(
                f"{task.get('task_id','')[:8]}  {base_ans!r} -> {guard_ans!r}  (exp={task.get('expected')!r})"
            )

    print()
    print(f"original pipeline (from output json):   correct={original_correct}/{len(tasks)}")
    print(f"baseline replay  (ratio 0.0):           correct={base_correct}/{len(tasks) - unreplayable} replayable")
    print(f"guarded  replay  (ratio 2.0):           correct={guard_correct}/{len(tasks) - unreplayable} replayable")
    print(f"unreplayable rows (no candidate trace): {unreplayable}")
    print()
    print(f"guarded vs baseline diff (both replayable, so apples-to-apples):")
    print(f"  both correct  = {both_correct}")
    print(f"  both wrong    = {both_wrong}")
    print(f"  flip to correct = {len(flip_to_correct)}  <-- guard win")
    print(f"  flip to wrong   = {len(flip_to_wrong)}   <-- guard regression")
    print(f"  net delta       = {len(flip_to_correct) - len(flip_to_wrong):+d}")
    print()
    print(f"=== flipped: baseline wrong -> guard correct ({len(flip_to_correct)}) ===")
    for row in flip_to_correct:
        print(f"  {row}")
    print()
    print(f"=== flipped: baseline correct -> guard wrong ({len(flip_to_wrong)}) ===")
    for row in flip_to_wrong:
        print(f"  {row}")


if __name__ == "__main__":
    main()
