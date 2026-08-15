"""Replay winner selection over a recorded run, and A/B a change against it.

Two things this reports that the ad-hoc replays did not.

First, fidelity. A replay that does not reproduce the recorded winner is still
usable for an A/B -- both arms share the same reconstructed inputs -- but it is
not evidence about production, and saying "52 of 53 unchanged" without that
distinction overstates what was shown. Each task is graded `exact`,
`candidate_only`, `approximate` or `unsupported`.

Second, the intervention is a parameter rather than an edit. An A/B runs the
same reconstruction twice with one callable applied, so the only difference
between arms is the change under test.

Usage:
    from scripts.replay.replay_winner_selection import ab_replay, replay_run

    report = ab_replay("level1_final_21", disable=lambda: patch_out_my_change())
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import glob
import json
import os
from typing import Any, Callable, Iterator

from benchmark.gaia.answer_matcher import exact_match
from score.final_winner_selector import FinalWinnerSelector
from scripts.replay.candidate_rebuilder import rebuild_candidates, rebuildable

REPLAY_SCHEMA_VERSION = "1.0"


@dataclass
class TaskReplay:
    task_id: str
    gold: str
    recorded_winner: str
    replayed_winner: str
    fidelity: str
    candidate_set_match: bool

    @property
    def recorded_correct(self) -> bool:
        return exact_match(self.recorded_winner, self.gold)

    @property
    def replayed_correct(self) -> bool:
        return exact_match(self.replayed_winner, self.gold)


@dataclass
class RunReplay:
    source_run: str
    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    tasks: list[TaskReplay] = field(default_factory=list)

    def fidelity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.fidelity] = counts.get(task.fidelity, 0) + 1
        return counts


def _task_files(run: str) -> list[str]:
    return sorted(glob.glob(f"c:/SCP/outputs/{run}/tasks/*.json"))


def _trace(task: dict[str, Any]) -> dict[str, Any]:
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    return ((meta.get("winner_selection") or {}).get("selection_trace") or {})


def _evidence(task: dict[str, Any]) -> dict[str, Any]:
    """The evidence bundle the gates read, including the fetched corpus.

    `tool_usage` matters: `_corpus_mention_counts` reads the retrieval text out
    of it, so a replay that omits it hands corpus attestation an empty corpus
    and the gate silently does nothing. An A/B on an attestation change would
    then compare two no-ops and report that the change is safe.
    """

    meta = (task.get("network_summary") or {}).get("metadata") or {}
    routing = meta.get("routing") or {}
    return {
        "answer_requirement": str(routing.get("answer_requirement") or ""),
        "answer_role": str(routing.get("answer_role") or ""),
        "tool_usage": list(meta.get("tool_usage") or []),
    }


def replay_run(run: str) -> RunReplay:
    """Every task in one recorded run, re-decided from its gate inputs."""

    report = RunReplay(source_run=run)
    for path in _task_files(run):
        task = json.loads(open(path, encoding="utf-8").read())
        task_id = os.path.basename(path).split("_")[0]
        trace = _trace(task)
        gold = str(task.get("expected") or "")
        recorded = str(trace.get("selected_answer") or "")
        if not trace.get("candidates"):
            report.tasks.append(
                TaskReplay(task_id, gold, recorded, "", "unsupported", False)
            )
            continue

        candidates = rebuild_candidates(trace)
        selection = FinalWinnerSelector(
            question=str(task.get("question") or "")
        ).resolve_evaluations(candidates, evidence=_evidence(task))
        replayed = str((selection.evaluation.answer if selection.evaluation else "") or "")

        recorded_keys = {str(c.get("candidate_key")) for c in trace["candidates"]}
        replayed_keys = {c.candidate_key for c in selection.evaluations}
        same_set = recorded_keys == replayed_keys

        if not rebuildable(trace):
            fidelity = "approximate"
        elif replayed == recorded and same_set:
            fidelity = "exact"
        elif same_set:
            fidelity = "candidate_only"
        else:
            fidelity = "approximate"
        report.tasks.append(
            TaskReplay(task_id, gold, recorded, replayed, fidelity, same_set)
        )
    return report


@contextmanager
def _applied(patch: Callable[[], Callable[[], None]] | None) -> Iterator[None]:
    """Run the body with `patch` in force; `patch` returns its own undo."""

    undo = patch() if patch else None
    try:
        yield
    finally:
        if undo:
            undo()


def ab_replay(
    run: str,
    *,
    disable: Callable[[], Callable[[], None]],
) -> dict[str, Any]:
    """Compare the current code against itself with `disable` applied.

    `disable` turns the change under test off and returns a callable restoring
    it, so both arms run the same reconstruction and differ only by the change.
    """

    with _applied(disable):
        without = {t.task_id: t for t in replay_run(run).tasks}
    with_change = {t.task_id: t for t in replay_run(run).tasks}

    changed = []
    regressions = []
    for task_id, after in with_change.items():
        before = without.get(task_id)
        if before is None or before.replayed_winner == after.replayed_winner:
            continue
        changed.append(
            {
                "task_id": task_id,
                "gold": after.gold,
                "before": before.replayed_winner,
                "after": after.replayed_winner,
                "before_correct": before.replayed_correct,
                "after_correct": after.replayed_correct,
                # Graded with the change off. Grading it on would mark every
                # task the change is *meant* to move as unfaithful, because the
                # replayed winner would differ from the recorded one by design.
                "baseline_fidelity": before.fidelity,
            }
        )
        if before.replayed_correct and not after.replayed_correct:
            regressions.append(task_id)

    return {
        "source_run": run,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "baseline_fidelity": RunReplay(run, tasks=list(without.values())).fidelity_counts(),
        "correct_without": sum(1 for t in without.values() if t.replayed_correct),
        "correct_with": sum(1 for t in with_change.values() if t.replayed_correct),
        "recorded_correct": sum(1 for t in with_change.values() if t.recorded_correct),
        "changed_by_intervention": changed,
        "regressions": regressions,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"{report['source_run']:<20} 保真度 {report['baseline_fidelity']}")
    print(
        f"    錄製正確 {report['recorded_correct']}  "
        f"重播正確 關 {report['correct_without']} → 開 {report['correct_with']}  "
        f"改變 {len(report['changed_by_intervention'])}  "
        f"退步 {report['regressions'] or '無'}"
    )
    for row in report["changed_by_intervention"]:
        mark = lambda ok: "✓" if ok else "✗"
        print(
            f"      {row['task_id']}  {mark(row['before_correct'])} {row['before'][:30]!r}"
            f"  →  {mark(row['after_correct'])} {row['after'][:30]!r}"
            f"   [baseline {row['baseline_fidelity']}]"
        )


__all__ = ["RunReplay", "TaskReplay", "ab_replay", "replay_run", "print_report"]
