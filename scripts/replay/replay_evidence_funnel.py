"""Where does the gold answer stop surviving, stage by stage?

Prints the five gold funnel fields per task. The first three are what P2 is
judged on -- retrieval, references, Stage 1 context -- because those are the
stages this replay actually re-runs. The last two come from the recording:
nothing here executes an Agent, so a change that improves delivery cannot be
credited with improving a run or a winner until a live benchmark says so.

Baseline first. Before any delivery change is proposed, the rebuild has to
reproduce what the recorded run produced; a funnel that cannot do that cannot be
trusted to say where an answer died.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

from scripts.replay.evidence_funnel_rebuilder import digest, rebuild

DEFAULT_RUNS = ("level_1_final_20", "level1_final_21")


def carries(haystack: str, gold: str) -> bool:
    """Word boundaries at every length; `CUB` must not match inside `Cuba`."""

    needle = re.sub(r"\s+", " ", str(gold or "")).strip().casefold()
    if not needle:
        return False
    text = re.sub(r"\s+", " ", str(haystack or "")).casefold()
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def recorded_baseline(task: dict[str, Any]) -> dict[str, Any]:
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    raw = next(
        (
            item.get("raw_result")
            for item in (meta.get("tool_usage") or [])
            if isinstance(item, dict)
            and item.get("tool_name") == "search"
            and isinstance(item.get("raw_result"), dict)
        ),
        {},
    ) or {}
    summary = next(
        (
            str(item.get("output_text") or "")
            for item in (meta.get("tool_usage") or [])
            if isinstance(item, dict) and item.get("tool_name") == "search"
        ),
        "",
    )
    return {
        "evidence_items": raw.get("evidence_items") or [],
        "unverified_references": raw.get("unverified_references") or [],
        "summary": summary,
    }


def stage1_texts(task: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for agent in (task.get("network_summary") or {}).get("stage1_results") or []:
        for run in agent.get("runs") or []:
            texts.append(str(run.get("tool_context") or ""))
            for item in run.get("tool_results") or []:
                if isinstance(item, dict):
                    texts.append(str(item.get("output_text") or ""))
    return texts


def report_task(run: str, path: str) -> None:
    task = json.loads(open(path, encoding="utf-8").read())
    task_id = os.path.basename(path).split("_")[0]
    gold = str(task.get("expected") or "")
    replay = rebuild(task, task_id=task_id)
    baseline = recorded_baseline(task)

    documents_text = ""
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        raw = item.get("raw_result") if isinstance(item, dict) else None
        if not isinstance(raw, dict):
            continue
        for entry in (raw.get("retrieval") or {}).get("rounds") or []:
            for document in entry.get("documents") or []:
                documents_text += str((document or {}).get("text") or "") + "\n"

    reference_text = "\n".join(
        str(item.get("text") or "") for item in baseline["unverified_references"]
    )
    winner = (
        ((meta.get("winner_selection") or {}).get("selection_trace") or {}).get(
            "selected_answer"
        )
        or ""
    )

    print(f"--- {run}/{task_id}  gold={gold[:38]!r}")
    print("    分層保真度:")
    for stage in replay.stages:
        print(f"       {stage.name:<12} {stage.fidelity:<16} {stage.detail}")
    print("    baseline 對照:")
    print(f"       strict evidence  重建 {len(replay.strict_evidence_ids):>2}"
          f"  錄製 {len(baseline['evidence_items']):>2}")
    print(f"       references       重建 {len(replay.relaxed_reference_ids):>2}"
          f"  錄製 {len(baseline['unverified_references']):>2}")
    print(f"       rendered summary 重建 {len(replay.rendered_search_context):>5} 字元"
          f"  錄製 {len(baseline['summary']):>5} 字元"
          f"  hash {'相同' if digest(replay.rendered_search_context) == digest(baseline['summary']) else '不同'}")
    print("    gold 漏斗:")
    print(f"       gold_in_retrieved_documents  {carries(documents_text, gold)}")
    print(f"       gold_in_unverified_references {carries(reference_text, gold)}")
    print(f"       gold_in_stage1_context        {carries(replay.budgeted_stage1_context, gold)}"
          f"   （重建，P2 主要判準）")
    print(f"       gold_in_any_stage1_run        "
          f"{any(carries(text, gold) for text in stage1_texts(task))}   （錄製，僅觀察）")
    print(f"       gold_selected_as_winner       {carries(str(winner), gold)}"
          f"   （錄製，僅觀察）")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--tasks", default="004,013")
    args = parser.parse_args(argv)

    for run in args.runs.split(","):
        for task_number in args.tasks.split(","):
            for path in glob.glob(f"c:/SCP/outputs/{run}/tasks/{task_number}_*.json"):
                report_task(run, path)


if __name__ == "__main__":
    sys.exit(main())
