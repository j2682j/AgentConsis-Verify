"""Block lineage per run and per turn, not one row per task.

A single disposition per task hides the thing that matters. The same prepared
evidence goes to three agents, three runs each, and the budget does not treat
them alike: on task 002 in final_23 the first turn left Nemotron 3072 characters
of search evidence and Qwen and Gemma about 3603. Whatever the difference costs,
a task-level row cannot show it, and "which agents lost which blocks" is the
question the delivery hypothesis actually turns on.

So the unit is (run, agent, turn, block), and the run's own recorded
`section_chars` is the check. Reconstructing what the budget produced and
comparing its length against what was recorded is what separates a lineage that
describes this run from one that describes a plausible run. Where they disagree
the row is marked `mismatch` and attributed to nothing.

Scope, stated because it is easy to lose: the population is the 30 tasks wrong
in both final_22 and final_23. This covers seven of them -- the ones previously
filed as "gold reached the context, the agent did not use it", a reading that
came from misinterpreting `tool_context`. Nothing here generalises to the other
23, and nothing here is a statement about the system.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"c:/SCP")

from context.context_budget import ContextBudgetManager
from context.evidence_block_lineage import trace

RUNS = {
    "f22": "c:/SCP/outputs/level1_final_22/tasks",
    "f23": "c:/SCP/outputs/level1_final_23/tasks",
}

#: Seven of the thirty. Named rather than derived so the scope cannot widen by
#: accident when this is re-run.
TASKS = ("002", "007", "018", "019", "044", "047", "053")


def load(directory: str, task: str) -> dict | None:
    matches = [
        p for p in glob.glob(f"{directory}/*.json")
        if os.path.basename(p).startswith(f"{task}_")
    ]
    return json.load(open(matches[0], encoding="utf-8")) if matches else None


def prepared_evidence(record: dict) -> str:
    meta = (record.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
            return str(item["raw_result"].get("summary") or "")
    return ""


def turns_of(run: dict) -> list[dict]:
    """Only trajectory events that actually built a prompt.

    Tool request and result events reuse the turn number and carry no budget;
    counting them as missing prompts would invent delivery failures.
    """

    return [
        event["context_budget"]
        for event in (run.get("trajectory") or [])
        if isinstance(event, dict) and isinstance(event.get("context_budget"), dict)
    ]


def main() -> None:
    manager = ContextBudgetManager()
    rows = []

    for label, directory in RUNS.items():
        for task in TASKS:
            record = load(directory, task)
            if not record:
                continue
            prepared = prepared_evidence(record)
            question = str(record.get("question") or "")
            for agent in (record.get("network_summary") or {}).get("stage1_results") or []:
                for run in agent.get("runs") or []:
                    for index, budget in enumerate(turns_of(run), 1):
                        recorded = int(
                            (budget.get("section_chars") or {}).get("search_result", -1)
                        )
                        rendered = manager.apply(
                            {"question": question, "search_result": prepared}
                        ).sections["search_result"]
                        lineage = trace(prepared, rendered).to_dict()
                        # Length is the only historical anchor: these runs
                        # predate the content hashes, so a match is structural
                        # and never verbatim.
                        fidelity = (
                            "unsupported" if recorded < 0
                            else "structural_match" if len(rendered) == recorded
                            else "mismatch"
                        )
                        rows.append({
                            "run": label,
                            "task": task,
                            "agent": agent.get("agent_id"),
                            "run_index": run.get("run_index"),
                            "turn": index,
                            "recorded_search_chars": recorded,
                            "replayed_search_chars": len(rendered),
                            "fidelity": fidelity,
                            "block_count": lineage["block_count"],
                            "lost_block_ids": lineage["lost_block_ids"],
                        })

    print(f"母體：final_22 與 final_23 皆錯的 30 題中的 {len(TASKS)} 題")
    print(f"單位：(run, task, agent, run_index, turn, block)　共 {len(rows)} 列\n")

    fidelity = Counter(r["fidelity"] for r in rows)
    print(f"=== 重播保真度")
    for name in ("structural_match", "mismatch", "unsupported"):
        print(f"   {name:<20} {fidelity[name]:>4}/{len(rows)}")
    print("   無歷史內容 hash，故上限為 structural_match，不可稱逐字一致")

    usable = [r for r in rows if r["fidelity"] == "structural_match"]
    if not usable:
        print(f"\n   沒有任何一列達到 structural_match —— "
              f"重建與歷史紀錄的 search_result 長度不符，")
        print(f"   因此不歸因於 Context Budget。差異樣本：")
        for row in rows[:6]:
            print(f"      {row['run']} {row['task']} {row['agent']} run{row['run_index']}"
                  f" turn{row['turn']}: 紀錄 {row['recorded_search_chars']}"
                  f"、重建 {row['replayed_search_chars']}")
        print(f"\n   下一步應找出 budget 之外造成差異的處理，"
              f"而不是用不符的重建去判定 block 存活。")
        return

    print(f"\n=== 各 agent 失去的 block（僅 structural_match 的列）")
    by_agent: dict[tuple, set] = defaultdict(set)
    for row in usable:
        by_agent[(row["run"], row["task"], row["agent"])].update(row["lost_block_ids"])
    for key in sorted(by_agent):
        label, task, agent = key
        lost = sorted(by_agent[key])
        print(f"   {label} {task} {agent:<9} 失去 {len(lost)}: {lost}")


if __name__ == "__main__":
    sys.exit(main())
