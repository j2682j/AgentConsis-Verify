"""The tasks that were wrong in final_22 and wrong again in final_23.

Either run on its own mixes two populations: tasks that fail for a reason, and
tasks that happened to fail this time. Three tasks moved between the two runs and
the net was minus one, which is inside the four-to-six churn measured earlier, so
single-run failure lists are partly noise. The intersection is not. A task wrong
twice, under two independent samplings of the agents and the web, is failing for
something the system does rather than something it rolled.

Each one is placed at the furthest point its gold answer reached:

    never_searched          no retrieval ran at all
    not_in_documents        retrieval ran, gold never appeared
    stuck_at_documents      gold in the fetched documents, absent downstream
    stuck_at_references     gold reached the references, not the context
    stuck_at_context        gold reached Stage 1, no run produced it
    stuck_at_runs           a run produced it, selection chose something else
    delivered_but_wrong     gold present throughout and still scored wrong

The last two are the expensive ones. Everything upstream of them is a retrieval
problem; those two are the system discarding an answer it already had, and they
are counted separately for that reason.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, r"c:/SCP")

RUNS = {
    "f22": "c:/SCP/outputs/level1_final_22/tasks",
    "f23": "c:/SCP/outputs/level1_final_23/tasks",
}


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def load(directory: str) -> dict[str, dict]:
    return {
        os.path.basename(path)[:3]: json.load(open(path, encoding="utf-8"))
        for path in sorted(glob.glob(f"{directory}/*.json"))
    }


def search_raw(record: dict) -> dict:
    meta = (record.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
            return item["raw_result"]
    return {}


def components(gold: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*[,;]\s*", gold) if p.strip()]
    return parts if len(parts) > 1 else [gold]


def contains_gold(text: str, gold: str) -> bool:
    if not gold:
        return False
    folded = normalise(text).casefold()
    return all(
        re.search(rf"(?<![\w]){re.escape(p.casefold())}(?![\w])", folded)
        for p in components(gold)
    )


def document_text(raw: dict) -> str:
    """Everything retrieval actually fetched, from the record itself.

    `sources` is empty on every task, so reading it reported the gold as absent
    from documents on all 29 -- which contradicted a separate measurement over
    the same corpora. The documents are in `retrieval.rounds`.
    """

    parts = []
    for round_ in (raw.get("retrieval") or {}).get("rounds") or []:
        for document in round_.get("documents") or []:
            if isinstance(document, dict):
                parts.append(str(document.get("text") or ""))
                parts.append(str(document.get("title") or ""))
    return "\n".join(parts)


def stage1(record: dict) -> tuple[list[str], list[str]]:
    runs = [
        run
        for agent in ((record.get("network_summary") or {}).get("stage1_results") or [])
        for run in (agent.get("runs") or [])
        if isinstance(run, dict)
    ]
    return (
        [normalise(r.get("tool_context")) for r in runs if r.get("tool_context")],
        [normalise(r.get("final_answer")) for r in runs if r.get("final_answer")],
    )


def classify(record: dict, gold: str) -> tuple[str, dict[str, bool]]:
    raw = search_raw(record)
    contexts, answers = stage1(record)

    reached = {
        "documents": contains_gold(document_text(raw), gold),
        "references": contains_gold(
            json.dumps(raw.get("unverified_references") or [], ensure_ascii=False), gold
        ),
        "strict": contains_gold(
            json.dumps(raw.get("verified_evidence_items") or [], ensure_ascii=False), gold
        ),
        "context": any(contains_gold(c, gold) for c in contexts),
        "runs": any(contains_gold(a, gold) for a in answers),
    }

    if not raw:
        return "never_searched", reached
    if reached["runs"]:
        return "stuck_at_runs", reached
    if reached["context"]:
        return "stuck_at_context", reached
    if reached["references"] or reached["strict"]:
        return "stuck_at_references", reached
    if reached["documents"]:
        return "stuck_at_documents", reached
    return "not_in_documents", reached


def main() -> None:
    runs = {name: load(path) for name, path in RUNS.items()}
    tasks = sorted(set(runs["f22"]) & set(runs["f23"]))
    both_wrong = [
        t for t in tasks
        if not runs["f22"][t].get("exact_match") and not runs["f23"][t].get("exact_match")
    ]
    print(f"f22 錯 {sum(1 for t in tasks if not runs['f22'][t].get('exact_match'))}"
          f"、f23 錯 {sum(1 for t in tasks if not runs['f23'][t].get('exact_match'))}"
          f"、兩輪皆錯 {len(both_wrong)}/{len(tasks)}")

    rows = []
    for task in both_wrong:
        record = runs["f23"][task]
        gold = normalise(record.get("expected"))
        stage, reached = classify(record, gold)
        rows.append({
            "task": task,
            "gold": gold,
            "f22_pred": normalise(runs["f22"][task].get("predicted")),
            "f23_pred": normalise(record.get("predicted")),
            "stage": stage,
            "reached": reached,
        })

    print(f"\n=== 卡住的位置")
    order = ("never_searched", "not_in_documents", "stuck_at_documents",
             "stuck_at_references", "stuck_at_context", "stuck_at_runs")
    counts = Counter(r["stage"] for r in rows)
    for stage in order:
        if counts[stage]:
            ids = [r["task"] for r in rows if r["stage"] == stage]
            print(f"   {stage:<22} {counts[stage]:>3}　{', '.join(ids)}")

    downstream = counts["stuck_at_context"] + counts["stuck_at_runs"]
    print(f"\n   檢索端（未搜尋／文件裡沒有）      "
          f"{counts['never_searched'] + counts['not_in_documents']}")
    print(f"   遞送端（文件有、下游遺失）        "
          f"{counts['stuck_at_documents'] + counts['stuck_at_references']}")
    print(f"   選擇端（已到 agent 面前仍未產出）  {downstream}")

    print(f"\n=== 兩輪答案是否一致")
    identical = [r for r in rows if r["f22_pred"].casefold() == r["f23_pred"].casefold()]
    print(f"   兩輪給出相同錯答 {len(identical)}/{len(rows)}"
          f" —— 相同代表失敗是確定性的，不同代表是取樣雜訊")

    print(f"\n=== 逐題")
    for row in sorted(rows, key=lambda r: (order.index(r["stage"]), r["task"])):
        marks = "".join(
            k[0].upper() if v else "-" for k, v in row["reached"].items()
        )
        same = "＝" if row["f22_pred"].casefold() == row["f23_pred"].casefold() else "≠"
        print(f"   {row['task']}  {row['stage']:<20} [{marks}] {same}")
        print(f"        gold={row['gold'][:44]!r}")
        print(f"        f22={row['f22_pred'][:40]!r}  f23={row['f23_pred'][:40]!r}")
    print(f"\n   [D R S C U] = documents / references / strict / context / runs")


if __name__ == "__main__":
    sys.exit(main())
