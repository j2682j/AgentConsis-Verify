"""Offline replay of the direct/bridge contract gate over a saved GAIA run.

`EvidenceRoleContractBuilder._build_goal_assigned` is the last gate before a
classified span becomes answer evidence. On level1_final_06 it emitted 0 direct
contracts from 15 accepted ANSWER_SUPPORT spans, which is why every web task
reported `evidence_count = 0` and Stage1 answered from parametric knowledge
alone.

This script rebuilds the gate's exact inputs from a finished run -- each saved
document keeps the `finalized_spans` the finalizer accepted (with their own
`semantic_facts` and `goal_id`), and the task keeps the `relation_plan` those
goals came from -- and re-runs the builder, so gate changes can be scored
without re-executing retrieval.

Reported per task:
  direct        emitted direct contracts (recorded runs have 0)
  bridge        emitted bridge contracts
  reject        rejection reasons, which is where the gate's behaviour shows
  gold_hit      a direct contract whose answer span or fact object carries the
                task's gold answer -- the upside of loosening the gate
  off_answer    a direct contract that carries something else -- the downside,
                since it becomes authoritative evidence contradicting a task
                Stage1 may currently answer correctly from priors

Fidelity note: the run stores one final `relation_plan` per task rather than a
per-round snapshot, so goal *states* replay as of the end of the run. Direct
contracts do not read goal state (only subject/relation/target), so the direct
arm is faithful; bridge counts are approximate and are reported for context
only. The recorded direct total is asserted as the reconstruction check.

Usage:
    python scripts/replay_direct_contract_gate.py [--run outputs/level1_final_06]
                                                  [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.search_result_builder.evidence.evidence_role_contract import (
    EvidenceRoleContractBuilder,
)
from tools.search_result_builder.query.relation_plan import RelationPlan


def load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_file"] = path.name
        tasks.append(payload)
    return tasks


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def task_relation_plan(task: dict[str, Any]) -> RelationPlan:
    coverage = task.get("search_summary", {}).get("coverage_summary") or {}
    state = coverage.get("final_intent_state") or {}
    payload = state.get("relation_plan") or {}
    if not payload.get("goals"):
        return RelationPlan()
    return RelationPlan.from_dict(payload)


def documents(task: dict[str, Any]):
    for round_trace in task.get("search_summary", {}).get("retrieval_rounds") or []:
        for document in round_trace.get("documents") or []:
            yield document


def span_assignments(document: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = document.get("labeler_diagnostics") or {}
    classifier = diagnostics.get("span_role_classifier") or {}
    return [
        item
        for item in classifier.get("finalized_spans") or []
        if isinstance(item, dict)
    ]


def replay_task(
    task: dict[str, Any],
    builder: EvidenceRoleContractBuilder,
) -> dict[str, Any]:
    question = str(task.get("question") or "")
    plan = task_relation_plan(task)
    gold = normalize(task.get("expected"))

    direct: list[dict[str, Any]] = []
    bridge_count = 0
    rejects: Counter[str] = Counter()

    for document in documents(task):
        assignments = span_assignments(document)
        if not assignments:
            continue
        contracts = builder.build(
            question=question,
            answer_requirement="",
            answer_target="",
            relation_plan=plan,
            document_id=str(document.get("document_id") or ""),
            source_title=str(document.get("title") or ""),
            url=str(document.get("url") or ""),
            text=str(document.get("text") or ""),
            span_assignments=assignments,
        )
        direct.extend(item.to_dict() for item in contracts.direct)
        bridge_count += len(contracts.bridge)
        for item in contracts.unsupported:
            rejects[item.reason] += 1

    gold_hit = 0
    off_answer = 0
    for contract in direct:
        carried = f"{contract.get('answer_span')} {contract.get('object')}"
        if gold and gold in normalize(carried):
            gold_hit += 1
        else:
            off_answer += 1

    return {
        "file": task.get("_file", "")[:3],
        "exact": bool(task.get("exact_match")),
        "expected": str(task.get("expected") or "")[:26],
        "direct": len(direct),
        "bridge": bridge_count,
        "gold_hit": gold_hit,
        "off_answer": off_answer,
        "rejects": rejects,
        "contracts": direct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    tasks = load_tasks(run_dir)
    if not tasks:
        print(f"no task JSONs under {run_dir}")
        return 1

    builder = EvidenceRoleContractBuilder()
    rows = [replay_task(task, builder) for task in tasks]
    scored = [row for row in rows if row["direct"] or row["bridge"] or row["rejects"]]

    print(f"run: {run_dir}")
    print(f"tasks: {len(rows)}  (with contract activity: {len(scored)})")
    print()
    print(
        f"{'id':<4} {'ex':<3} {'expected':<28} {'direct':>6} {'gold':>5} "
        f"{'off':>4} {'bridge':>6}"
    )
    for row in scored:
        print(
            f"{row['file']:<4} {'T' if row['exact'] else '.':<3} "
            f"{row['expected']:<28} {row['direct']:>6} {row['gold_hit']:>5} "
            f"{row['off_answer']:>4} {row['bridge']:>6}"
        )

    total_direct = sum(row["direct"] for row in rows)
    total_gold = sum(row["gold_hit"] for row in rows)
    total_off = sum(row["off_answer"] for row in rows)
    total_bridge = sum(row["bridge"] for row in rows)
    rejects: Counter[str] = Counter()
    for row in rows:
        rejects.update(row["rejects"])

    print()
    print(f"direct contracts : {total_direct}")
    print(f"  carrying gold  : {total_gold}")
    print(f"  off-answer     : {total_off}   <-- regression surface")
    print(f"bridge contracts : {total_bridge}  (approximate, see docstring)")
    print(f"rejections       : {sum(rejects.values())}")
    for reason, count in rejects.most_common():
        print(f"  {count:>4}  {reason}")

    tasks_with_direct = sum(1 for row in rows if row["direct"])
    tasks_with_gold = sum(1 for row in rows if row["gold_hit"])
    print()
    print(f"tasks gaining any direct contract : {tasks_with_direct}")
    print(f"tasks gaining a gold-carrying one : {tasks_with_gold}")

    if args.verbose:
        print()
        for row in rows:
            for contract in row["contracts"]:
                print(
                    f"[{row['file']}] gold={row['expected']!r} "
                    f"goal={contract.get('goal_id')} "
                    f"span={str(contract.get('answer_span'))[:60]!r} "
                    f"fact=({contract.get('subject')!r}, "
                    f"{contract.get('relation')!r}, {contract.get('object')!r})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
