"""Offline replay of the span-candidate budget over a saved GAIA run.

`PassageEvidenceUnitBuilder.max_units` is a batch-global cap applied across every
document of a retrieval round, not a per-document one. On level1_final_06 the
average round carried 21.9 documents into a budget of 10 units, and 90 of 116
rounds saturated it exactly -- so 1918 of 2535 documents contributed no
candidate at all and the answer-bearing sentence never reached role
classification. `max_units_per_document` (6) never binds at that scale, which is
why raising it alone did not move end-to-end answer survival.

This script rebuilds each round's document set from a finished run and re-runs
the builder under different budgets, scoring what actually decides the outcome:
whether the task's gold answer survives into the selected units.

Only tasks whose gold answer is present in some retrieved document are scored --
elsewhere no budget can help, and including them would dilute the measurement.

A second, separable defect sits alongside the small budget: selection is a
global top-k, so a few documents monopolise it. Across 105 scored rounds only
5.9 of 22.5 documents contributed anything (26.1% document coverage) and 9
rounds spent all 10 slots on a single document. `round_robin` tests a
per-document floor instead of a larger budget -- every document contributes its
best unit first, then the remainder fills by rank -- which reaches full document
coverage at roughly 2x the units rather than 13x.

Arms:
  baseline      max_units=10 (current); reproduces the recorded selection
  per_document  max_units = max_units_per_document * document_count
  ceiling_60    max_units = min(60, per_document budget)
  round_robin   per-document floor of 1, then rank order, capped at 3*documents

Reported per arm:
  gold_tasks    tasks whose gold answer reaches the selected units
  units         total selected units, the cost the classifier pays downstream

Fidelity: the run recorded `embedding_model = bge-m3` and
`ranking_method = e5_semantic_similarity`, so the replay loads bge-m3 and ranks
exactly as production did. Baseline gold_tasks is asserted against the recorded
`candidate_units` as the reconstruction check.

Usage:
    python scripts/replay_span_candidate_budget.py [--run outputs/level1_final_06]
                                                   [--model bge-m3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.search_result_builder.embeddings.embedder import Embedder
from tools.search_result_builder.evidence.passage_evidence_unit_builder import (
    PassageEvidenceUnitBuilder,
)

PER_DOCUMENT_CAP = 6


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_id"] = path.name[:3]
        tasks.append(payload)
    return tasks


def round_documents(round_trace: dict[str, Any]) -> list[dict[str, Any]]:
    documents = []
    for trace in round_trace.get("documents") or []:
        text = str(trace.get("text") or "")
        if not text:
            continue
        documents.append(
            {
                "id": str(trace.get("document_id") or ""),
                "document_id": str(trace.get("document_id") or ""),
                "title": str(trace.get("title") or ""),
                "text": text,
                "record_type": str(trace.get("record_type") or "passage"),
            }
        )
    return documents


def gold_in_documents(task: dict[str, Any]) -> bool:
    gold = normalize(task.get("expected"))
    if not gold:
        return False
    for round_trace in task.get("search_summary", {}).get("retrieval_rounds") or []:
        for trace in round_trace.get("documents") or []:
            if gold in normalize(trace.get("text")) or gold in normalize(
                trace.get("title")
            ):
                return True
    return False


def recorded_gold_in_units(task: dict[str, Any]) -> bool:
    gold = normalize(task.get("expected"))
    for round_trace in task.get("search_summary", {}).get("retrieval_rounds") or []:
        for trace in round_trace.get("documents") or []:
            diagnostics = trace.get("labeler_diagnostics") or {}
            for unit in diagnostics.get("candidate_units") or []:
                if gold in normalize(unit):
                    return True
    return False


def budget_for(arm: str, document_count: int) -> int:
    if arm == "baseline":
        return 10
    scaled = PER_DOCUMENT_CAP * max(1, document_count)
    if arm.startswith("ceiling_"):
        return min(int(arm.split("_", 1)[1]), scaled)
    return scaled


def round_robin_units(
    *,
    question: str,
    documents: list[dict[str, Any]],
    embedder: Any,
) -> list[Any]:
    """Select with a per-document floor instead of a larger global budget.

    Ranking is reused from the builder, so this differs from the other arms only
    in how the ranked list is drained: one pass giving every document its best
    unit, then rank order for the remainder.
    """

    builder = PassageEvidenceUnitBuilder(
        max_units=PER_DOCUMENT_CAP * max(1, len(documents)),
        max_units_per_document=PER_DOCUMENT_CAP,
    )
    ranked = builder.build(
        question=question,
        documents=documents,
        embedder=embedder,
    ).units

    budget = 3 * max(1, len(documents))
    selected: list[Any] = []
    claimed: set[int] = set()
    for unit in ranked:
        if unit.document_index in claimed:
            continue
        claimed.add(unit.document_index)
        selected.append(unit)
    for unit in ranked:
        if len(selected) >= budget:
            break
        if unit not in selected:
            selected.append(unit)
    return selected[:budget]


def score_arm(
    arm: str,
    tasks: list[dict[str, Any]],
    embedder: Any,
) -> dict[str, Any]:
    gold_tasks: list[str] = []
    total_units = 0
    for task in tasks:
        gold = normalize(task.get("expected"))
        question = str(task.get("question") or "")
        found = False
        for round_trace in task.get("search_summary", {}).get("retrieval_rounds") or []:
            documents = round_documents(round_trace)
            if not documents:
                continue
            if arm == "round_robin":
                units = round_robin_units(
                    question=question,
                    documents=documents,
                    embedder=embedder,
                )
            else:
                builder = PassageEvidenceUnitBuilder(
                    max_units=budget_for(arm, len(documents)),
                    max_units_per_document=PER_DOCUMENT_CAP,
                )
                units = builder.build(
                    question=question,
                    documents=documents,
                    embedder=embedder,
                ).units
            total_units += len(units)
            if any(gold in normalize(unit.text) for unit in units):
                found = True
        if found:
            gold_tasks.append(task["_id"])
    return {"arm": arm, "gold_tasks": gold_tasks, "units": total_units}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument(
        "--arms",
        default="baseline,per_document,ceiling_60",
        help="comma-separated arm names",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    tasks = [task for task in load_tasks(run_dir) if gold_in_documents(task)]
    if not tasks:
        print(f"no gold-bearing web tasks under {run_dir}")
        return 1

    recorded = [task["_id"] for task in tasks if recorded_gold_in_units(task)]
    print(f"run: {run_dir}")
    print(f"gold-bearing web tasks: {len(tasks)}  {[t['_id'] for t in tasks]}")
    print(f"recorded gold-in-units : {len(recorded)}  {recorded}")
    print()

    embedder = Embedder(args.model, batch_size=64, text_normalize=True)

    results = []
    for arm in args.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue
        outcome = score_arm(arm, tasks, embedder)
        results.append(outcome)
        print(
            f"{arm:<14} gold_tasks={len(outcome['gold_tasks']):>2}/{len(tasks)}  "
            f"units={outcome['units']:>5}  {outcome['gold_tasks']}"
        )

    baseline = next((r for r in results if r["arm"] == "baseline"), None)
    if baseline is not None:
        print()
        if sorted(baseline["gold_tasks"]) == sorted(recorded):
            print("fidelity: baseline arm reproduces the recorded selection")
        else:
            print(
                "fidelity WARNING: baseline "
                f"{sorted(baseline['gold_tasks'])} != recorded {sorted(recorded)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
