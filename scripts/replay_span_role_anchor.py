"""Replay semantic fact extraction over a saved run, scored against gold answers.

SemanticFactExtractor treats a unit's `requested_role` as a hard override of the
model's own role output (see `_normalize_and_ground`), and RetrievalControl
forwards the SpanRoleClassifier label into it. On level1_final_06 that
classifier emitted BRIDGE 337 times against ANSWER_SUPPORT 8, which pinned 317
of 325 facts to BRIDGE regardless of what the extractor actually said.

This script rebuilds the extractor's input units from a finished run (each saved
fact keeps its `source_unit_id`, the unit `context` it was read from, and its
`evidence_refs` candidate span) and runs them with the anchor on and off. It
scores the arms on what decides the outcome rather than on fact volume:

  answer_hit    a fact whose object contains the task's gold answer
  direct_hit    an answer_hit that is also ANSWER_SUPPORT + answer_binding=direct
                and grounded — the only shape that reaches a direct contract,
                and therefore evidence authority
  bridge_count  retained BRIDGE facts, the material next-hop queries build on;
                this is what dropping the anchor costs

Measured on level1_final_06 lookup tasks: dropping the anchor moved
ANSWER_SUPPORT from 0 to 10 but direct_hit stayed at 0 in both arms, while
BRIDGE halved. The gold answer reaches the extractor on only 3 of 20 tasks,
which is why role-level fixes cannot move the result.

Usage:
    python scripts/replay_span_role_anchor.py [--run outputs/level1_final_06]
                                              [--question-type lookup]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evidence.fact_extraction import AggregationFactDeriver
from tools.evidence.fact_extraction.models import SemanticSourceUnit
from tools.evidence.fact_extraction.semantic_fact_extractor import SemanticFactExtractor


def load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            tasks.append(json.load(handle))
    return tasks


def rebuild_units(
    task: dict[str, Any],
    *,
    requested_role: str,
) -> list[SemanticSourceUnit]:
    """One unit per distinct source_unit_id seen in the saved fact store."""

    units: dict[str, SemanticSourceUnit] = {}
    rounds = task.get("search_summary", {}).get("retrieval_rounds", []) or []
    for round_trace in rounds:
        store = (round_trace.get("filter_metadata") or {}).get(
            "semantic_fact_store"
        ) or {}
        for fact in store.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            qualifiers = fact.get("qualifiers") or {}
            unit_id = str(qualifiers.get("source_unit_id") or "").strip()
            text = str(fact.get("context") or "").strip()
            if not unit_id or not text or unit_id in units:
                continue
            refs = [
                item for item in fact.get("evidence_refs") or [] if isinstance(item, dict)
            ]
            ref = refs[0] if refs else {}
            units[unit_id] = SemanticSourceUnit(
                unit_id=unit_id,
                text=text,
                source_id=str(ref.get("source_id") or fact.get("source_id") or unit_id),
                source_type=str(fact.get("source_type") or "web"),
                source_title=str(fact.get("source_title") or ""),
                candidate_span=str(ref.get("text") or ""),
                requested_role=requested_role,
                goal_id=str(fact.get("goal_id") or ""),
            )
    return list(units.values())


def answer_requirement(task: dict[str, Any]) -> str:
    rounds = task.get("search_summary", {}).get("retrieval_rounds", []) or []
    for round_trace in rounds:
        store = (round_trace.get("filter_metadata") or {}).get(
            "semantic_fact_store"
        ) or {}
        for fact in store.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            requirement = (fact.get("qualifiers") or {}).get("answer_requirement")
            if requirement:
                return " ".join(str(requirement).split())
    return ""


def select_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    question_type: str,
) -> list[dict[str, Any]]:
    """Keep replayable tasks, optionally narrowed to one question type.

    Aggregate/lookup is decided by AggregationFactDeriver's operator inference,
    the same rule the aggregation path uses.
    """

    deriver = AggregationFactDeriver()
    selected = []
    for task in tasks:
        question = " ".join(str(task.get("question") or "").split())
        if not question:
            continue
        if question_type != "all":
            operator = deriver.infer_operator(question)
            resolved = "aggregate" if operator else "lookup"
            if resolved != question_type:
                continue
        if rebuild_units(task, requested_role=""):
            selected.append(task)
    return selected


def _norm(value: str) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "")).casefold().strip()


def _is_answer_hit(fact_object: str, expected: str) -> bool:
    gold = _norm(expected)
    return bool(gold) and gold in _norm(fact_object)


def run_arm(tasks: list[dict[str, Any]], *, anchored: bool) -> dict[str, Any]:
    extractor = SemanticFactExtractor()
    roles: Counter[str] = Counter()
    fact_total = 0
    answer_hits = 0
    direct_hits = 0
    tasks_with_direct_hit: set[str] = set()

    for task in tasks:
        task_id = str(task.get("task_id") or "")
        expected = str(task.get("expected") or "")
        question = " ".join(str(task.get("question") or "").split())
        requirement = answer_requirement(task) or question
        units = rebuild_units(task, requested_role="BRIDGE" if anchored else "")
        if not units:
            continue
        size = max(1, extractor.max_units_per_call)
        for start in range(0, len(units), size):
            result = extractor.extract_batch(
                question=question,
                answer_requirement=requirement,
                current_goal="",
                units=units[start : start + size],
                keep_alive="2m",
            )
            for fact in result.facts:
                fact_total += 1
                roles[fact.role] += 1
                if not _is_answer_hit(fact.object, expected):
                    continue
                answer_hits += 1
                if (
                    fact.role == "ANSWER_SUPPORT"
                    and fact.qualifiers.get("answer_binding") == "direct"
                    and fact.grounding_status == "grounded"
                ):
                    direct_hits += 1
                    tasks_with_direct_hit.add(task_id)

    return {
        "fact_count": fact_total,
        "roles": dict(roles),
        "bridge_count": roles.get("BRIDGE", 0),
        "answer_hits": answer_hits,
        "direct_hits": direct_hits,
        "tasks_with_direct_hit": sorted(tasks_with_direct_hit),
    }


def report_answer_reachability(tasks: list[dict[str, Any]]) -> None:
    """Where the gold answer survives: retrieved documents vs extraction units.

    No model calls — this is the measurement that showed span selection, not
    role labelling, is what loses the answer.
    """

    in_docs = in_units = counted = 0
    for task in tasks:
        gold = _norm(task.get("expected"))
        if not gold:
            continue
        counted += 1
        documents = " ".join(
            _norm(document.get("text"))
            for round_trace in task.get("search_summary", {}).get("retrieval_rounds", [])
            for document in round_trace.get("documents", []) or []
        )
        units = " ".join(
            _norm(unit.text) for unit in rebuild_units(task, requested_role="")
        )
        in_docs += gold in documents
        in_units += gold in units
    print("\n=== gold answer reachability (no model calls) ===")
    print(f"  present in retrieved documents : {in_docs}/{counted}")
    print(f"  present in extraction units    : {in_units}/{counted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument(
        "--question-type", default="lookup", choices=["lookup", "aggregate", "all"]
    )
    parser.add_argument(
        "--reachability-only",
        action="store_true",
        help="Skip the model arms and only report where the gold answer survives.",
    )
    args = parser.parse_args()

    tasks = select_tasks(
        load_tasks(ROOT / args.run), question_type=args.question_type
    )
    print(f"run={args.run} question_type={args.question_type} tasks={len(tasks)}")

    if args.reachability_only:
        report_answer_reachability(tasks)
        return

    results = {}
    for name, anchored in (("anchored (final_06)", True), ("unanchored", False)):
        print(f"\n--- arm: {name} ---", flush=True)
        summary = run_arm(tasks, anchored=anchored)
        results[name] = summary
        print(f"    facts={summary['fact_count']} roles={summary['roles']}")
        print(
            f"    answer_hits={summary['answer_hits']} "
            f"direct_hits={summary['direct_hits']} bridge={summary['bridge_count']}"
        )

    print("\n=== trade-off ===")
    print(f"{'arm':<22}{'facts':>7}{'BRIDGE':>8}{'answer_hit':>12}{'direct_hit':>12}")
    for name, summary in results.items():
        print(
            f"{name:<22}{summary['fact_count']:>7}{summary['bridge_count']:>8}"
            f"{summary['answer_hits']:>12}{summary['direct_hits']:>12}"
        )

    report_answer_reachability(tasks)

    out = ROOT / args.run / "replay_span_role_anchor.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
