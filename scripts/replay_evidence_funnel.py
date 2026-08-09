"""Trace where the answer stops surviving, on fixed recorded retrieval.

Live reruns cannot attribute a change: measured across level1_final_13, _14 and
_15 with the same seed, no task retrieved the same document set twice and the
median Jaccard overlap was 0.33. Anything compared across runs is therefore
confounded by two thirds of the corpus turning over.

This reads one recorded run and walks the answer through the stages it has to
survive, all of which are already in the trace except the last, which is
rebuilt with the real context budget:

    retrieved document -> useful span -> classified span -> semantic fact
    -> direct contract -> grounded evidence -> unverified reference
    -> the search block Stage 1 actually sees

Every task is assigned the stage where the answer was last seen, so the output
says which step to work on rather than that something upstream is wrong.

    .\\venv312\\Scripts\\python.exe scripts/replay_evidence_funnel.py \\
      --run outputs/level1_final_15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unicodedata import normalize as un

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context.context_budget import ContextBudgetManager
from context.context_builder import ContextBuilder
from core.evidence_runner import EvidenceRunner

RENDERER = EvidenceRunner.__new__(EvidenceRunner)
BUDGET = ContextBudgetManager()
COMPRESSOR = ContextBuilder()

STAGES = (
    "retrieved_document",
    "useful_span",
    "classified_span",
    "semantic_fact",
    "direct_contract",
    "grounded_evidence",
    "unverified_reference",
    "stage1_context",
)


def normalise(value: Any) -> str:
    without_urls = re.sub(r"https?://\S+", " ", str(value or ""))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", un("NFKC", without_urls).casefold()).split())


def carries(haystack: Any, gold: str) -> bool:
    """Whether the answer appears in the text, ignoring URLs.

    Short answers are matched on word boundaries: a gold of "3" otherwise
    matches any digit anywhere and reports survival that is not there.
    """

    needle = normalise(gold)
    if not needle:
        return False
    text = normalise(haystack)
    if len(needle) <= 4:
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None
    return needle in text


def _blob(items: Any) -> str:
    if isinstance(items, list):
        return "\n".join(json.dumps(item, ensure_ascii=False, default=str) if not isinstance(item, str) else item for item in items)
    return str(items or "")


class _Contract:
    def __init__(self, requirement: str, target: str) -> None:
        self.answer_requirement = requirement
        self.answer_target = target


def _contract(summary: dict) -> _Contract:
    refs = summary.get("unverified_references") or []
    return _Contract(
        next((str(r.get("answer_requirement") or "") for r in refs if r.get("answer_requirement")), ""),
        next((str(r.get("answer_target") or "") for r in refs if r.get("answer_target")), ""),
    )


def stage1_context(summary: dict) -> str:
    """The search block after the budget, which is what Stage 1 reads."""

    rendered = EvidenceRunner._render_web_retrieval_evidence(
        RENDERER,
        evidence_items=list(summary.get("evidence_items") or []),
        unverified_references=list(summary.get("unverified_references") or []),
        answer_candidates=list(summary.get("answer_candidates") or []),
        contract=_contract(summary),
    )
    compacted, _dropped = BUDGET._compact_search_evidence(
        COMPRESSOR._compress_multiline_text(rendered)
    )
    return compacted


def survival(task: dict) -> dict[str, bool] | None:
    gold = str(task.get("expected") or "")
    summary = task.get("search_summary") or {}
    rounds = summary.get("retrieval_rounds") or []
    if not rounds or not (3 <= len(gold) <= 60):
        return None

    documents = [doc for round_trace in rounds for doc in round_trace.get("documents") or []]
    seen = {
        "retrieved_document": any(carries(doc.get("text"), gold) for doc in documents),
        "useful_span": any(carries(span, gold) for doc in documents for span in doc.get("useful_spans") or []),
        "classified_span": any(
            carries(_blob([item]), gold) for doc in documents for item in doc.get("span_roles") or []
        ),
        "semantic_fact": any(
            carries(_blob([fact]), gold)
            for doc in documents
            for item in doc.get("span_roles") or []
            for fact in item.get("semantic_facts") or []
        )
        or any(carries(_blob(doc.get("semantic_facts")), gold) for doc in documents),
        "direct_contract": any(carries(_blob(doc.get("direct_contracts")), gold) for doc in documents),
        "grounded_evidence": any(carries(item.get("text"), gold) for item in summary.get("evidence_items") or []),
        "unverified_reference": any(
            carries(item.get("text"), gold) for item in summary.get("unverified_references") or []
        ),
        "stage1_context": carries(stage1_context(summary), gold),
    }
    return seen


def reference_position(task: dict) -> int:
    gold = str(task.get("expected") or "")
    refs = (task.get("search_summary") or {}).get("unverified_references") or []
    for index, item in enumerate(refs, start=1):
        if carries(item.get("text"), gold):
            return index
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="An outputs/<log_name> directory.")
    args = parser.parse_args(argv)

    rows: list[tuple[str, str, dict[str, bool], int, bool]] = []
    for path in sorted(glob.glob(os.path.join(args.run, "tasks", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            task = json.load(handle)
        seen = survival(task)
        if seen is None:
            continue
        rows.append(
            (
                os.path.basename(path)[:3],
                str(task.get("expected") or "")[:18],
                seen,
                reference_position(task),
                bool(task.get("exact_match")),
            )
        )

    header = "%-5s %-18s " % ("task", "gold") + " ".join("%-6s" % s[:6] for s in STAGES) + " %5s %5s"
    print(header % ("refN", "exact"))
    for key, gold, seen, position, exact in rows:
        marks = " ".join("%-6s" % ("o" if seen[stage] else ".") for stage in STAGES)
        print("%-5s %-18s %s %5s %5s" % (key, gold, marks, position or "-", "O" if exact else "X"))

    print("\n每個階段仍存活的題數（共 %d 題有檢索且 gold 可比對）" % len(rows))
    for stage in STAGES:
        alive = sum(1 for _k, _g, seen, _p, _e in rows if seen[stage])
        print("  %-22s %3d" % (stage, alive))

    print("\n答案最後出現在哪個階段（即下一步在哪裡消失）")
    death = Counter()
    for _key, _gold, seen, _pos, _exact in rows:
        last = ""
        for stage in STAGES:
            if seen[stage]:
                last = stage
        death[last or "(從未出現)"] += 1
    for stage, count in death.most_common():
        print("  %-22s %3d" % (stage, count))

    positions = [p for _k, _g, _s, p, _e in rows if p]
    if positions:
        positions.sort()
        print(
            "\ngold 所在的參考序位：%s   （預算通常只保留前 3-4 筆）"
            % ", ".join(str(p) for p in positions)
        )


if __name__ == "__main__":
    main()
