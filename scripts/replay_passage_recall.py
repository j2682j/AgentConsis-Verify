"""Measure answer-passage recall of the Stage1 context over saved GAIA runs.

Re-runs evidence conversion on each task's recorded retrieval trace and asks
one question: does the text actually handed to Stage1 (grounded evidence plus
read-only references) contain the expected answer?

That number is the ceiling on what Stage1 can possibly get right from
retrieval, so it isolates passage selection from agent reasoning.

Usage:
    python scripts/replay_passage_recall.py outputs/level1_full_system_final_2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_result_builder.evidence.evidence_contract import (
    EvidenceSelectionContract,
)
from tools.search_result_builder.evidence.evidence_converter import EvidenceConverter


def search_raw(metadata: dict) -> dict:
    for usage in metadata.get("tool_usage") or []:
        if not isinstance(usage, dict) or usage.get("tool_name") != "search":
            continue
        raw = usage.get("raw_result")
        if isinstance(raw, dict) and isinstance(raw.get("retrieval"), dict):
            return raw
    return {}


def contract_for(raw: dict, question: str) -> EvidenceSelectionContract:
    diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
    stored = diagnostics.get("evidence_selection_contract")
    stored = stored if isinstance(stored, dict) else {}
    return EvidenceSelectionContract.from_parts(
        question=question,
        answer_requirement=str(stored.get("answer_requirement") or ""),
        answer_target=str(stored.get("answer_target") or ""),
        must_include=list(stored.get("must_include") or []),
    )


def answer_in_corpus(raw: dict, expected: str) -> int:
    hits = 0
    retrieval = raw.get("retrieval") or {}
    for round_info in retrieval.get("rounds") or []:
        for document in round_info.get("documents") or []:
            if isinstance(document, dict) and expected.casefold() in str(
                document.get("text") or ""
            ).casefold():
                hits += 1
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=650)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "tasks", "*.json")))
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        metadata = (data.get("network_summary") or {}).get("metadata") or {}
        raw = search_raw(metadata)
        if not raw:
            continue
        expected = str(data.get("expected") or "").strip()
        if len(expected) < 3:
            continue
        question = str(data.get("question") or "")

        converter = EvidenceConverter(
            max_items=args.max_items, max_chars=args.max_chars
        )
        items = converter.convert_web_retrieval_output(
            {"retrieval": raw.get("retrieval") or {}},
            contract=contract_for(raw, question),
        )
        shown = " ".join(
            [str(item.get("text") or "") for item in items]
            + [
                str(reference.get("text") or "")
                for reference in converter.last_relaxed_references
            ]
        )
        rows.append(
            {
                "task": os.path.basename(path)[:3],
                "exact": bool(data.get("exact_match")),
                "expected": expected[:34],
                "in_corpus": answer_in_corpus(raw, expected),
                "in_shown": expected.casefold() in shown.casefold(),
                "grounded": len(items),
                "references": len(converter.last_relaxed_references),
            }
        )

    recoverable = [r for r in rows if r["in_corpus"] > 0]
    shown_ok = [r for r in rows if r["in_shown"]]
    print(f"search tasks with a checkable answer: {len(rows)}")
    print(f"answer present somewhere in retrieved corpus: {len(recoverable)}")
    print(
        f"answer present in text shown to Stage1:      {len(shown_ok)}"
        f"   (recall over recoverable: {len(shown_ok)}/{len(recoverable)})"
    )
    print()
    print("task | exact | inCorpus | inShown | grounded | refs | expected")
    for r in rows:
        flag = "**" if r["in_corpus"] > 0 and not r["in_shown"] else "  "
        print(
            f"{flag}{r['task']} | {str(r['exact']):5} | {r['in_corpus']:8} |"
            f" {str(r['in_shown']):7} | {r['grounded']:8} | {r['references']:4} | {r['expected']}"
        )
    print()
    missed = [r for r in rows if r["in_corpus"] > 0 and not r["in_shown"]]
    if missed:
        print("** = answer is in the corpus but was NOT shown to Stage1 (recoverable by better passage selection):")
        print("   " + ", ".join(r["task"] for r in missed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
