"""Offline replay of search evidence conversion over saved GAIA task JSONs.

Rebuilds each task's recorded retrieval rounds from
``network_summary.metadata.tool_usage[search].raw_result.retrieval`` and
re-runs EvidenceConverter, so conversion changes can be validated against a
finished benchmark run without re-executing retrieval.

Reported per task:
- strict item count (must match the recorded strict_evidence_count, since
  the strict path is unchanged — this validates the reconstruction)
- relaxed item count and bucket mix
- whether the expected answer string appears in the emitted evidence text
  (the ceiling on what better evidence can give Stage1)

Usage:
    python scripts/replay_evidence_conversion.py outputs/level1_40_system_final
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


def search_raw_result(metadata: dict) -> dict:
    for usage in metadata.get("tool_usage") or []:
        if not isinstance(usage, dict) or usage.get("tool_name") != "search":
            continue
        raw = usage.get("raw_result")
        if isinstance(raw, dict) and isinstance(raw.get("retrieval"), dict):
            return raw
    return {}


def recorded_contract(raw: dict, question: str) -> EvidenceSelectionContract:
    diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
    stored = diagnostics.get("evidence_selection_contract")
    stored = stored if isinstance(stored, dict) else {}
    return EvidenceSelectionContract.from_parts(
        question=question,
        answer_requirement=str(stored.get("answer_requirement") or ""),
        answer_target=str(stored.get("answer_target") or ""),
        must_include=list(stored.get("must_include") or []),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="GAIA report directory containing tasks/*.json")
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=650)
    args = parser.parse_args()

    task_files = sorted(glob.glob(os.path.join(args.output_dir, "tasks", "*.json")))
    if not task_files:
        print(f"no task JSONs under {args.output_dir}")
        return 1

    fidelity_breaks = []
    rows = []
    for path in task_files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        metadata = (data.get("network_summary") or {}).get("metadata") or {}
        raw = search_raw_result(metadata)
        if not raw:
            continue
        question = str(data.get("question") or "")
        expected = str(data.get("expected") or "").strip()
        recorded_strict = int(
            (data.get("search_summary") or {}).get("strict_evidence_count") or 0
        )

        converter = EvidenceConverter(
            max_items=args.max_items, max_chars=args.max_chars
        )
        items = converter.convert_web_retrieval_output(
            {"retrieval": raw.get("retrieval") or {}},
            contract=recorded_contract(raw, question),
        )
        strict_items = [item for item in items if not item.get("relaxed")]
        relaxed_items = [item for item in items if item.get("relaxed")]
        if len(strict_items) != recorded_strict:
            fidelity_breaks.append(
                (os.path.basename(path), recorded_strict, len(strict_items))
            )
        evidence_text = " ".join(str(item.get("text") or "") for item in items)
        answer_in_evidence = bool(
            expected and len(expected) >= 2
            and expected.casefold() in evidence_text.casefold()
        )
        diag = converter.last_diagnostics
        rows.append(
            {
                "task": os.path.basename(path)[:3],
                "exact": bool(data.get("exact_match")),
                "expected": expected[:40],
                "strict": len(strict_items),
                "relaxed": len(relaxed_items),
                "buckets": dict(diag.relaxed_bucket_counts),
                "answer_in_evidence": answer_in_evidence,
            }
        )

    if fidelity_breaks:
        print(f"FIDELITY: {len(fidelity_breaks)} task(s) where replayed strict count != recorded")
        for name, recorded, replayed in fidelity_breaks:
            print(f"  {name}: recorded={recorded} replayed={replayed}")
    else:
        print(
            f"FIDELITY: replayed strict evidence matches recorded counts on all {len(rows)} search tasks"
        )
    print()
    with_evidence = sum(1 for r in rows if r["strict"] + r["relaxed"] > 0)
    with_answer = sum(1 for r in rows if r["answer_in_evidence"])
    print(f"search tasks: {len(rows)}")
    print(f"tasks with non-empty evidence: {with_evidence} (was: 0 strict everywhere)")
    print(f"tasks whose emitted evidence contains the expected answer: {with_answer}")
    print()
    print("task | exact | strict | relaxed | answer_in_evidence | buckets | expected")
    for r in rows:
        print(
            f" {r['task']}  | {str(r['exact']):5} | {r['strict']:6} | {r['relaxed']:7} |"
            f" {str(r['answer_in_evidence']):5} | {r['buckets']} | {r['expected']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
