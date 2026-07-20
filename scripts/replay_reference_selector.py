from __future__ import annotations

"""Replay BestEffortReferenceSelector over recorded task JSONs.

The selector is a pure function of the recorded retrieval trace, so its
behavior before/after a code change can be compared offline without
re-running any experiment. Usage:

    python scripts/replay_reference_selector.py <output.json>

Writes one entry per task: run, task number, exact_match, and the selected
references (title/text/url/source_type). Diff two capture files to verify a
selector change only removes junk rows or appends content for tasks that
were previously answered correctly.
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_result_builder.evidence.best_effort_reference import (
    BestEffortReferenceSelector,
)

RUNS = {
    "L1_30": "outputs/level1_30_system_final/tasks/*.json",
    "L1_40": "outputs/level1_40_system_final/tasks/*.json",
    "L1_full": "outputs/level1_full_system_final/tasks/*.json",
    "L2_full": "outputs/level2_full_system_final/tasks/*.json",
}


def replay(out_path: str) -> list[dict[str, Any]]:
    selector = BestEffortReferenceSelector()
    entries = []
    for run_name, pattern in RUNS.items():
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            meta = (d.get("network_summary") or {}).get("metadata") or {}
            raw = None
            for usage in meta.get("tool_usage") or []:
                if not isinstance(usage, dict):
                    continue
                candidate = usage.get("raw_result")
                if isinstance(candidate, dict) and isinstance(
                    candidate.get("retrieval"), dict
                ):
                    raw = candidate
                    break
            if raw is None:
                continue
            output = {
                "retrieval": raw.get("retrieval"),
                "question": str(d.get("question") or ""),
            }
            strict = [
                item
                for item in list(raw.get("evidence_items") or [])
                if isinstance(item, dict)
            ]
            references = selector.select(
                output,
                strict_evidence_items=strict,
            )
            entries.append(
                {
                    "run": run_name,
                    "task": os.path.basename(path)[:3],
                    "exact_match": bool(d.get("exact_match")),
                    "strict_evidence_count": len(strict),
                    "references": [
                        {
                            "title": item.title,
                            "text": item.text,
                            "url": item.url,
                            "source_type": item.source_type,
                        }
                        for item in references
                    ],
                }
            )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"captured {len(entries)} tasks -> {out_path}")
    return entries


def _reference_key(reference: dict[str, Any]) -> tuple[str, str]:
    url = str(reference.get("url") or "").strip().casefold()
    title = str(reference.get("title") or "").strip().casefold()
    # A merged collection title contains its row count and may legitimately
    # change when sibling record types are unified. Its parent URL is stable.
    return ("url", url) if url else ("title", title)


def _meaningful_lines(reference: dict[str, Any]) -> list[str]:
    lines = []
    text = str(reference.get("text") or "")
    raw_lines = []
    for physical_line in text.splitlines():
        raw_lines.extend(re.split(r"\s+-\s+(?=Record Type:)", physical_line))
    for raw_line in raw_lines:
        line = raw_line.strip().removeprefix("-").strip()
        if not line:
            continue
        if BestEffortReferenceSelector._is_collection_placeholder(line):
            continue
        lines.append(" ".join(line.split()))
    return lines


def _is_subsequence(older: list[str], newer: list[str]) -> bool:
    if not older:
        return True
    position = 0
    for line in newer:
        if line == older[position]:
            position += 1
            if position == len(older):
                return True
    return False


def _text_is_tail_extended(old_text: str, new_text: str) -> bool:
    old_text = "\n".join(line.rstrip() for line in old_text.splitlines()).strip()
    new_text = "\n".join(line.rstrip() for line in new_text.splitlines()).strip()
    if old_text == new_text:
        return True
    if old_text.endswith(" ..."):
        return new_text.startswith(old_text[:-4].rstrip())
    return new_text.startswith(old_text)


def _change_is_additive(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> bool:
    """Accept junk removal and tail additions without losing/reordering content."""

    next_index = 0
    for old_reference in before:
        old_lines = _meaningful_lines(old_reference)
        if not old_lines:
            continue
        old_key = _reference_key(old_reference)
        match_index = next(
            (
                index
                for index in range(next_index, len(after))
                if _reference_key(after[index]) == old_key
            ),
            None,
        )
        if match_index is None:
            return False
        new_reference = after[match_index]
        if not (
            _text_is_tail_extended(
                str(old_reference.get("text") or ""),
                str(new_reference.get("text") or ""),
            )
            or _is_subsequence(old_lines, _meaningful_lines(new_reference))
        ):
            return False
        next_index = match_index + 1
    return True


def compare(
    baseline_path: str,
    current: list[dict[str, Any]],
    diff_path: str,
) -> dict[str, Any]:
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    old_by_key = {(item["run"], item["task"]): item for item in baseline}
    new_by_key = {(item["run"], item["task"]): item for item in current}
    changes = []
    unsafe_exact_changes = []
    for key in sorted(old_by_key.keys() & new_by_key.keys()):
        before = list(old_by_key[key].get("references") or [])
        after = list(new_by_key[key].get("references") or [])
        if before == after:
            continue
        additive = _change_is_additive(before, after)
        change = {
            "run": key[0],
            "task": key[1],
            "exact_match": bool(old_by_key[key].get("exact_match")),
            "additive_or_junk_only": additive,
            "before": before,
            "after": after,
        }
        changes.append(change)
        if change["exact_match"] and not additive:
            unsafe_exact_changes.append(change)
    report = {
        "baseline_tasks": len(baseline),
        "current_tasks": len(current),
        "comparable_tasks": len(old_by_key.keys() & new_by_key.keys()),
        "changed_tasks": len(changes),
        "changed_exact_tasks": sum(item["exact_match"] for item in changes),
        "unsafe_exact_changes": len(unsafe_exact_changes),
        "changes": changes,
    }
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(
        "comparison: "
        f"changed={len(changes)}, "
        f"changed_exact={report['changed_exact_tasks']}, "
        f"unsafe_exact={len(unsafe_exact_changes)} -> {diff_path}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and compare best-effort reference selection."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="reference_replay.json",
        help="Path for the current replay snapshot.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional baseline snapshot produced before the selector change.",
    )
    parser.add_argument(
        "--diff-output",
        default="reference_replay_diff.json",
        help="Path for the structured comparison report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay_entries = replay(args.output)
    if args.compare_to:
        report = compare(args.compare_to, replay_entries, args.diff_output)
        if report["unsafe_exact_changes"]:
            raise SystemExit(2)
