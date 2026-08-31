"""Do the tasks whose spans were cut mid-phrase actually fail more often?

The entire boundary line rests on an assumption nobody has measured: a span cut
mid-phrase makes a worse query, a worse query retrieves worse, and the task is
answered wrong. Fifteen of final_22's failures were classed as retrieval, and 38
of 133 annotated spans were fragmented. Those two facts have never been put
against each other, so the work has been prioritised on a plausible story rather
than on evidence.

This is the cheapest possible check and should have come first. It reads run
records already on disk, calls no model, re-runs no benchmark, and touches
nothing frozen. If fragmentation does not separate the tasks that fail from the
tasks that succeed, the boundary selector is a component-level result about a
component that does not decide much, and the eight agent-side failures are worth
more than anything downstream of here.

The sample is 29 tasks. That is small enough that only a large effect could show
up, and an absence of signal here is weak evidence of absence -- worth saying
before anyone reads a null result as a verdict.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import sys
from math import comb

sys.path.insert(0, r"c:/SCP")

RUN = "c:/SCP/outputs/level1_final_22/tasks"
ANNOTATIONS = "c:/SCP/outputs/query_span_analysis/query_span_annotation_merged.csv"


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact test, because the cells are far too small for a chi-square."""

    n1, n2, ok, total = a + b, c + d, a + c, a + b + c + d
    if not n1 or not n2 or not ok or ok == total:
        return 1.0

    def probability(x: int) -> float:
        if x < 0 or x > n1 or ok - x < 0 or ok - x > n2:
            return 0.0
        return comb(n1, x) * comb(n2, ok - x) / comb(total, ok)

    observed = probability(a)
    return min(1.0, sum(
        probability(x) for x in range(n1 + 1)
        if probability(x) <= observed * (1 + 1e-9)
    ))


def main() -> None:
    rows = list(csv.DictReader(open(ANNOTATIONS, encoding="utf-8")))
    fragmented = {r["task_id"] for r in rows if r["human_boundary"] == "fragmented"}
    annotated = {r["task_id"] for r in rows}
    per_task = collections.Counter(
        r["task_id"] for r in rows if r["human_boundary"] == "fragmented"
    )

    results: dict[str, bool] = {}
    for path in sorted(glob.glob(f"{RUN}/*.json")):
        record = json.load(open(path, encoding="utf-8"))
        ordinal = os.path.basename(path)[:3]
        results[ordinal] = bool(record.get("exact_match"))

    print(f"final_22: {sum(results.values())}/{len(results)} 正確")
    print(f"標註涵蓋 {len(annotated)} task，其中 {len(fragmented)} 個含 fragmented span\n")

    covered = {t: ok for t, ok in results.items() if t in annotated}
    a = sum(1 for t, ok in covered.items() if t in fragmented and ok)
    b = sum(1 for t, ok in covered.items() if t in fragmented and not ok)
    c = sum(1 for t, ok in covered.items() if t not in fragmented and ok)
    d = sum(1 for t, ok in covered.items() if t not in fragmented and not ok)

    print(f"{'':18}{'答對':>6}{'答錯':>6}{'正確率':>10}")
    print(f"{'有 fragmented':18}{a:>6}{b:>6}{a / max(a + b, 1):>10.3f}")
    print(f"{'無 fragmented':18}{c:>6}{d:>6}{c / max(c + d, 1):>10.3f}")
    print(f"\n樣本 {a + b} vs {c + d} task"
          f"、Fisher exact two-sided p = {fisher_two_sided(a, b, c, d):.3f}")

    # A task can carry several fragmented spans. If fragmentation matters, more
    # of it should hurt more; a flat line across counts is the same null result
    # arriving by a second route.
    print(f"\n每個 task 的 fragmented span 數 vs 正確率")
    buckets: dict[int, list[bool]] = collections.defaultdict(list)
    for task, ok in covered.items():
        buckets[per_task.get(task, 0)].append(ok)
    for count in sorted(buckets):
        flags = buckets[count]
        print(f"   {count} 筆: {sum(flags)}/{len(flags)} = {sum(flags) / len(flags):.3f}")

    wrong_fragmented = sorted(t for t, ok in covered.items() if t in fragmented and not ok)
    print(f"\n有 fragmented span 且答錯的 {len(wrong_fragmented)} 個 task:")
    print(f"   {', '.join(wrong_fragmented)}")
    print(f"\n這些是 boundary 修復最多能影響的上界，且僅為上界："
          f"\n   答錯的原因未必是 boundary，修好也未必翻正。")


if __name__ == "__main__":
    sys.exit(main())
