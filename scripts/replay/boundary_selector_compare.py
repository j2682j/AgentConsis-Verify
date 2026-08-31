"""Compare the shadow runs, and separate detection from selection.

Two prompt revisions both scored 5 of 37 exact, which looks like a stable
result and is not one: only 3 of those 5 are the same span. A 13% that lands on
different cases each run is 13% of noise, and reporting the number alone would
hide that.

What does not move is the failure mode. In v2, 21 of the 32 unrepaired
fragmented spans came back KEEP -- the model said the span was already whole.
One REPLACE landed on the wrong boundary. So the selector is not choosing badly
among boundaries; it is not noticing that a boundary needs choosing, and those
need different fixes. A better candidate set cannot help the first.

Detection is scored here as a two-class decision -- did the model propose a
change, on a span that needed one -- kept separate from whether the boundary it
then chose was right. Both are reported per run, so a revision that improves one
while breaking the other cannot show up as a wash.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, r"c:/SCP")

BASE = "c:/SCP/outputs/query_span_analysis"
RUNS = ("v1", "v2", "v3", "v4")


def load(version: str) -> dict[str, dict] | None:
    path = f"{BASE}/boundary_selector_shadow_{version}.jsonl"
    if not os.path.exists(path):
        return None
    return {
        json.loads(line)["annotation_id"]: json.loads(line)
        for line in open(path, encoding="utf-8")
        if line.strip()
    }


def main() -> None:
    frozen = json.load(open(f"{BASE}/boundary_candidates_frozen.json", encoding="utf-8"))
    entries = {e["annotation_id"]: e for e in frozen["entries"]}
    unreachable = set(frozen["candidate_unreachable"])

    runs = {version: load(version) for version in RUNS}
    runs = {version: rows for version, rows in runs.items() if rows}

    hits: dict[str, set[str]] = {}
    print(f"{'run':<5}{'n':>5}{'cond exact':>16}{'e2e exact':>16}"
          f"{'complete 誤動':>14}{'unrelated':>12}{'DEFER':>7}")
    for version, rows in runs.items():
        fragmented = [
            (entries[a], r) for a, r in rows.items()
            if entries[a]["population"] == "fragmented"
        ]
        reachable = [(e, r) for e, r in fragmented if e["annotation_id"] not in unreachable]
        hit = {
            e["annotation_id"] for e, r in reachable
            if e["gold_offsets"] and r["applied"] == e["gold_offsets"]
        }
        hits[version] = hit
        complete = [(entries[a], r) for a, r in rows.items()
                    if entries[a]["population"] == "complete"]
        unrelated = [(entries[a], r) for a, r in rows.items()
                     if entries[a]["population"] == "unrelated"]
        # A dropped span counts as moved: v1 and v2 could delete, v3 cannot.
        moved_complete = sum(
            1 for e, r in complete if r["applied"] is None or r["applied"] != e["span"]
        )
        moved_unrelated = sum(
            1 for e, r in unrelated if r["applied"] is None or r["applied"] != e["span"]
        )
        deferred = sum(1 for r in rows.values() if r["action"] == "DEFER")
        cond = f"{len(hit)}/{len(reachable)}={len(hit)/len(reachable):.3f}"
        e2e = f"{len(hit)}/{len(fragmented)}={len(hit)/len(fragmented):.3f}"
        print(f"{version:<5}{len(rows):>5}{cond:>16}{e2e:>16}"
              f"{f'{moved_complete}/{len(complete)}':>14}"
              f"{f'{moved_unrelated}/{len(unrelated)}':>12}{deferred:>7}")

    if len(hits) > 1:
        print(f"\n=== 命中是否穩定")
        versions = list(hits)
        common = set.intersection(*hits.values())
        union = set.union(*hits.values())
        print(f"   全部 run 皆命中 {len(common)}/{len(union)}: {sorted(common)}")
        for version in versions:
            only = hits[version] - set.union(
                *(hits[v] for v in versions if v != version)
            )
            if only:
                print(f"   僅 {version} 命中: {sorted(only)}")

    print(f"\n=== 偵測 vs 選擇（fragmented，排除 {sorted(unreachable)}）")
    print("   偵測 = 是否提出改動；選擇 = 提出後是否落在 gold")
    for version, rows in runs.items():
        reachable = [
            (entries[a], r) for a, r in rows.items()
            if entries[a]["population"] == "fragmented" and a not in unreachable
        ]
        proposed = [(e, r) for e, r in reachable if r["action"] == "REPLACE"]
        correct = [
            (e, r) for e, r in proposed
            if e["gold_offsets"] and r["applied"] == e["gold_offsets"]
        ]
        actions = Counter(r["action"] for _, r in reachable)
        print(f"   {version}: 偵測 {len(proposed)}/{len(reachable)}"
              f" = {len(proposed)/len(reachable):.3f}"
              f"、偵測後選對 {len(correct)}/{max(len(proposed), 1)}"
              f"   動作 {dict(actions)}")

    print(f"\n=== 判別力（REPLACE 的條件機率）")
    for version, rows in runs.items():
        def rate(population: str) -> tuple[int, int]:
            group = [r for a, r in rows.items()
                     if entries[a]["population"] == population]
            return sum(1 for r in group if r["action"] == "REPLACE"), len(group)

        frag_hit, frag_n = rate("fragmented")
        comp_hit, comp_n = rate("complete")
        ratio = (frag_hit / frag_n) / (comp_hit / comp_n) if comp_hit else float("inf")
        print(f"   {version}: P(REPLACE|fragmented) {frag_hit}/{frag_n} = {frag_hit/frag_n:.3f}"
              f"、P(REPLACE|complete) {comp_hit}/{comp_n} = {comp_hit/comp_n:.3f}"
              f"、比值 {ratio:.1f}")


if __name__ == "__main__":
    sys.exit(main())
