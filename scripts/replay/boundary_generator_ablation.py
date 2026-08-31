"""What does each generator actually contribute to the 0.974 ceiling?

Per-generator recall answers a different question than the one that matters for
pruning. `subtree` reaches 15 of 38 gold spans on its own, but `expansion`
reaches 32, and if every subtree hit is also an expansion hit then dropping
subtree costs nothing. Removing a generator by its solo score would delete
coverage that is not duplicated and keep generators that only ever agree.

So each generator is removed from the union and the union re-scored. A candidate
survives removal if any *other* generator also proposed that boundary, which is
what the provenance set on `Candidate` is for. Two numbers come out of it: what
the ceiling costs, and how many candidates the selector no longer has to rank.

Cost is reported in cases, not percentages. With 38 gold spans a single case is
2.6 points, and a generator whose removal costs "only 2.6%" has cost a task.
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, r"c:/SCP")

from scripts.replay.boundary_candidate_oracle import GENERATORS, candidates_for
from scripts.replay.boundary_recovery_prototype import Recovery, collapse, load_gold


def main() -> None:
    gold_cases = load_gold()
    recovery = Recovery()

    per_case = []
    for case in gold_cases:
        doc = recovery.nlp(case.context)
        per_case.append(
            (case, candidates_for(doc, case.context, case.span_text),
             collapse(case.gold_span).casefold())
        )

    full_hits = 0
    full_sizes: list[int] = []
    for case, candidates, target in per_case:
        texts = {c.text(case.context).casefold() for c in candidates}
        full_sizes.append(len(texts))
        full_hits += target in texts

    total = len(gold_cases)
    print(f"完整 union: {full_hits}/{total} = {full_hits/total:.3f}"
          f"、平均候選 {sum(full_sizes)/total:.1f}\n")

    print(f"{'移除的 generator':<22}{'union recall':>16}{'損失':>8}{'平均候選':>10}{'減少':>8}")
    rows = []
    for dropped in GENERATORS:
        hits, sizes, lost_ids = 0, [], []
        for case, candidates, target in per_case:
            texts = {
                c.text(case.context).casefold()
                for c in candidates
                if any(g != dropped for g in c.generators)
            }
            sizes.append(len(texts))
            if target in texts:
                hits += 1
            elif target in {c.text(case.context).casefold() for c in candidates}:
                lost_ids.append(case.annotation_id)
        mean = sum(sizes) / total
        rows.append((dropped, hits, mean, lost_ids))
        print(f"{dropped:<22}{hits}/{total} = {hits/total:>8.3f}"
              f"{full_hits - hits:>8}{mean:>10.1f}"
              f"{sum(full_sizes)/total - mean:>8.1f}")

    print(f"\n可移除（損失 0 筆）:")
    free = [r for r in rows if r[1] == full_hits]
    for dropped, _, mean, _ in free:
        print(f"   {dropped:<22} 候選減少 {sum(full_sizes)/total - mean:.1f}")
    if not free:
        print("   無")

    print(f"\n不可移除（各自帶走的題目）:")
    for dropped, hits, _, lost_ids in rows:
        if lost_ids:
            print(f"   {dropped:<22} -{len(lost_ids)} 筆: {', '.join(lost_ids)}")

    # Removing several at once is not the sum of removing each alone: two
    # generators can be individually free while jointly covering the only route
    # to a case, so the free set is re-scored together rather than assumed.
    free_names = {r[0] for r in free}
    if free_names:
        hits, sizes = 0, []
        for case, candidates, target in per_case:
            texts = {
                c.text(case.context).casefold()
                for c in candidates
                if any(g not in free_names for g in c.generators)
            }
            sizes.append(len(texts))
            hits += target in texts
        print(f"\n同時移除 {sorted(free_names)}:")
        print(f"   union {hits}/{total} = {hits/total:.3f}"
              f"、平均候選 {sum(sizes)/total:.1f}"
              f"（原 {sum(full_sizes)/total:.1f}）")


def controls_check() -> None:
    """The free set is only free on the population it was scored on.

    The ablation above runs on the 38 fragmented spans, where `keep` proposes a
    boundary already known to be wrong and therefore never matches gold. The 95
    control spans are the opposite case: the correct action there is to leave
    the span alone, and `keep` is the generator that makes that expressible. A
    prune decided on the fragmented set alone would remove it.
    """

    from scripts.replay.boundary_recovery_prototype import load_controls

    controls = load_controls()
    recovery = Recovery()
    intact = defaultdict(int)
    for annotation_id, span, context, kind in controls:
        doc = recovery.nlp(context)
        candidates = candidates_for(doc, context, span)
        for dropped in GENERATORS:
            texts = {
                c.text(context).casefold()
                for c in candidates
                if any(g != dropped for g in c.generators)
            }
            intact[dropped] += span.casefold() in texts

    print(f"\n=== 對照組 {len(controls)} 筆：移除後原 span 是否仍在候選內")
    for dropped in GENERATORS:
        flag = "" if intact[dropped] == len(controls) else "   <- 不可移除"
        print(f"   移除 {dropped:<20} {intact[dropped]}/{len(controls)}{flag}")


if __name__ == "__main__":
    main()
    sys.exit(controls_check())
