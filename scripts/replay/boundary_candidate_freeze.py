"""Freeze the candidate lattice, so the selector is measured against a fixed set.

Candidate generation reached 37 of 38 and stops here. What follows is selection,
and a selector cannot be evaluated against a target that moves underneath it: if
generation keeps changing, a selector that improves and a lattice that widened
look identical in the numbers. So the offsets are written out once, with the
generator provenance that produced them, and the selector is only ever allowed
to return a boundary from this file.

Two things are recorded that the offsets alone do not carry.

The coverage attribution, because the ablation's answer was counterintuitive and
will not survive in anyone's memory: `expansion` is the whole engine (16 cases
reachable through it alone), `subtree`, `bracket` and `overlap_replacement`
contribute one case each, and `keep` looks worthless on the fragmented set while
being the only generator that lets 22 of the 95 controls stay untouched. Five
generators contribute nothing here and are kept anyway -- removing them saves
1.2 candidates out of 98, which is not worth losing type coverage on data that
has not been collected yet.

And S127, which is unreachable: its gold needs 13 tokens of right expansion and
the cap is 12. The cap was already raised from 6 to 12 against these same 38
cases, so raising it again would be fitting the same constant to the same data
twice. It is marked here so the selector is never charged for missing it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, r"c:/SCP")

from scripts.replay.boundary_candidate_oracle import (
    GENERATORS,
    MAX_EXPANSION_TOKENS,
    candidates_for,
)
from scripts.replay.boundary_recovery_prototype import (
    Recovery,
    collapse,
    load_controls,
    load_gold,
)

BASE = "c:/SCP/outputs/query_span_analysis"
FROZEN = f"{BASE}/boundary_candidates_frozen.json"
SCHEMA_VERSION = 1

#: Read off the ablation rather than asserted. Regenerating this file will not
#: recompute them; they describe the batch the lattice was frozen on.
COVERAGE = {
    "expansion": "16 筆獨佔命中；佔平均候選 98.7 中的 88.3",
    "subtree": "1 筆獨佔命中（S096）",
    "bracket": "1 筆獨佔命中（S022）",
    "overlap_replacement": "1 筆獨佔命中（S129）",
    "keep": "fragmented 0 筆，但移除後 22/95 controls 失去原 span",
    "noun_chunk": "0 筆；保留以維持未來資料的型別覆蓋",
    "ner": "0 筆；保留以維持未來資料的型別覆蓋",
    "contraction": "0 筆；possessive 已由 candidate 層處理，保留備用",
    "sentence": "0 筆；保留以維持未來資料的型別覆蓋",
    "hyphen": "0 筆；保留以維持未來資料的型別覆蓋",
}

#: Not a selector error. Charged to generation, and excluded from selector
#: accuracy wherever it is reported.
CANDIDATE_UNREACHABLE = {"S127": "gold 需向右 13 token，MAX_EXPANSION_TOKENS = 12"}


def freeze() -> dict:
    recovery = Recovery()
    entries: list[dict] = []

    for case in load_gold():
        candidates = candidates_for(recovery.nlp(case.context), case.context, case.span_text)
        target = collapse(case.gold_span).casefold()
        reachable = [
            [c.start, c.end]
            for c in candidates
            if c.text(case.context).casefold() == target
        ]
        entries.append(
            {
                "annotation_id": case.annotation_id,
                "population": "fragmented",
                "context": case.context,
                "span": list(case.span_offsets) if case.span_offsets else None,
                "span_text": case.span_text,
                "gold_text": case.gold_span,
                "gold_offsets": reachable[0] if reachable else None,
                "gold_reachable": bool(reachable),
                "unit_type": case.unit_type,
                "repair_direction": case.repair_direction,
                "acceptable_alternative": case.acceptable_alternative,
                "gold_repair_source": case.gold_repair_source,
                "candidates": [
                    [c.start, c.end, list(c.generators)] for c in candidates
                ],
            }
        )

    for annotation_id, span, context, kind in load_controls():
        candidates = candidates_for(recovery.nlp(context), context, span)
        at = context.casefold().find(span.casefold())
        entries.append(
            {
                "annotation_id": annotation_id,
                "population": kind,
                "context": context,
                "span": [at, at + len(span)] if at >= 0 else None,
                "span_text": span,
                "gold_text": span if kind == "complete" else "",
                "gold_offsets": [at, at + len(span)] if kind == "complete" and at >= 0 else None,
                "gold_reachable": at >= 0,
                "unit_type": "",
                "repair_direction": "keep" if kind == "complete" else "drop",
                "acceptable_alternative": "",
                "gold_repair_source": "control",
                "candidates": [
                    [c.start, c.end, list(c.generators)] for c in candidates
                ],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generators": list(GENERATORS),
        "max_expansion_tokens": MAX_EXPANSION_TOKENS,
        "coverage_attribution": COVERAGE,
        "candidate_unreachable": CANDIDATE_UNREACHABLE,
        "entries": entries,
    }


def main() -> None:
    frozen = freeze()
    with open(FROZEN, "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=1)

    entries = frozen["entries"]
    populations = Counter(e["population"] for e in entries)
    sizes = sorted(len(e["candidates"]) for e in entries)
    fragmented = [e for e in entries if e["population"] == "fragmented"]
    reachable = sum(1 for e in fragmented if e["gold_reachable"])

    print(f"凍結 {len(entries)} 筆 -> {FROZEN}")
    print(f"   {dict(populations)}")
    print(f"   fragmented gold 可達 {reachable}/{len(fragmented)}")
    print(f"   候選數 中位數 {sizes[len(sizes)//2]}、P95 {sizes[int(len(sizes)*0.95)-1]}、"
          f"最大 {sizes[-1]}、總計 {sum(sizes)}")
    for annotation_id, why in CANDIDATE_UNREACHABLE.items():
        print(f"   已知不可達（不計入 selector 錯誤）: {annotation_id} — {why}")


if __name__ == "__main__":
    sys.exit(main())
