"""Run the selector without letting it change anything, and count what it did.

Shadow mode exists because the cost of the two error directions is not
symmetric. A fragmented span that stays fragmented is a task that was already
going to be wrong. A complete span that gets rewritten is a task that was going
to be right and now is not, and the standing rule on this system is that no
already-correct winner may regress. So the decisions are recorded and discarded,
and the 90 complete spans are the measurement that decides whether this ever
runs for real.

Every decision is written to JSONL as it is made, keyed by annotation, so a run
that dies partway resumes instead of re-paying for 133 model calls -- and so the
raw model output stays inspectable after the fact rather than being reduced to a
counter.

What these numbers are not: evidence of generalisation. These 38 fragmented
spans were read one by one while the generators were being debugged, and a
selector tuned against them is being tuned against its own development set.
S127 is excluded from selector accuracy entirely; its gold is not in the frozen
lattice, so no selector could reach it.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"c:/SCP")

from score.boundary_action_selector import (
    SelectorInput,
    apply,
    build_messages,
    call_model,
    defer_class,
    validate,
)

BASE = "c:/SCP/outputs/query_span_analysis"
FROZEN = f"{BASE}/boundary_candidates_frozen.json"
#: Each prompt revision gets its own file. v1 and v2 are kept as they ran --
#: re-parsing an old run under a newer contract would rewrite history, and v1's
#: 47 deletions are the reason the contract changed.
VERSION = os.getenv("SHADOW_VERSION", "v3")
SHADOW = f"{BASE}/boundary_selector_shadow_{VERSION}.jsonl"
MODEL = os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b")


def load_roles() -> dict[str, dict[str, str]]:
    return {
        row["annotation_id"]: row
        for row in csv.DictReader(
            open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8")
        )
    }


def existing() -> dict[str, dict]:
    if not os.path.exists(SHADOW):
        return {}
    out = {}
    for line in open(SHADOW, encoding="utf-8"):
        line = line.strip()
        if line:
            record = json.loads(line)
            out[record["annotation_id"]] = record
    return out


def run() -> None:
    frozen = json.load(open(FROZEN, encoding="utf-8"))
    roles = load_roles()
    done = existing()
    unreachable = set(frozen["candidate_unreachable"])

    from core.llm_client import LLMClient

    client = LLMClient()
    pending = [e for e in frozen["entries"] if e["annotation_id"] not in done]
    print(f"model={MODEL}  已完成 {len(done)}、待跑 {len(pending)}")

    with open(SHADOW, "a", encoding="utf-8") as handle:
        for index, entry in enumerate(pending, 1):
            annotation = roles.get(entry["annotation_id"], {})
            item = SelectorInput(
                annotation_id=entry["annotation_id"],
                context=entry["context"],
                span=tuple(entry["span"]),
                question_role=annotation.get("answer_role", ""),
                answer_target=annotation.get("answer_target", ""),
            )
            try:
                raw = call_model(client, MODEL, build_messages(item))
            except Exception as exc:  # a dead call is a DEFER, not a crash
                raw = f"__error__ {type(exc).__name__}: {exc}"
            allowed = {(c[0], c[1]) for c in entry["candidates"]}
            decision = validate(raw, item, allowed)
            applied = apply(decision, item)
            record = {
                "annotation_id": entry["annotation_id"],
                "population": entry["population"],
                "action": decision.action,
                "defer_reason": decision.defer_reason,
                "defer_class": defer_class(decision.defer_reason),
                "normalisations": list(decision.normalisations),
                "raw_model_text": decision.raw_model_text,
                "marker_stripped_text": decision.marker_stripped_text,
                "marker_stripped": decision.marker_stripped,
                "selected": [decision.start, decision.end]
                if decision.start is not None
                else None,
                "applied": list(applied),
                "raw": raw[:2000],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0:
                print(f"   {index}/{len(pending)}")

    report(frozen, roles, existing(), unreachable)


def report(frozen: dict, roles: dict, decisions: dict, unreachable: set) -> None:
    entries = {e["annotation_id"]: e for e in frozen["entries"]}
    by_population: dict[str, list] = defaultdict(list)
    for annotation_id, record in decisions.items():
        by_population[record["population"]].append((entries[annotation_id], record))

    print(f"\n=== 動作分布（期望：complete→KEEP、fragmented→REPLACE、"
          f"unrelated→KEEP 或 DEFER）")
    for population, rows in sorted(by_population.items()):
        counts = Counter(r["action"] for _, r in rows)
        line = "  ".join(f"{a} {counts[a]}" for a in ("KEEP", "REPLACE", "DEFER"))
        print(f"   {population:<12} n={len(rows):<4} {line}")

    # Two denominators, and both are reported. Scoring only the 37 spans whose
    # gold the lattice can reach measures the selector on its own terms; scoring
    # all 38 measures what the system delivers. Publishing only the first hides
    # S127, which generation cannot produce and which is a real miss.
    all_fragmented = by_population["fragmented"]
    reachable = [
        (e, r) for e, r in all_fragmented if e["annotation_id"] not in unreachable
    ]

    def exact_of(rows: list) -> int:
        return sum(
            1 for e, r in rows
            if e["gold_offsets"] and r["applied"] == e["gold_offsets"]
        )

    def acceptable_of(rows: list) -> int:
        return sum(
            1 for e, r in rows
            if e["acceptable_alternative"]
            and e["context"][r["applied"][0] : r["applied"][1]].casefold()
            == e["acceptable_alternative"].casefold()
        )

    conditional, end_to_end = exact_of(reachable), exact_of(all_fragmented)
    unchanged = sum(1 for e, r in reachable if r["applied"] == e["span"])
    print(f"\n=== fragmented")
    print(f"   conditional selector accuracy（僅 candidate-reachable {len(reachable)} 筆）")
    print(f"      exact          {conditional}/{len(reachable)}"
          f" = {conditional/len(reachable):.3f}")
    print(f"      含 acceptable  {conditional + acceptable_of(reachable)}/{len(reachable)}")
    print(f"   end-to-end recovery（全部 {len(all_fragmented)} 筆）")
    print(f"      exact          {end_to_end}/{len(all_fragmented)}"
          f" = {end_to_end/len(all_fragmented):.3f}")
    print(f"      未達成中 {len(unreachable)} 筆為 candidate generation 失敗: "
          f"{sorted(unreachable)}")
    print(f"   維持原樣（未修復）  {unchanged}/{len(reachable)}")

    complete = by_population["complete"]
    kept = sum(1 for e, r in complete if r["applied"] == e["span"])
    mutated = sum(1 for e, r in complete if r["applied"] != e["span"])
    deferred = sum(1 for _, r in complete if r["action"] == "DEFER")
    print(f"\n=== complete {len(complete)} 筆（期望 KEEP，不得 REPLACE）")
    print(f"   維持原樣          {kept}/{len(complete)} = {kept/len(complete):.3f}")
    print(f"      其中 DEFER 回退 {deferred}")
    print(f"   錯誤 REPLACE      {mutated}/{len(complete)}"
          f"{'   <- 退步' if mutated else ''}")

    unrelated = by_population["unrelated"]
    if unrelated:
        safe = sum(1 for _, r in unrelated if r["action"] in ("KEEP", "DEFER"))
        expanded = sum(1 for e, r in unrelated if r["applied"] != e["span"])
        print(f"\n=== unrelated {len(unrelated)} 筆（期望 KEEP 或 DEFER，不得 REPLACE）")
        print(f"   KEEP 或 DEFER     {safe}/{len(unrelated)}")
        print(f"   錯誤 REPLACE      {expanded}/{len(unrelated)}"
              f"{'   <- 退步' if expanded else ''}")
        print("   是否應進入 query 由 query admission 決定，不在此模組")

    print(f"\n=== DEFER {sum(1 for r in decisions.values() if r['action'] == 'DEFER')}"
          f"/{len(decisions)}")
    reasons = Counter(
        r["defer_reason"] for r in decisions.values() if r["defer_reason"]
    )
    for reason, count in reasons.most_common():
        print(f"   {reason:<24} {count}")
    grounding = sum(
        count for reason, count in reasons.items()
        if reason in ("not_in_context", "not_in_candidate_set", "ambiguous_occurrence")
    )
    print(f"   其中 grounding 失敗      {grounding}/{len(decisions)}"
          f" = {grounding/len(decisions):.3f}")

    leaked = sum(1 for r in decisions.values() if r.get("marker_stripped"))
    print(f"\n=== marker leakage {leaked}/{len(decisions)}"
          f" = {leaked/len(decisions):.3f}")
    print("   只還原本模組注入的精確字串，未通用移除括號")

    print(f"\n=== format_instruction 邊界變化")
    format_rows = [
        (entries[annotation_id], r) for annotation_id, r in decisions.items()
        if roles.get(annotation_id, {}).get("human_role") == "format_instruction"
    ]
    if format_rows:
        intact = sum(1 for e, r in format_rows if r["applied"] == e["span"])
        moved = len(format_rows) - intact
        print(f"   {len(format_rows)} 筆: 未動 {intact}、邊界移動 {moved}")
        print("   是否排除由 role/admission 處理；此處只確認邊界完整恢復")

    print(f"\n=== 最終 query 差異（僅記錄，未套用）")
    per_task: dict[str, list] = defaultdict(list)
    for annotation_id, record in decisions.items():
        task = roles.get(annotation_id, {}).get("task_id", "?")
        per_task[task].append((entries[annotation_id], record))
    changed_tasks = [
        task for task, rows in per_task.items()
        if any(r["applied"] != e["span"] for e, r in rows)
    ]
    moved = sum(
        1 for rows in per_task.values() for e, r in rows if r["applied"] != e["span"]
    )
    print(f"   {len(changed_tasks)}/{len(per_task)} 個 task 的 span 文字會改變")
    print(f"   span 數不變（{len(decisions)}），僅 {moved} 筆邊界移動")


if __name__ == "__main__":
    sys.exit(run())
