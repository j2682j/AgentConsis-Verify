"""final_23 against final_22, and the funnel underneath both.

This is a new baseline for the current system, not an A/B against f22. The two
runs share their settings except for one addition -- protected-literal repair in
query generation -- and everything else that moved between them moved because
agents sample and the web answers differently each time. Earlier measurement put
that churn at four to six tasks with a net of zero, so a difference smaller than
that says nothing on its own, in either direction.

Two questions get asked of the funnel.

Where the gold answer stops. It is tracked through retrieved documents,
unverified references and strict evidence, because the layers fail differently
and a single "did it work" number cannot tell them apart.

And why reasoning steps come back unsupported. The obvious reading -- no
verified evidence, so nothing to support -- does not survive contact with task
027, which carried two verified evidence items and still had every step marked
unsupported. So evidence status is crossed against step support: if unsupported
steps appear under strict evidence too, the gap is in binding reasoning back to
evidence, not in the evidence itself.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, r"c:/SCP")

RUNS = {
    "f22": "c:/SCP/outputs/level1_final_22/tasks",
    "f23": "c:/SCP/outputs/level1_final_23/tasks",
}


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def search_raw(record: dict) -> dict:
    meta = (record.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
            return item["raw_result"]
    return {}


def load(directory: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(f"{directory}/*.json")):
        record = json.load(open(path, encoding="utf-8"))
        out[os.path.basename(path)[:3]] = record
    return out


def components(gold: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*[,;]\s*", gold) if p.strip()]
    return parts if len(parts) > 1 else [gold]


def contains_gold(text: str, gold: str) -> bool:
    folded = normalise(text).casefold()
    return all(
        re.search(rf"(?<![\w]){re.escape(p.casefold())}(?![\w])", folded)
        for p in components(gold)
    )


def funnel(record: dict, gold: str) -> dict[str, bool | None]:
    """Where the gold answer reached, layer by layer.

    `None` means the layer produced nothing to look in, which is a different
    fact from looking and not finding it.
    """

    raw = search_raw(record)

    def blob(items) -> str | None:
        if not isinstance(items, list) or not items:
            return None
        return json.dumps(items, ensure_ascii=False, default=str)

    layers: dict[str, bool | None] = {}
    for name, source in (
        ("retrieved_documents", raw.get("sources") or raw.get("web_searches")),
        ("unverified_references", raw.get("unverified_references")),
        ("strict_evidence", raw.get("verified_evidence_items")),
    ):
        text = blob(source)
        layers[name] = contains_gold(text, gold) if text else None

    # The context each Stage 1 run was actually given, which is `tool_context`
    # on the run itself. There is no single assembled context field; reading one
    # that does not exist reported every task as having no data, which is not
    # the same as the gold being absent.
    runs = [
        run
        for agent in ((record.get("network_summary") or {}).get("stage1_results") or [])
        for run in (agent.get("runs") or [])
        if isinstance(run, dict)
    ]
    contexts = [normalise(r.get("tool_context")) for r in runs]
    contexts = [c for c in contexts if c]
    layers["stage1_context"] = (
        any(contains_gold(c, gold) for c in contexts) if contexts else None
    )
    answers = [normalise(r.get("final_answer")) for r in runs]
    answers = [a for a in answers if a]
    layers["any_stage1_run"] = (
        any(contains_gold(a, gold) for a in answers) if answers else None
    )
    layers["selected_as_winner"] = bool(record.get("exact_match"))
    return layers


def step_support(record: dict) -> tuple[int, int]:
    """Supported and total reasoning steps across every Versa run."""

    supported = total = 0
    summary = record.get("network_summary") or {}
    for entry in summary.get("verifier_results") or []:
        for step in (entry or {}).get("step_scores") or []:
            if isinstance(step, dict):
                total += 1
                supported += str(step.get("support_status", "")).lower() == "supported"
    return supported, total


def support_reasons(record: dict) -> Counter:
    """Why steps were judged unsupported, in the checker's own words."""

    reasons = Counter()
    for entry in (record.get("network_summary") or {}).get("verifier_results") or []:
        for step in (entry or {}).get("step_scores") or []:
            if isinstance(step, dict) and str(step.get("support_status")) != "supported":
                reasons[str(step.get("support_reason") or "unknown")] += 1
    return reasons


def main() -> None:
    runs = {name: load(path) for name, path in RUNS.items()}
    tasks = sorted(set(runs["f22"]) | set(runs["f23"]))
    gold = {
        t: normalise(runs["f23"].get(t, runs["f22"].get(t, {})).get("expected"))
        for t in tasks
    }

    scores = {
        name: sum(1 for r in data.values() if r.get("exact_match"))
        for name, data in runs.items()
    }
    print(f"final_22 {scores['f22']}/{len(runs['f22'])}"
          f" = {scores['f22']/len(runs['f22']):.4f}")
    print(f"final_23 {scores['f23']}/{len(runs['f23'])}"
          f" = {scores['f23']/len(runs['f23']):.4f}")
    print(f"淨差 {scores['f23'] - scores['f22']:+d}")

    flipped, broke, same = [], [], 0
    for task in tasks:
        a = bool(runs["f22"].get(task, {}).get("exact_match"))
        b = bool(runs["f23"].get(task, {}).get("exact_match"))
        if b and not a:
            flipped.append(task)
        elif a and not b:
            broke.append(task)
        else:
            same += 1
    churn = len(flipped) + len(broke)
    print(f"\n=== 逐題轉移")
    print(f"   翻正 {len(flipped)}: {flipped}")
    print(f"   退步 {len(broke)}: {broke}")
    print(f"   不變 {same}　churn {churn} 題、淨 {len(flipped)-len(broke):+d}")
    print(f"   既有 baseline churn 為 4–6 題且淨值 0；"
          f"本次 churn {churn}、淨 {len(flipped)-len(broke):+d}"
          f" —— {'在該範圍內，不可歸因於任何改動' if churn <= 8 and abs(len(flipped)-len(broke)) <= 3 else '超出該範圍，值得追查'}")

    print(f"\n=== query literal integrity 觸發")
    triggered = []
    for task, record in runs["f23"].items():
        raw = search_raw(record)
        repairs = ((raw.get("diagnostics") or {}).get("query_plan") or {}).get(
            "query_repairs"
        ) or raw.get("query_repairs") or []
        if repairs:
            triggered.append((task, repairs))
    if triggered:
        for task, repairs in triggered:
            print(f"   {task}: {json.dumps(repairs, ensure_ascii=False)[:180]}")
    else:
        print("   0 次 —— 記錄本身是結果：本輪沒有任何 query 被修復器改寫，"
              "因此 f22→f23 的差異不可歸因於它")

    print(f"\n=== gold 漏斗（f23，僅有搜尋記錄的 task）")
    layer_names = ("retrieved_documents", "unverified_references", "strict_evidence",
                   "stage1_context", "any_stage1_run", "selected_as_winner")
    counts = {n: Counter() for n in layer_names}
    searched = [t for t in tasks if search_raw(runs["f23"].get(t, {}))]
    for task in searched:
        layers = funnel(runs["f23"][task], gold[task])
        for name in layer_names:
            value = layers[name]
            counts[name]["有" if value else ("無資料" if value is None else "無")] += 1
    for name in layer_names:
        c = counts[name]
        print(f"   {name:<24} 有 {c['有']:>3}　無 {c['無']:>3}　無資料 {c['無資料']:>3}"
              f"　／{len(searched)}")

    print(f"\n=== evidence 狀態 × reasoning step 支持度")
    table: dict[str, Counter] = defaultdict(Counter)
    for task in searched:
        record = runs["f23"][task]
        diagnostics = (search_raw(record).get("diagnostics") or {})
        status = diagnostics.get("evidence_status") or "unknown"
        supported, total = step_support(record)
        if not total:
            bucket = "無 step 記錄"
        elif supported:
            bucket = "有 supported step"
        else:
            bucket = "全部 unsupported"
        table[status][bucket] += 1
    buckets = ("有 supported step", "全部 unsupported", "無 step 記錄")
    print(f"   {'evidence_status':<20}" + "".join(f"{b:>20}" for b in buckets))
    for status, row in sorted(table.items()):
        print(f"   {status:<20}" + "".join(f"{row[b]:>20}" for b in buckets))
    print("   若 strict 列也集中在『全部 unsupported』，缺口在於把推理綁回證據，"
          "而非證據本身缺失")

    print(f"\n=== unsupported 的理由（checker 自述）")
    reasons = Counter()
    for task in searched:
        reasons.update(support_reasons(runs["f23"][task]))
    for reason, count in reasons.most_common(8):
        print(f"   {reason:<44} {count}")


if __name__ == "__main__":
    sys.exit(main())
