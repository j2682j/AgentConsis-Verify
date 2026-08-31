"""The holdout test. Written before the labels exist, and run once.

Everything this file measures was decided in advance, including which numbers
would count as a refusal to deploy. That ordering is the whole point: with the
labels in hand it is easy to find a denominator that flatters a result, and the
design set has already been through four prompt revisions chosen by looking at
exactly these numbers.

Two things are reported side by side throughout.

Raw, where the gold boundary is the only correct answer. And alternative-aware,
where a human-marked equivalent counts too. The design set had no alternative
column on its controls, and so 12 spans the selector widened -- `1928` to `1928
Summer Olympics`, `NASA` to `NASA award number` -- could not be told apart from
damage. Reporting only the generous number would hide real mutation; reporting
only the strict one would count a difference of wording as a regression. Both,
or neither means anything.

DEFER is likewise never a single count. In the design runs the model chose to
defer zero times out of 133, and the 36 refusals that looked like caution were
replies that failed to parse. An accident that lands in the safe direction is
not an abstention, and a report that adds them together would say the selector
knows when to stop, which it does not.

`unclear` spans stay out of every accuracy denominator and are counted anyway:
a span the annotator could not judge is not evidence either way, but a selector
that mangles them is still doing something.

What this set is, precisely: a **label-blind amended holdout**. The tasks took no
part in any design decision and no human label existed when the configuration
was corrected twice -- but raw decisions, parse failures and the difference
between two decode paths have all been looked at on these spans, so it is not an
untouched test set and must not be described as one. The 93 reserved Level 1
test questions are the confirmatory holdout, and stay unopened.
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
    parse,
    validate,
)

OUT = "c:/SCP/outputs/query_span_analysis/holdout"
DECISIONS = f"{OUT}/boundary_holdout_decisions.jsonl"
MODEL = os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b")

#: What the labels mean for the selector. `overexpanded` joins `fragmented` as a
#: population needing repair -- the boundary is wrong in the other direction,
#: and REPLACE is still the right action.
NEEDS_REPAIR = ("fragmented", "overexpanded")
MUST_NOT_MOVE = ("complete",)
MUST_NOT_REPLACE = ("complete", "unrelated")

#: Decided before the run. Each rule names the number that triggers it, so a
#: result cannot be argued into passing afterwards.
DECISION_RULES = (
    ("complete harmful mutation ≈10% 以上", "拒絕部署"),
    ("安全性主要來自 parse failure", "不得視為有效 abstention"),
    ("改善只出現在 alternative-aware", "必須同時保留 raw 指標"),
    ("oracle 高但 selector 低", "下一步是 abstention/selection"),
    ("oracle 本身低", "回到 generator，但不得用本 holdout 調參"),
)


def load_annotations() -> dict[str, dict]:
    """Labels, joined to canonical text by id and nothing else.

    Only the label columns are taken from the annotated file. Every other field
    comes from the canonical copy, because the design set lost 17 rows to an
    editor that rewrote non-ASCII characters on save, and a gold span that no
    longer matches its own context is not a gold span.
    """

    path = f"{OUT}/boundary_holdout_annotated.csv"
    if not os.path.exists(path):
        raise SystemExit(f"尚未標註: {path} 不存在")
    canonical = {
        row["annotation_id"]: row
        for row in csv.DictReader(open(f"{OUT}/_canonical_holdout.csv", encoding="utf-8"))
    }
    label_columns = (
        "human_boundary", "human_gold_span", "repair_direction", "unit_type",
        "boundary_form", "boundary_confidence", "acceptable_alternative", "notes",
    )
    out: dict[str, dict] = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        annotation_id = (row.get("annotation_id") or "").strip()
        if annotation_id not in canonical:
            continue
        record = dict(canonical[annotation_id])
        record.update({k: (row.get(k) or "").strip() for k in label_columns})
        out[annotation_id] = record
    return out


def ground(text: str, context: str) -> tuple[int, int] | None:
    """Where a labelled span sits, or nothing if it is not verbatim.

    A gold boundary that cannot be found in its own context is a paraphrase, and
    scoring against it would measure agreement with someone's wording.
    """

    if not text:
        return None
    at = context.casefold().find(text.casefold())
    return (at, at + len(text)) if at >= 0 else None


def load_canonical() -> dict[str, dict]:
    return {
        row["annotation_id"]: row
        for row in csv.DictReader(open(f"{OUT}/_canonical_holdout.csv", encoding="utf-8"))
    }


def decide(client, row: dict, frozen: dict) -> dict:
    """One span, one call, one validated decision."""

    item = SelectorInput(
        annotation_id=row["annotation_id"],
        context=frozen["context"],
        span=tuple(frozen["span"]),
        question_role=row.get("answer_role", ""),
        answer_target=row.get("answer_target", ""),
    )
    try:
        raw = call_model(client, MODEL, build_messages(item))
    except Exception as exc:
        raw = f"__error__ {type(exc).__name__}: {exc}"
    allowed = {(c[0], c[1]) for c in frozen["candidates"]}
    decision = validate(raw, item, allowed)
    return {
        "annotation_id": row["annotation_id"],
        "action": decision.action,
        "defer_reason": decision.defer_reason,
        "defer_class": defer_class(decision.defer_reason),
        "raw_model_text": decision.raw_model_text,
        "marker_stripped_text": decision.marker_stripped_text,
        "marker_stripped": decision.marker_stripped,
        "applied": list(apply(decision, item)),
        "raw": raw[:2000],
    }


def determinism_probe(rows: dict[str, dict], lattice: dict[str, dict], size: int = 10) -> bool:
    """Run the same spans twice and require the same answers.

    Declared in the run protocol before the labels existed, because a single
    scored run is only defensible if the run is reproducible. This does not feed
    the score -- it says whether the seed is doing anything.

    Sameness is judged on the parsed decision, not the returned bytes. The model
    emits the same object pretty-printed on one call and compact on the next,
    and counting that as drift would report a whitespace difference as an
    unstable system. What has to be stable is the action and the span.
    """

    from core.llm_client import LLMClient

    client = LLMClient()
    sample = [a for a in sorted(rows) if lattice[a]["span"]][:size]
    print(f"determinism probe: {len(sample)} span，各跑兩次")
    stable = True
    for annotation_id in sample:
        first = decide(client, rows[annotation_id], lattice[annotation_id])
        second = decide(client, rows[annotation_id], lattice[annotation_id])
        same = (
            (first["action"], first["applied"]) == (second["action"], second["applied"])
            and parse(first["raw"]) == parse(second["raw"])
        )
        stable &= same
        if not same:
            print(f"   {annotation_id} 不一致")
            print(f"      1: {first['action']} {parse(first['raw'])}")
            print(f"      2: {second['action']} {parse(second['raw'])}")
    print(f"   結果: {'完全一致' if stable else '不一致 —— 單次結果不可重現，'
                                            '依 protocol 應如實報告，不做平均'}")
    return stable


def run_selector(rows: dict[str, dict], lattice: dict[str, dict]) -> dict[str, dict]:
    """The one scored run. Labels are not needed and deliberately not loaded."""

    from core.llm_client import LLMClient

    done: dict[str, dict] = {}
    if os.path.exists(DECISIONS):
        for line in open(DECISIONS, encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                done[record["annotation_id"]] = record

    client = LLMClient()
    pending = [a for a in sorted(rows) if a not in done]
    print(f"model={MODEL}  已完成 {len(done)}、待跑 {len(pending)}")

    with open(DECISIONS, "a", encoding="utf-8") as handle:
        for index, annotation_id in enumerate(pending, 1):
            row, frozen = rows[annotation_id], lattice[annotation_id]
            if not frozen["span"]:
                continue
            record = decide(client, row, frozen)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done[annotation_id] = record
            if index % 10 == 0:
                print(f"   {index}/{len(pending)}")
    return done


def report(annotations: dict, lattice: dict, decisions: dict) -> None:
    classes = Counter(r["human_boundary"] or "unlabelled" for r in annotations.values())
    print(f"\n=== boundary class 分布（n={len(annotations)}）")
    for name, count in classes.most_common():
        print(f"   {name:<16} {count}")

    scored = {
        a: r for a, r in annotations.items()
        if r["human_boundary"] not in ("unclear", "", "unlabelled") and a in decisions
    }
    unclear = [a for a, r in annotations.items() if r["human_boundary"] == "unclear"]
    print(f"   計入 accuracy 分母 {len(scored)}、unclear 排除 {len(unclear)}")
    if unclear:
        moved = sum(
            1 for a in unclear
            if a in decisions and lattice[a]["span"]
            and decisions[a]["applied"] != lattice[a]["span"]
        )
        print(f"   unclear 中被改動 {moved}/{len(unclear)}（不計分，但記錄）")

    def gold_of(annotation_id: str) -> tuple[int, int] | None:
        row, frozen = annotations[annotation_id], lattice[annotation_id]
        if row["human_boundary"] in MUST_NOT_MOVE:
            return tuple(frozen["span"]) if frozen["span"] else None
        return ground(row["human_gold_span"], frozen["context"])

    def alternative_of(annotation_id: str) -> tuple[int, int] | None:
        return ground(
            annotations[annotation_id]["acceptable_alternative"],
            lattice[annotation_id]["context"],
        )

    repair = [a for a in scored if annotations[a]["human_boundary"] in NEEDS_REPAIR]
    reachable, unreachable_ids = [], []
    for annotation_id in repair:
        gold = gold_of(annotation_id)
        allowed = {(c[0], c[1]) for c in lattice[annotation_id]["candidates"]}
        (reachable if gold in allowed else unreachable_ids).append(annotation_id)

    print(f"\n=== candidate oracle（{len(repair)} 筆需修復）")
    print(f"   gold 在 lattice 內 {len(reachable)}/{len(repair)}"
          f" = {len(reachable)/max(len(repair), 1):.3f}")
    ungrounded = [a for a in repair if gold_of(a) is None]
    if ungrounded:
        print(f"   gold 無法在 context 中定位 {len(ungrounded)}: {sorted(ungrounded)[:8]}")

    def hits(ids: list[str], allow_alternative: bool) -> int:
        total = 0
        for annotation_id in ids:
            applied = tuple(decisions[annotation_id]["applied"])
            targets = {gold_of(annotation_id)}
            if allow_alternative:
                targets.add(alternative_of(annotation_id))
            total += applied in {t for t in targets if t}
        return total

    print(f"\n=== 修復（{len(repair)} 筆 fragmented/overexpanded）")
    for label, ids in (("conditional（僅 candidate-reachable）", reachable),
                       ("end-to-end（全部）", repair)):
        raw_hit = hits(ids, False)
        alt_hit = hits(ids, True)
        print(f"   {label} n={len(ids)}")
        print(f"      raw               {raw_hit}/{len(ids)}"
              f" = {raw_hit/max(len(ids), 1):.3f}")
        print(f"      alternative-aware {alt_hit}/{len(ids)}"
              f" = {alt_hit/max(len(ids), 1):.3f}")
    if unreachable_ids:
        print(f"   candidate generation 失敗 {len(unreachable_ids)}: "
              f"{sorted(unreachable_ids)}")

    complete = [a for a in scored if annotations[a]["human_boundary"] in MUST_NOT_MOVE]
    raw_mutated = [
        a for a in complete
        if tuple(decisions[a]["applied"]) != tuple(lattice[a]["span"])
    ]
    harmful = [a for a in raw_mutated if tuple(decisions[a]["applied"]) != alternative_of(a)]
    print(f"\n=== complete {len(complete)} 筆（不得移動）")
    print(f"   raw mutation      {len(raw_mutated)}/{len(complete)}"
          f" = {len(raw_mutated)/max(len(complete), 1):.3f}")
    print(f"   harmful mutation  {len(harmful)}/{len(complete)}"
          f" = {len(harmful)/max(len(complete), 1):.3f}")
    for annotation_id in harmful[:10]:
        frozen, applied = lattice[annotation_id], decisions[annotation_id]["applied"]
        print(f"      {annotation_id} {frozen['span_text'][:26]!r}"
              f" -> {frozen['context'][applied[0]:applied[1]][:44]!r}")

    unrelated = [a for a in scored if annotations[a]["human_boundary"] == "unrelated"]
    if unrelated:
        replaced = sum(1 for a in unrelated if decisions[a]["action"] == "REPLACE")
        print(f"\n=== unrelated {len(unrelated)} 筆（不得 REPLACE）")
        print(f"   錯誤 REPLACE      {replaced}/{len(unrelated)}"
              f" = {replaced/len(unrelated):.3f}")

    kinds = Counter(r["defer_class"] for r in decisions.values() if r["defer_class"])
    total = len(decisions)
    explicit = kinds.get("explicit_defer", 0)
    accidental = total and sum(v for k, v in kinds.items() if k != "explicit_defer")
    print(f"\n=== DEFER {sum(kinds.values())}/{total}")
    for name in ("explicit_defer", "parse_failure_fallback",
                 "invalid_action_fallback", "grounding_failure_fallback"):
        print(f"   {name:<28} {kinds.get(name, 0)}")
    print(f"   explicit {explicit}/{total} = {explicit/total:.3f}"
          f"、accidental {accidental}/{total} = {accidental/total:.3f}")
    fallback = sum(
        kinds.get(k, 0) for k in
        ("parse_failure_fallback", "invalid_action_fallback", "grounding_failure_fallback")
    )
    print(f"   parser/validator fallback {fallback}/{total} = {fallback/total:.3f}")
    leaked = sum(1 for r in decisions.values() if r.get("marker_stripped"))
    print(f"   marker leakage {leaked}/{total} = {leaked/total:.3f}")

    print(f"\n=== 事前決策規則")
    harmful_rate = len(harmful) / max(len(complete), 1)
    safety_from_parse = kinds.get("parse_failure_fallback", 0) > explicit
    oracle_rate = len(reachable) / max(len(repair), 1)
    selector_rate = hits(reachable, False) / max(len(reachable), 1)
    raw_rate = hits(repair, False) / max(len(repair), 1)
    alt_rate = hits(repair, True) / max(len(repair), 1)
    gates = [
        ("1. complete harmful mutation ≈10% 以上 -> 拒絕部署",
         harmful_rate >= 0.08, f"{harmful_rate:.3f}"),
        ("2. 安全性主要來自 parse failure -> 不得視為有效 abstention",
         safety_from_parse,
         f"parse {kinds.get('parse_failure_fallback', 0)} vs explicit {explicit}"),
        ("3. 改善只出現在 alternative-aware -> 必須同時保留 raw",
         alt_rate > raw_rate and raw_rate < 0.3,
         f"raw {raw_rate:.3f} / alt {alt_rate:.3f}"),
        ("4. oracle 高但 selector 低 -> 下一步是 abstention/selection",
         oracle_rate >= 0.8 and selector_rate < 0.6,
         f"oracle {oracle_rate:.3f} / selector {selector_rate:.3f}"),
        ("5. oracle 本身低 -> 回到 generator，不得用本 holdout 調參",
         oracle_rate < 0.8, f"oracle {oracle_rate:.3f}"),
    ]
    for label, triggered, evidence in gates:
        print(f"   {'觸發' if triggered else '未觸發'}  {label}")
        print(f"           {evidence}")

    deployable = not gates[0][1]
    print(f"\n=== 判定")
    if deployable:
        print("   通過部署閘門。不再修改，改於保留的 93 題抽取新 task-level"
              "樣本做 confirmatory evaluation。")
    else:
        print("   拒絕部署目前 selector。若要依本批錯誤調整，這 24 個 task"
              "立即降級為 development evidence，修改後只能以 93 題作新 holdout。")
    print("   結果接近門檻時不得靠重跑取較好者：依事前規則判定，"
          "並將約 10% 不穩定性列為限制。")
    print(f"\n   本 holdout 已計分一次，不得因結果修改 prompt、candidate"
          f"generator、decode 或 validator 後重跑並仍稱為 holdout。")
    print(f"   confirmatory evaluation 使用保留的 93 題 Level 1 test split。")


def load_lattice() -> dict[str, dict]:
    return {
        e["annotation_id"]: e
        for e in json.load(
            open(f"{OUT}/boundary_holdout_candidates.json", encoding="utf-8")
        )["entries"]
    }


def main(argv: list[str] | None = None) -> None:
    """Two phases, because the selector does not need the labels.

    `decide` runs first and can run before anyone has annotated anything, which
    is the stronger ordering: the annotator never sees a selector output, and
    the selector never sees a label. `score` then joins the two. Splitting them
    also means a disputed score can be recomputed from the recorded decisions
    without paying for the model again -- or re-running it, which the protocol
    forbids.
    """

    argv = sys.argv[1:] if argv is None else argv
    phase = argv[0] if argv else "score"
    verify_freeze()
    lattice = load_lattice()

    if phase == "decide":
        rows = load_canonical()
        if "--probe" in argv:
            determinism_probe(rows, lattice)
        decisions = run_selector(rows, lattice)
        actions = Counter(r["action"] for r in decisions.values())
        print(f"\n決策已記錄 {len(decisions)}/{len(rows)} -> {DECISIONS}")
        print(f"   {dict(actions)}")
        print("   尚未計分：需要 boundary_holdout_annotated.csv")
        return

    annotations = load_annotations()
    decisions = {}
    for line in open(DECISIONS, encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            decisions[record["annotation_id"]] = record
    if not decisions:
        raise SystemExit("尚未執行 decide 階段")
    report(annotations, lattice, decisions)


def verify_freeze() -> None:
    """Confirm nothing moved between freezing and testing."""

    import hashlib

    manifest = json.load(open(f"{OUT}/holdout_manifest.json", encoding="utf-8"))
    from score.boundary_action_selector import SYSTEM_PROMPT

    checks = {
        "selector_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "selector_sha256": hashlib.sha256(
            open("c:/SCP/score/boundary_action_selector.py", "rb").read()
        ).hexdigest(),
        "boundary_candidates_sha256": hashlib.sha256(
            open("c:/SCP/scripts/replay/boundary_candidates.py", "rb").read()
        ).hexdigest(),
    }
    drift = [name for name, value in checks.items() if manifest.get(name) != value]
    if drift:
        raise SystemExit(
            f"凍結後有變動，holdout 失效: {drift}\n"
            f"若變動是刻意的，應改用保留的 93 題，不得用本批重測。"
        )
    print("凍結驗證通過：prompt、selector、candidate generation 皆未變動")


if __name__ == "__main__":
    sys.exit(main())
