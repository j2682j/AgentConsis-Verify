"""What the paired oracle-boundary runs actually show.

Read in three layers, in the order the manifest fixed before the runs started.
Whether the query changed at all, whether retrieval behaved differently, and
whether the gold answer travelled further down the funnel. Only the third is a
result; the first two explain it.

The funnel was pre-registered with three levels and only two of them were run.
Evidence conversion is reproduced offline here from the saved payloads, using
the production converter, so `gold_in_evidence` is the real metric rather than a
stand-in. Stage 1 context was never assembled -- the experiment deliberately
stops before the agent -- so that level is reported as not computed rather than
quietly replaced with something adjacent.

Both denominators are reported throughout. If strict and lenient agree, the
eligibility argument was moot; if they disagree, that is the finding.

With eighteen paired tasks, McNemar needs roughly six one-directional
transitions to reach p<0.05. The p-value is printed because it was promised, and
it is not the result. The result is which tasks moved, in which direction, and
whether the two repeats agree.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from math import comb

sys.path.insert(0, r"c:/SCP")

from scripts.replay.eligibility_frozen import LENIENT, STRICT

OUT = "c:/SCP/outputs/oracle_boundary_ablation"
LEDGER = f"{OUT}/ablation_runs.jsonl"

TRACKING = re.compile(r"^(utm_|fbclid|gclid|ref|source)", re.IGNORECASE)


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def canonical_url(url: str) -> str:
    """Same page, same string, so a Jaccard measures pages and not spellings."""

    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query) if not TRACKING.match(k)
    ])
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    for scheme, port in (("http", ":80"), ("https", ":443")):
        if parts.scheme.lower() == scheme and netloc.endswith(port):
            netloc = netloc[: -len(port)]
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def gold_answers() -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(
        "c:/SCP/data/gaia/2023/validation/metadata.level1.parquet"
    ).reset_index(drop=True)
    return {
        f"{i + 1:03d}": normalise(frame.loc[i, "Final answer"]) for i in range(len(frame))
    }


def components(gold: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*[,;]\s*", gold) if p.strip()]
    return parts if len(parts) > 1 else [gold]


def contains_gold(text: str, gold: str) -> bool:
    """Word-bounded, and a list answer needs every part of it."""

    folded = text.casefold()
    for part in components(gold):
        pattern = re.compile(
            rf"(?<![\w]){re.escape(part.casefold())}(?![\w])"
        )
        if not pattern.search(folded):
            return False
    return True


def corpus_text_and_urls(payload: dict) -> tuple[str, set[str]]:
    path = payload.get("corpus_path") or ""
    if not os.path.exists(path):
        return "", set()
    texts, urls = [], set()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        texts.append(str(record.get("text") or ""))
        canonical = canonical_url(record.get("url") or "")
        if canonical:
            urls.add(canonical)
    return "\n".join(texts), urls


def evidence_text(payload: dict) -> str | None:
    """Rerun the production evidence conversion over the saved retrieval output."""

    try:
        from core.evidence_runner import EvidenceRunner
        from tools.search_result_builder.evidence.evidence_converter import (
            EvidenceConverter,
        )
        from tools.search_result_builder.evidence.span_builder import SpanBuilder

        from tools.evidence.fact_extraction.fact_store import TaskFactStore

        runner = EvidenceRunner.__new__(EvidenceRunner)
        runner.question = payload.get("question", "")
        runner.span_builder = SpanBuilder()
        runner.evidence_converter = EvidenceConverter(span_builder=runner.span_builder)
        runner.fact_store = TaskFactStore()
        contract = runner._evidence_selection_contract(payload)
        items = runner._web_retrieval_evidence_items(payload, contract=contract)
        # An empty bundle is a result -- the converter kept nothing -- and has to
        # stay distinguishable from a conversion that never ran. Returning "" for
        # the first and None for the second is the whole difference between
        # "evidence delivered nothing" and "the metric was not computed".
        return "\n".join(json.dumps(i, ensure_ascii=False, default=str) for i in items)
    except Exception as exc:
        evidence_text.failure = f"{type(exc).__name__}: {exc}"
        return None


def mcnemar(improved: int, regressed: int) -> float:
    n = improved + regressed
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(improved, regressed) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def main() -> None:
    records = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    gold = gold_answers()
    by_run: dict[tuple[str, str, int], dict] = {
        (r["task_id"], r["arm"], r["repeat"]): r for r in records
    }
    paired_tasks = sorted({
        r["task_id"] for r in records if r["population"] != "control_no_fragment"
    })

    print(f"{len(records)} run、配對 task {len(paired_tasks)}、"
          f"對照 {len({r['task_id'] for r in records if r['population'] == 'control_no_fragment'})}")

    # --- Layer 1: did the intervention change anything at all
    print(f"\n=== 第一層　Boundary intervention")
    changed_query, injected_total = 0, 0
    for task in paired_tasks:
        for repeat in (1, 2):
            a, b = by_run.get((task, "A", repeat)), by_run.get((task, "B", repeat))
            if not a or not b:
                continue
            qa = tuple(a["payload"].get("generated_queries") or [])
            qb = tuple(b["payload"].get("generated_queries") or [])
            changed_query += qa != qb
            injected_total += len(b.get("injected") or [])
    pairs = sum(1 for t in paired_tasks for r in (1, 2)
                if by_run.get((t, "A", r)) and by_run.get((t, "B", r)))
    print(f"   query 改變 {changed_query}/{pairs} 配對"
          f" = {changed_query / max(pairs, 1):.3f}")
    print(f"   注入的 gold boundary 共 {injected_total} 次")

    # --- Layer 2: mechanism, reported as description and never as success
    print(f"\n=== 第二層　Retrieval mechanism（機制性，非成效）")
    deltas: dict[str, list[float]] = defaultdict(list)
    jaccard: list[float] = []
    for task in paired_tasks:
        for repeat in (1, 2):
            a, b = by_run.get((task, "A", repeat)), by_run.get((task, "B", repeat))
            if not a or not b:
                continue
            _, ua = corpus_text_and_urls(a["payload"])
            _, ub = corpus_text_and_urls(b["payload"])
            if ua or ub:
                jaccard.append(len(ua & ub) / max(len(ua | ub), 1))
            for field, where in (("unique_document_count", "retrieval"),
                                 ("source_count", "diagnostics"),
                                 ("filtered_source_count", "diagnostics"),
                                 ("fetched_page_count", "diagnostics")):
                va = (a["payload"].get(where) or {}).get(field)
                vb = (b["payload"].get(where) or {}).get(field)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    deltas[field].append(vb - va)
            deltas["corpus_urls"].append(len(ub) - len(ua))
    for field, values in deltas.items():
        if values:
            better = sum(1 for v in values if v > 0)
            worse = sum(1 for v in values if v < 0)
            print(f"   {field:<24} B−A 平均 {sum(values)/len(values):+.1f}"
                  f"　增 {better}／減 {worse}／持平 {len(values)-better-worse}")
    if jaccard:
        print(f"   corpus_url_jaccard        平均 {sum(jaccard)/len(jaccard):.3f}"
              f"（高低皆不代表品質）")

    # --- Layer 3: the result
    print(f"\n=== 第三層　Paired outcome（主要結果）")
    for level, extract in (
        ("gold_in_retrieved_documents", lambda p: corpus_text_and_urls(p)[0]),
        ("gold_in_evidence", evidence_text),
    ):
        print(f"\n   [{level}]")
        for name, denominator in (("strict", STRICT), ("lenient", LENIENT)):
            eligible = [t for t in paired_tasks if t in denominator]
            transitions: dict[str, list[str]] = defaultdict(list)
            unavailable = 0
            for task in eligible:
                votes = []
                for repeat in (1, 2):
                    a, b = by_run.get((task, "A", repeat)), by_run.get((task, "B", repeat))
                    if not a or not b:
                        continue
                    ta, tb = extract(a["payload"]), extract(b["payload"])
                    if ta is None or tb is None:
                        continue
                    votes.append((contains_gold(ta, gold[task]),
                                  contains_gold(tb, gold[task])))
                if not votes:
                    unavailable += 1
                    continue
                gained = sum(1 for x, y in votes if y and not x)
                lost = sum(1 for x, y in votes if x and not y)
                if gained and not lost:
                    transitions["improved"].append(task)
                elif lost and not gained:
                    transitions["regressed"].append(task)
                elif gained and lost:
                    transitions["mixed"].append(task)
                elif all(x and y for x, y in votes):
                    transitions["unchanged_both_hit"].append(task)
                elif not any(x or y for x, y in votes):
                    # The distinction the headline number hides. A task where
                    # neither arm ever found the gold is not evidence that the
                    # intervention did nothing; it is a task where retrieval
                    # failed in both arms and the comparison has no purchase.
                    transitions["unchanged_both_miss"].append(task)
                else:
                    transitions["unchanged_mixed_repeats"].append(task)
            improved, regressed = len(transitions["improved"]), len(transitions["regressed"])
            print(f"      {name:<8} n={len(eligible)}"
                  f"  改善 {improved}"
                  f"　退步 {regressed}"
                  f"　混合 {len(transitions['mixed'])}")
            print(f"               不變：兩臂皆命中 {len(transitions['unchanged_both_hit'])}"
                  f"、兩臂皆未命中 {len(transitions['unchanged_both_miss'])}"
                  f"、repeat 不一致 {len(transitions['unchanged_mixed_repeats'])}"
                  f"、無資料 {unavailable}")
            if improved or regressed:
                print(f"               改善 {transitions['improved']}"
                      f"　退步 {transitions['regressed']}")
            print(f"               McNemar exact p = {mcnemar(improved, regressed):.3f}"
                  f"（輔助，非結論）")

    print(f"\n   [gold_in_stage1_context] 未計算 —— 實驗停在 Stage1 之前，"
          f"未組裝 context。不以鄰近指標替代。")

    # --- Control
    controls = sorted({
        r["task_id"] for r in records if r["population"] == "control_no_fragment"
    })
    print(f"\n=== 對照組 {len(controls)} 題（單臂 × 2）")
    fingerprints = json.load(
        open(f"{OUT}/control_injection_fingerprints.json", encoding="utf-8"))
    print(f"   would_inject 為 True 的: "
          f"{[f['task_id'] for f in fingerprints if f['would_inject']] or '無'}")
    for task in controls:
        queries = [
            tuple(by_run[(task, "A", r)]["payload"].get("generated_queries") or [])
            for r in (1, 2) if (task, "A", r) in by_run
        ]
        same = len(set(queries)) <= 1
        print(f"   {task}: 兩次 repeat query {'相同' if same else '不同'}")


if __name__ == "__main__":
    sys.exit(main())
