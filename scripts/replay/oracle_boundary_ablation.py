"""If boundaries were repaired perfectly, how much better would retrieval get?

Fragmentation and failure co-occur -- 19 of the 23 annotated tasks carrying a
cut span were answered wrong -- but co-occurrence is all that was shown. The
contrast group is six tasks, fragmentation is near-universal among the tasks
that search at all, and tasks that search are three times harder to begin with.
Nothing in that separates cause from correlation.

So this replaces the cut spans with the boundaries a human wrote, and measures
what retrieval does differently. It is an upper bound in two senses: the
boundaries are oracle, not something the selector produced, and a bound on
retrieval, not on answers. If Arm B does not move retrieval, boundary repair is
not worth further investment regardless of how well a selector could do it.

All 29 annotated tasks run, not the 19 failures. Nineteen say whether repair
helps, four say whether it breaks tasks that already worked, and six carry no
fragmented span at all and must come back byte-identical -- a control that fails
loudly if the injection touches anything it should not.

This is not an offline replay. Changing the query means searching again, and the
web answers differently from one minute to the next, so a paired design is the
only thing that makes the comparison mean anything: both arms run back to back
per task, never Arm B against a baseline recorded yesterday.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from dataclasses import replace
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, r"c:/SCP")

BASE = "c:/SCP/outputs/query_span_analysis"
OUT = "c:/SCP/outputs/oracle_boundary_ablation"

#: Copied from `EvidenceRunner._build_search_evidence`. Both arms must run the
#: retrieval the system actually runs; a budget that differs from production
#: would measure a pipeline nobody uses.
CONTROLLER_SETTINGS = dict(
    max_queries=3,
    max_results_per_query=5,
    max_pages_to_fetch=6,
    max_chunks_per_url=20,
    max_corpus_records=120,
    max_iter=5,
    top_k=16,
    min_retrieval_score=0.0,
    relative_score_margin=1.0,
    embedding_batch_size=8,
    bypass_labeler=False,
)

#: Fixed before the first run. Every repeat is reported.
REPEATS = 2

#: The order flips between repeats. Running A before B every time would give one
#: arm the earlier slot in every pair, and a search backend whose results drift
#: over minutes would turn that into a systematic difference between arms rather
#: than noise spread across both.
ARM_ORDER_BY_REPEAT = {1: ("A", "B"), 2: ("B", "A")}

#: Tasks with no fragmented span get one arm. Injection is keyed on span text
#: that these questions do not contain, so Arm B is Arm A by construction --
#: paying for it twice would buy a second sample of network noise, not a control.
#: The claim is checked by hashing the injection input instead.
CONTROL_ARM = "A"

#: Each run is its own process. Nothing in the retrieval path caches results
#: across arms today, but "today" is the wrong basis for an experiment whose
#: whole point is that the two arms differ only in their spans.
SUBPROCESS_ISOLATION = True


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


class GoldBoundaryInjector:
    """Swap the annotated fragments for their gold boundaries, and nothing else.

    Wrapping the repairer rather than editing it keeps Arm A on the exact code
    path production uses: Arm A never constructs this object. Spans the human
    marked `complete` pass through untouched by construction, which is what the
    six-task control checks from the outside.
    """

    def __init__(self, repairer, replacements: dict[str, str]) -> None:
        self._repairer = repairer
        self._replacements = replacements
        self.applied: list[tuple[str, str]] = []
        self.unmatched: list[str] = []

    def build_spans(self, question, tokens, scorer=None):
        spans = self._repairer.build_spans(question, tokens, scorer=scorer)
        out = []
        for span in spans:
            gold = self._replacements.get(normalise(span.text).casefold())
            at = question.casefold().find(gold.casefold()) if gold else -1
            if gold and at >= 0:
                out.append(
                    replace(
                        span,
                        text=question[at : at + len(gold)],
                        start=at,
                        end=at + len(gold),
                        original_text=span.text,
                        repair_source=f"{span.repair_source}+oracle_gold",
                    )
                )
                self.applied.append((span.text, gold))
            else:
                if gold:
                    self.unmatched.append(span.text)
                out.append(span)
        return out

    def __getattr__(self, name):
        return getattr(self._repairer, name)


def load_gold_by_task() -> dict[str, dict[str, str]]:
    """Human gold boundaries, keyed by task then by the span they replace."""

    from scripts.replay.boundary_recovery_prototype import load_gold

    merged = {
        row["annotation_id"]: row
        for row in csv.DictReader(
            open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8")
        )
    }
    out: dict[str, dict[str, str]] = {}
    for case in load_gold():
        task = merged[case.annotation_id]["task_id"]
        # An unaligned gold is one that could not be located verbatim in its own
        # context. Injecting it would put text in the query that the question
        # does not contain, which is a different experiment.
        if case.gold_repair_source == "unaligned" or not case.gold_span:
            continue
        out.setdefault(task, {})[normalise(case.span_text).casefold()] = case.gold_span
    return out


def populations() -> dict[str, str]:
    """Which of the three groups each task belongs to."""

    import glob

    rows = list(csv.DictReader(
        open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8")))
    fragmented = {r["task_id"] for r in rows if r["human_boundary"] == "fragmented"}
    correct = set()
    for path in glob.glob("c:/SCP/outputs/level1_final_22/tasks/*.json"):
        record = json.load(open(path, encoding="utf-8"))
        if record.get("exact_match"):
            correct.add(os.path.basename(path)[:3])
    out = {}
    for task in sorted({r["task_id"] for r in rows}):
        if task not in fragmented:
            out[task] = "control_no_fragment"
        elif task in correct:
            out[task] = "fragmented_correct"
        else:
            out[task] = "fragmented_wrong"
    return out


def questions() -> dict[str, str]:
    rows = csv.DictReader(open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8"))
    return {r["task_id"]: normalise(r["question"]) for r in rows}


def build_controller(replacements: dict[str, str] | None):
    from tools.search_result_builder.query import QueryGenerator
    from tools.search_result_builder.query.mask_salience_query import (
        MaskSalienceQueryGenerator,
    )
    from tools.search_result_builder.retrieval_control import WebRetrievalControl

    generator = MaskSalienceQueryGenerator(
        query_model_name=os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b")
    )
    injector = None
    if replacements:
        injector = GoldBoundaryInjector(generator.span_repairer, replacements)
        generator.span_repairer = injector
    controller = WebRetrievalControl(
        query_generator=QueryGenerator(generator=generator), **CONTROLLER_SETTINGS
    )
    return controller, injector


def run_arm(task: str, question: str, arm: str, repeat: int,
            replacements: dict[str, str] | None) -> dict:
    from pathlib import Path

    directory = Path(f"{OUT}/runs/{task}_{arm}_{repeat}")
    directory.mkdir(parents=True, exist_ok=True)
    controller, injector = build_controller(replacements if arm == "B" else None)
    started = datetime.now(timezone.utc).isoformat()
    try:
        output = controller.run(question, output_dir=directory)
        payload = output if isinstance(output, dict) else _as_dict(output)
        error = None
    except Exception as exc:
        import traceback
        payload, error = {}, "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-3000:]
    return {
        "task_id": task,
        "arm": arm,
        "repeat": repeat,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "injected": list(injector.applied) if injector else [],
        "injection_unmatched": list(injector.unmatched) if injector else [],
        "error": error,
        "payload": payload,
    }


def _as_dict(value):
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)[:2000]}


def _single(task: str, arm: str, repeat: int) -> None:
    """One run, in its own process, printing its record as JSON."""

    gold = load_gold_by_task()
    record = run_arm(task, questions()[task], arm, repeat, gold.get(task))
    print("__RECORD__" + json.dumps(record, ensure_ascii=False, default=str))


def injection_fingerprint(task: str, question: str, gold: dict | None) -> dict:
    """Evidence that Arm B would have been Arm A, for the tasks that skip it.

    A control that is simply not run proves nothing. This records what the
    injector would have been given, so the claim that nothing would have been
    replaced can be checked rather than asserted.
    """

    import hashlib

    replacements = gold or {}
    matched = [
        span for span in replacements
        if span in question.casefold()
    ]
    payload = json.dumps(
        {"question": question, "replacements": replacements}, sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "task_id": task,
        "replacement_count": len(replacements),
        "spans_present_in_question": matched,
        "would_inject": bool(matched),
        "input_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    import subprocess

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        return _single(sys.argv[2], sys.argv[3], int(sys.argv[4]))

    os.makedirs(f"{OUT}/runs", exist_ok=True)
    ledger = f"{OUT}/ablation_runs.jsonl"
    done = set()
    if os.path.exists(ledger):
        for line in open(ledger, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["task_id"], r["arm"], r["repeat"]))

    gold = load_gold_by_task()
    groups = populations()
    text = questions()

    # The control tasks never run Arm B, so their claim is recorded instead.
    fingerprints = [
        injection_fingerprint(task, text[task], gold.get(task))
        for task, group in sorted(groups.items())
        if group == "control_no_fragment"
    ]
    with open(f"{OUT}/control_injection_fingerprints.json", "w", encoding="utf-8") as handle:
        json.dump(fingerprints, handle, ensure_ascii=False, indent=1)
    leaking = [f["task_id"] for f in fingerprints if f["would_inject"]]
    if leaking:
        raise SystemExit(f"對照組不該有可注入的 span，但這些有: {leaking}")

    schedule: list[tuple[str, str, int]] = []
    for task in sorted(groups):
        for repeat in range(1, REPEATS + 1):
            if groups[task] == "control_no_fragment":
                schedule.append((task, CONTROL_ARM, repeat))
            else:
                for arm in ARM_ORDER_BY_REPEAT[repeat]:
                    schedule.append((task, arm, repeat))

    counts = Counter(groups.values())
    print(f"母體 {dict(counts)}")
    print(f"排程 {len(schedule)} run"
          f"（配對 {sum(1 for s in schedule if groups[s[0]] != 'control_no_fragment')}"
          f" + 對照 {sum(1 for s in schedule if groups[s[0]] == 'control_no_fragment')}）")
    print(f"repeat 1 順序 {ARM_ORDER_BY_REPEAT[1]}、repeat 2 順序 {ARM_ORDER_BY_REPEAT[2]}")
    print(f"對照組注入指紋已存，would_inject 全為 False\n")

    with open(ledger, "a", encoding="utf-8") as handle:
        for index, (task, arm, repeat) in enumerate(schedule, 1):
            if (task, arm, repeat) in done:
                continue
            # A separate process per run. Nothing carries over: not a model, not
            # a token cache, not whatever the search client keeps to itself.
            completed = subprocess.run(
                [sys.executable, __file__, "--single", task, arm, str(repeat)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            marker = "__RECORD__"
            line = next(
                (l for l in completed.stdout.splitlines() if l.startswith(marker)), ""
            )
            if line:
                record = json.loads(line[len(marker):])
            else:
                record = {
                    "task_id": task, "arm": arm, "repeat": repeat,
                    "error": f"subprocess exit {completed.returncode}: "
                             f"{completed.stderr[-1500:]}",
                    "payload": {}, "injected": [], "injection_unmatched": [],
                }
            record["population"] = groups[task]
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            flag = "ERR" if record.get("error") else "ok"
            print(f"   [{index}/{len(schedule)}] {task} {groups[task]:<20}"
                  f" arm{arm} r{repeat} {flag}  注入 {len(record.get('injected') or [])}")

    mark_incomplete_pairs(ledger, groups)


def mark_incomplete_pairs(ledger: str, groups: dict[str, str]) -> None:
    """A pair with one failed arm is not half a result.

    Keeping the surviving arm and comparing it to a different task's pair would
    be comparing across network conditions, which is the one thing the paired
    design exists to avoid. Both runs stay on disk; neither is scored.
    """

    records = [json.loads(l) for l in open(ledger, encoding="utf-8") if l.strip()]
    by_pair: dict[tuple[str, int], dict[str, dict]] = {}
    for record in records:
        if groups.get(record["task_id"]) == "control_no_fragment":
            continue
        by_pair.setdefault((record["task_id"], record["repeat"]), {})[record["arm"]] = record

    incomplete = [
        key for key, arms in by_pair.items()
        if set(arms) != {"A", "B"} or any(a.get("error") for a in arms.values())
    ]
    with open(f"{OUT}/incomplete_pairs.json", "w", encoding="utf-8") as handle:
        json.dump(
            [{"task_id": t, "repeat": r} for t, r in sorted(incomplete)],
            handle, ensure_ascii=False, indent=1,
        )
    print(f"\n完整配對 {len(by_pair) - len(incomplete)}/{len(by_pair)}"
          f"、incomplete {len(incomplete)}: {sorted(incomplete)}")


if __name__ == "__main__":
    sys.exit(main())
