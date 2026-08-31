"""Extract query spans from tasks that took no part in designing any of this.

Everything measured so far -- the generators, the cap, the prompt, twice -- was
measured on 133 spans from 29 Level 1 validation tasks, and every one of those
numbers was looked at while the code was being changed. The 38 fragmented cases
in particular were read one at a time during generator debugging. They cannot
say whether this works; they can only say it works on them.

So this builds a second set from the 24 Level 1 validation tasks that were never
touched. Same split, same distribution, disjoint tasks -- a task-level holdout
rather than a span-level one, because spans from the same question share a
context and splitting inside a task would leak it.

Only span extraction runs. There is no search, no agent, no query model call:
salience scoring, span repair and role classification, which is everything the
boundary problem lives in. The 93 Level 1 test questions stay in reserve for a
second pass if this one is spent.

The point of a holdout is that it is spent once. This writes a manifest first,
hashing the generator source, the frozen lattice settings and the selector
prompt, so that what was frozen before the labels existed can be checked
afterwards rather than remembered.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, r"c:/SCP")

BASE = "c:/SCP/outputs/query_span_analysis"
OUT = f"{BASE}/holdout"

#: The 29 that produced the design set, as row indices into the Level 1
#: validation parquet. Recorded rather than recomputed: the design set is fixed
#: now, and a later change to the annotation file must not silently move the
#: boundary between development and holdout.
DESIGN_TASK_INDICES = frozenset(
    {0, 1, 3, 4, 5, 6, 12, 13, 15, 17, 18, 19, 26, 28, 29, 31, 32, 33,
     38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52}
)

#: The design set had no such column on its controls, and that is why 12 spans
#: the model widened -- `1928` to `1928 Summer Olympics`, `NASA` to `NASA award
#: number` -- could not be told apart from damage. The rule is deliberately
#: narrow: an alternative must mean the same search, not merely be defensible.
#: `a dinosaur` and `a dinosaur that was promoted in November 2016` are not
#: equivalent, because the second adds a constraint the first does not carry.
#:
#: Blank is the default and blank is safe. Filling this column to accommodate
#: something a selector produced would turn the holdout into a way of scoring
#: the selector against itself.
ACCEPTABLE_ALTERNATIVE_RULE = (
    "原文 substring，且與 gold 的搜尋語意、指涉對象、限制條件完全相同，"
    "僅邊界略異；無真正等價形式則留白"
)

HUMAN_BOUNDARY_OPTIONS = "complete | fragmented | overexpanded | unrelated | unclear"
UNIT_TYPES = "entity | title | date | noun_phrase | clause | other"
BOUNDARY_FORMS = "plain | parenthetical | quoted | punctuation_delimited | other"
REPAIR_DIRECTIONS = "keep | left | right | both | replace"


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def _sha256(data: bytes) -> str:
    """Full digest, not a prefix. A truncated hash is fine for reading aloud and
    useless for settling an argument about whether a file changed."""

    return hashlib.sha256(data).hexdigest()


def environment() -> dict:
    """The parts of the machine that change an answer.

    Code hashes alone do not pin a result: the same prompt against a different
    model build, a different quantisation, or a different decoding temperature
    is a different system. A holdout that cannot name which one it tested
    cannot be re-run to check a disputed number.
    """

    import json as _json
    import platform
    import urllib.request

    import spacy

    def ollama(path: str, payload: dict | None = None) -> dict:
        body = _json.dumps(payload).encode() if payload else None
        request = urllib.request.Request(
            f"http://localhost:11434{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
        )
        return _json.loads(urllib.request.urlopen(request, timeout=30).read())

    model = os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b")
    try:
        version = ollama("/api/version").get("version", "")
        tags = ollama("/api/tags").get("models", [])
        entry = next((m for m in tags if m.get("name") == model), {})
        details = ollama("/api/show", {"model": model}).get("details", {})
    except Exception as exc:  # recorded, not silently omitted
        version, entry, details = f"unavailable: {exc}", {}, {}

    packages = {}
    for name in ("torch", "transformers", "numpy", "pandas", "openai", "faiss"):
        try:
            packages[name] = __import__(name).__version__
        except Exception:
            packages[name] = "absent"

    nlp_meta = spacy.load("en_core_web_md").meta
    return {
        "selector_model": model,
        "selector_model_digest": entry.get("digest", ""),
        "selector_model_quantisation": details.get("quantization_level", ""),
        "selector_model_parameters": details.get("parameter_size", ""),
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "provider_version": version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "spacy": spacy.__version__,
        "spacy_model": f"{nlp_meta['lang']}_{nlp_meta['name']}",
        "spacy_model_version": nlp_meta["version"],
        "packages": packages,
    }


#: Declared before the labels exist, because "how many runs" is a decision that
#: can be made after the fact to suit the answer. One run, because one run is
#: what deployment does -- aggregating three would measure a system nobody is
#: going to ship. The determinism probe exists to show the seed is doing its
#: job, not to improve the score.
RUN_PROTOCOL = {
    #: What this set is, stated so a later write-up cannot overstate it. The
    #: tasks took no part in any design decision and no label existed when the
    #: configuration was corrected -- but raw decisions, parse behaviour and the
    #: gap between two decode paths have all been inspected on these spans. That
    #: is a protocol amendment, not tuning against holdout labels, and it is
    #: still not an untouched test set.
    "designation": "label-blind amended holdout",
    "not": "untouched test set",
    #: The reserved split is not pristine either, and saying so costs nothing.
    #: It was read once already, to count how often three unrelated rules fired
    #: -- the literal answer contract, the full-form directive, and `all of the`
    #: -- all of which returned zero. No boundary label was revealed, no
    #: boundary decision was inspected, and nothing in the selector or the
    #: candidate generator was tuned from it, so it remains usable as a
    #: component-level confirmatory set. It is not a globally untouched test
    #: set, and a write-up that calls it one would be overclaiming.
    "confirmatory_holdout": "93-task Level 1 test split",
    "confirmatory_designation": "component-level boundary-selector confirmatory set",
    "confirmatory_prior_exposure": [
        "Used for unrelated rule-trigger frequency audits",
        "No boundary annotations were revealed",
        "No boundary selector decisions were inspected",
        "No boundary candidate or selector parameters were tuned from this split",
    ],
    "confirmatory_not": "globally untouched test set",
    "runs": 1,
    "aggregation": "none — the single run is the result",
    "determinism_probe": "re-run 10 spans; identical parsed decision required",
    "on_probe_failure": "report as non-deterministic; do not average",
    "scored_once": True,
}

#: Changes made after the first freeze, recorded rather than folded in.
#:
#: A freeze that gets quietly rewritten is not a freeze, so what moved and why
#: is written down. What makes these admissible is the ordering, not the reason:
#: no label existed when they were made, and the decision run carries no score
#: until one does, so nothing here could have been chosen to suit an outcome.
#: Anything discovered after the labels are revealed goes to the reserved 93
#: test questions instead.
AMENDMENTS = (
    {
        "date": "2026-08-19",
        "trigger": "determinism probe 失敗：10 span 中 5 筆兩次結果不同",
        "found": "DECODE_SETTINGS 未實際生效。OpenAI-compat 路徑接受 seed 卻未"
                 "映射到 Ollama 的 options.seed；enable_thinking 僅在 provider"
                 "為 vllm 時套用，對 ollama 靜默忽略。manifest 所描述的系統與"
                 "實際執行的系統不同。",
        "fix": "selector 改走 Ollama native endpoint（call_model），seed 與 think"
               "在該路徑上確實生效；probe 改以解析後的決策比對，因為同一決策會"
               "時而 pretty-print 時而 compact，逐位元組比較會把空白差異報成不"
               "決定性。",
        "labels_existed": False,
        "discarded": "第一次 decision run（105 筆）另存為 _preamendment，不計分",
    },
    {
        "date": "2026-08-19",
        "trigger": "修正解碼路徑後，設計集同 prompt 的 fragmented exact 由 "
                   "14/37 掉到 5/37；兩條路徑的決策有 42% 不同",
        "found": "enable_thinking=False 是未經測試就寫入凍結設定的選擇，且先前"
                 "被 compat 路徑忽略，所以 v3/v4 的好數字其實是 thinking 開啟"
                 "下取得的。在 native 路徑上直接 A/B（同 seed、同 prompt）："
                 "fragmented exact 6/37 對 15/37。",
        "fix": "DECODE_SETTINGS.enable_thinking 改為 True。代價一併記錄："
               "complete 誤動由 5/40 升至 8/40，亦即推高了決策規則據以拒絕部署"
               "的那項指標。選擇 True 是因為 False 的 0.16 修復率不值得部署，"
               "不是因為它數字較好看。",
        "labels_existed": False,
        "discarded": "第二次 decision run（think=False，105 筆）另存為 "
                     "_thinkoff，不計分",
    },
)


def freeze_manifest() -> dict:
    """Hash what must not move once the labels are revealed."""

    from score.boundary_action_selector import (
        DECODE_SETTINGS,
        DEFER_CLASSES,
        RESPONSE_SCHEMA,
        SYSTEM_PROMPT,
    )
    from scripts.replay.boundary_candidate_oracle import GENERATORS, MAX_EXPANSION_TOKENS

    def digest(path: str) -> str:
        return _sha256(open(path, "rb").read())

    return {
        "selector_prompt_sha256": _sha256(SYSTEM_PROMPT.encode("utf-8")),
        "boundary_candidates_sha256": digest("c:/SCP/scripts/replay/boundary_candidates.py"),
        "boundary_oracle_sha256": digest("c:/SCP/scripts/replay/boundary_candidate_oracle.py"),
        "selector_sha256": digest("c:/SCP/score/boundary_action_selector.py"),
        "decode_settings": dict(DECODE_SETTINGS),
        "response_schema": RESPONSE_SCHEMA,
        "defer_classes": {k: list(v) for k, v in DEFER_CLASSES.items()},
        "generators": list(GENERATORS),
        "max_expansion_tokens": MAX_EXPANSION_TOKENS,
        "design_task_indices": sorted(DESIGN_TASK_INDICES),
        "run_protocol": RUN_PROTOCOL,
        "amendments": [dict(a) for a in AMENDMENTS],
        "environment": environment(),
    }


def holdout_questions() -> list[tuple[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(
        "c:/SCP/data/gaia/2023/validation/metadata.level1.parquet"
    ).reset_index(drop=True)
    return [
        (f"{index + 1:03d}", normalise(frame.loc[index, "Question"]))
        for index in range(len(frame))
        if index not in DESIGN_TASK_INDICES
    ]


def extract() -> list[dict]:
    """Salience -> span repair -> role classification, and nothing after it."""

    from tools.search_result_builder.query.mask_salience_query import (
        MaskSalienceQueryGenerator,
    )

    generator = MaskSalienceQueryGenerator()
    rows: list[dict] = []
    for task_id, question in holdout_questions():
        tokens = generator.score_tokens(question)
        kept = generator.filter_tokens(tokens)
        spans = generator.span_repairer.build_spans(
            question, kept, scorer=generator.semantic_scorer
        )
        role = generator.question_role_extractor.extract(question)
        classified = generator.classify_spans(question, spans, question_role=role)
        for span in classified:
            rows.append(
                {
                    "task_id": task_id,
                    "question": question,
                    "span_text": normalise(span.text),
                    "original_text": normalise(span.original_text),
                    "local_context": normalise(span.context) or question,
                    "question_head_span": normalise(
                        (span.question_role or {}).get("head_span")
                    ) or "unavailable",
                    "answer_role": normalise(
                        (span.question_role or {}).get("answer_role")
                    ) or "unavailable",
                    "answer_target": normalise(
                        (span.question_role or {}).get("answer_target")
                    ) or "unavailable",
                    "predicted_role": span.role,
                    "classification_status": span.classification_status,
                    "repair_source": span.repair_source,
                    "confidence": round(float(span.confidence), 6),
                }
            )
        print(f"   {task_id}: {len(classified)} spans")
    return rows


def freeze_candidates(unique: list[dict]) -> None:
    """The boundaries the selector will be allowed to name, fixed in advance."""

    from scripts.replay.boundary_candidate_oracle import candidates_for
    from scripts.replay.boundary_recovery_prototype import Recovery

    recovery = Recovery()
    entries = []
    for row in unique:
        context, span = row["local_context"], row["span_text"]
        at = context.casefold().find(span.casefold())
        candidates = (
            candidates_for(recovery.nlp(context), context, span) if at >= 0 else []
        )
        entries.append(
            {
                "annotation_id": row["annotation_id"],
                "task_id": row["task_id"],
                "context": context,
                "span": [at, at + len(span)] if at >= 0 else None,
                "span_text": span,
                "candidates": [[c.start, c.end, list(c.generators)] for c in candidates],
            }
        )
    with open(f"{OUT}/boundary_holdout_candidates.json", "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "entries": entries}, handle,
                  ensure_ascii=False, indent=1)

    sizes = sorted(len(e["candidates"]) for e in entries)
    ungrounded = sum(1 for e in entries if e["span"] is None)
    print(f"\n候選 lattice: {len(entries)} span、"
          f"中位數 {sizes[len(sizes)//2]}、P95 {sizes[int(len(sizes)*0.95)-1]}、"
          f"總計 {sum(sizes)}")
    if ungrounded:
        print(f"   span 無法在 local_context 中定位: {ungrounded}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest = freeze_manifest()

    rows = extract()
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["task_id"], row["span_text"].casefold())
        if row["span_text"] and key not in seen:
            seen.add(key)
            unique.append(row)
    for index, row in enumerate(unique, 1):
        row["annotation_id"] = f"H{index:03d}"

    # Extraction is deterministic or the holdout is not a fixed set. Comparing
    # against the previous canonical copy is free and says so out loud, rather
    # than assuming it and discovering otherwise when a number moves.
    previous = f"{OUT}/_canonical_holdout.csv"
    if os.path.exists(previous):
        before = [
            (r["task_id"], r["span_text"])
            for r in csv.DictReader(open(previous, encoding="utf-8"))
        ]
        now = [(r["task_id"], r["span_text"]) for r in unique]
        print(f"\n與前次抽取一致: {before == now}"
              f"（前次 {len(before)} span、本次 {len(now)}）")

    manifest.update(
        {
            "holdout_tasks": len({r["task_id"] for r in unique}),
            "spans_extracted": len(rows),
            "spans_unique": len(unique),
        }
    )

    # Two files, for the same reason as the design set: the blind copy carries
    # nothing the classifier concluded, because seeing `source_clue` beside a
    # span is enough to anchor an annotator onto it.
    blind_columns = [
        "annotation_id", "task_id", "question", "question_head_span",
        "answer_role", "answer_target", "span_text", "original_text",
        "local_context", "human_boundary", "human_gold_span",
        "repair_direction", "unit_type", "boundary_form",
        "boundary_confidence", "acceptable_alternative", "notes",
    ]
    with open(f"{OUT}/boundary_holdout_blind.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(blind_columns)
        writer.writerow(
            ["", "", "", "", "", "", "", "", "",
             HUMAN_BOUNDARY_OPTIONS, "從 local_context 逐字複製；complete 留空",
             REPAIR_DIRECTIONS, UNIT_TYPES, BOUNDARY_FORMS, "high | medium | low",
             ACCEPTABLE_ALTERNATIVE_RULE, ""]
        )
        for row in unique:
            writer.writerow(
                [row["annotation_id"], row["task_id"], row["question"],
                 row["question_head_span"], row["answer_role"], row["answer_target"],
                 row["span_text"], row["original_text"], row["local_context"],
                 "", "", "", "", "", "", "", ""]
            )

    with open(f"{OUT}/boundary_holdout_predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["annotation_id", "task_id", "predicted_role", "classification_status",
             "repair_source", "confidence"]
        )
        for row in unique:
            writer.writerow(
                [row["annotation_id"], row["task_id"], row["predicted_role"],
                 row["classification_status"], row["repair_source"], row["confidence"]]
            )

    # The canonical copy exists because the design set lost 17 rows to an editor
    # that rewrote non-ASCII characters on save. Labels get merged back by
    # `annotation_id`; every other column comes from here.
    with open(f"{OUT}/_canonical_holdout.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unique[0]))
        writer.writeheader()
        writer.writerows(unique)

    # The lattice is frozen now, before any label exists, for the same reason
    # the prompt is: it is the set of boundaries the selector is permitted to
    # name, so building it after seeing the gold would let the permitted set be
    # shaped by the answer. It needs only the span and its context.
    freeze_candidates(unique)

    # The data is hashed too, and last, so the manifest covers the exact bytes
    # the annotator will be working from. An editor that rewrites the file on
    # save -- which is how the design set lost 17 rows -- shows up here.
    manifest["data_sha256"] = {
        name: _sha256(open(f"{OUT}/{name}", "rb").read())
        for name in ("_canonical_holdout.csv", "boundary_holdout_blind.csv",
                     "boundary_holdout_predictions.csv",
                     "boundary_holdout_candidates.json")
    }
    manifest["report_sha256"] = _sha256(
        open("c:/SCP/scripts/replay/boundary_holdout_report.py", "rb").read()
    )
    with open(f"{OUT}/holdout_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)

    print(f"\nholdout {manifest['holdout_tasks']} tasks、"
          f"{len(rows)} spans -> 去重 {len(unique)}")
    print(f"   -> {OUT}/boundary_holdout_blind.csv")
    print(f"   -> {OUT}/holdout_manifest.json")
    env = manifest["environment"]
    print(f"\n凍結（完整 SHA-256 見 manifest，此處顯示前 12 碼）")
    for key in ("selector_prompt_sha256", "boundary_candidates_sha256",
                "boundary_oracle_sha256", "selector_sha256"):
        print(f"   {key:<28} {manifest[key][:12]}")
    for name, value in manifest["data_sha256"].items():
        print(f"   {name:<28} {value[:12]}")
    print(f"   model                        {env['selector_model']} "
          f"{env['selector_model_digest'][:12]} {env['selector_model_quantisation']}")
    print(f"   provider                     {env['provider']} {env['provider_version']}")
    print(f"   decode                       {manifest['decode_settings']}")
    print(f"   spacy                        {env['spacy']} / "
          f"{env['spacy_model']} {env['spacy_model_version']}")
    print(f"   run protocol                 {manifest['run_protocol']['runs']} run、"
          f"{manifest['run_protocol']['aggregation']}")


if __name__ == "__main__":
    sys.exit(main())
