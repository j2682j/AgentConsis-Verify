"""Join the human labels onto canonical text, and flag what was annotated corrupt.

The annotated CSV came back in cp1252 with 17 rows altered by the editor that
saved it: smart quotes broken, `Taishō` reduced to `Taish?`, and six date-like
spans silently reformatted by a spreadsheet -- `June 2014` became `Jun-14`. The
labels themselves are intact, but the text an annotator was shown is not
necessarily the text the classifier saw.

So the merge takes only `annotation_id` and the five label columns from the
annotated file, and everything else from a canonical rebuild out of the task
JSON. Nothing else from the corrupted file is trusted; `text_altered_when_
annotated` marks the rows whose judgement rested on changed input, so they can
be re-checked before their labels are used as reference.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import subprocess
import sys

LABEL_FIELDS = (
    "human_role",
    "human_boundary",
    "should_enter_query",
    "annotation_confidence",
    "notes",
)
COMPARED = (
    "question",
    "question_head_span",
    "answer_role",
    "answer_target",
    "span_text",
    "original_text",
    "local_context",
    "enclosing_unit",
)


def read_annotated(path: str) -> dict[str, dict[str, str]]:
    """cp1252, because that is what the editor wrote back."""

    raw = open(path, "rb").read().decode("cp1252")
    return {row["annotation_id"]: row for row in csv.DictReader(io.StringIO(raw))}


def canonical(out_dir: str) -> dict[str, dict[str, str]]:
    subprocess.run(
        [sys.executable, "-m", "scripts.replay.build_span_annotation_set",
         "--out-dir", out_dir],
        capture_output=True,
        cwd="c:/SCP",
        check=True,
    )
    path = f"{out_dir}/query_span_annotation_blind.csv"
    return {r["annotation_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def main() -> None:
    base = "c:/SCP/outputs/query_span_analysis"
    annotated = read_annotated(f"{base}/query_span_annotation_blind.csv")
    canon = canonical(f"{base}/_canonical")
    if sorted(annotated) != sorted(canon):
        raise SystemExit("annotation_id sets differ; refusing to merge")

    rows = []
    for annotation_id in sorted(canon):
        clean, dirty = canon[annotation_id], annotated[annotation_id]
        altered = [
            field
            for field in COMPARED
            if (clean.get(field) or "").strip() != (dirty.get(field) or "").strip()
        ]
        row = {key: clean[key] for key in canon[annotation_id]}
        for field in LABEL_FIELDS:
            row[field] = (dirty.get(field) or "").strip()
        row["text_altered_when_annotated"] = "yes" if altered else "no"
        row["altered_fields"] = ";".join(altered)
        rows.append(row)

    out = f"{base}/query_span_annotation_merged.csv"
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    recheck = [r for r in rows if r["text_altered_when_annotated"] == "yes"]
    review = f"{base}/query_span_recheck_blind.csv"
    with open(review, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["recheck_id", "question", "question_head_span", "answer_role",
             "answer_target", "span_text", "original_text", "local_context",
             "enclosing_unit", "human_role", "human_boundary",
             "should_enter_query", "annotation_confidence", "notes"]
        )
        # Shuffled with a fixed seed, not sorted. Ordering by `span_text`
        # groups similar spans together, which is itself a hint: an annotator
        # who sees three date-like spans in a row reads them as a set rather
        # than judging each on its own.
        ordered = list(recheck)
        random.Random(20260818).shuffle(ordered)
        mapping = {}
        for index, row in enumerate(ordered, start=1):
            recheck_id = f"R{index:02d}"
            mapping[recheck_id] = row["annotation_id"]
            writer.writerow(
                [recheck_id, row["question"], row["question_head_span"],
                 row["answer_role"], row["answer_target"], row["span_text"],
                 row["original_text"], row["local_context"],
                 row["enclosing_unit"], "", "", "", "", ""]
            )
    with open(f"{base}/_recheck_mapping.json", "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)

    print(f"合併 {len(rows)} 列 → {out}")
    print(f"需覆核 {len(recheck)} 列 → {review}")
    print(f"mapping（私有）→ {base}/_recheck_mapping.json")


if __name__ == "__main__":
    main()
