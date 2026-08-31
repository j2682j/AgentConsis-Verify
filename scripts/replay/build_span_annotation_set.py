"""Build a blind annotation set for the query span roles, one row per decision.

The 784 recorded spans are 133 decisions repeated across six runs: the same span
in the same task classifies identically every time, with `in_generated_query` the
only field that ever varies (7 cases). Annotating the 784 would weight a handful
of judgements by however often their task happened to be re-run.

Two files, deliberately separate. The blind set carries the question, the span
and its context and nothing the model concluded -- seeing `source_clue` and a
margin next to a span is enough to anchor a human onto it, and the whole point
of annotating is to obtain a label the classifier did not produce. The
predictions file keeps the model's side, joined back by `annotation_id` once the
labels exist.

Analysis afterwards needs both weightings. Unique-case metrics say how good the
classifier is; occurrence-weighted metrics say what it costs the system, and for
`in_generated_query` the occurrence level is the only correct one because the
same span reaches the query in some runs and not others.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

ANALYSIS = "c:/SCP/outputs/query_span_analysis/query_span_role_analysis.json"

HUMAN_ROLE_OPTIONS = (
    "source_clue | constraint | answer_target | format_instruction | other | ambiguous"
)
HUMAN_BOUNDARY_OPTIONS = "complete | fragmented | overexpanded | unrelated | unclear"
SHOULD_ENTER_QUERY_OPTIONS = "yes | no | conditional"


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def dedup_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["task_id"],
        normalise(row["text"]).casefold(),
        normalise(row["original_text"]).casefold(),
        normalise(row["enclosing_unit"]).casefold(),
    )


def question_of(run: str, task_id: str) -> str:
    for path in glob.glob(f"c:/SCP/outputs/{run}/tasks/{task_id}_*.json"):
        return normalise(json.loads(open(path, encoding="utf-8").read()).get("question"))
    return ""


def span_record(run: str, task_id: str, text_key: str) -> dict[str, Any]:
    """The recorded span itself, for fields the analysis JSON does not carry."""

    for path in glob.glob(f"c:/SCP/outputs/{run}/tasks/{task_id}_*.json"):
        task = json.loads(open(path, encoding="utf-8").read())
        meta = (task.get("network_summary") or {}).get("metadata") or {}
        raw = next(
            (
                item.get("raw_result")
                for item in (meta.get("tool_usage") or [])
                if isinstance(item, dict)
                and item.get("tool_name") == "search"
                and isinstance(item.get("raw_result"), dict)
            ),
            {},
        )
        for span in ((raw.get("diagnostics") or {}).get("query_plan") or {}).get(
            "classified_spans"
        ) or []:
            if isinstance(span, dict) and normalise(span.get("text")).casefold() == text_key:
                return span
    return {}


def question_role_fields(span: dict[str, Any]) -> tuple[str, str, str]:
    """What the question was read as asking for, decided before classification.

    Carried into the blind set because the roles cannot be told apart without
    it: whether `title of` is the answer target or a source clue depends on what
    the question wants back. It is the classifier's own input, not the gold
    answer, so it leaks nothing. Missing values say `unavailable` rather than
    empty, so a blank column cannot be read as "the question asked for nothing".
    """

    role = span.get("question_role") or {}
    return tuple(
        normalise(role.get(key)) or "unavailable"
        for key in ("head_span", "answer_role", "answer_target")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=ANALYSIS)
    parser.add_argument("--out-dir", default="c:/SCP/outputs/query_span_analysis")
    args = parser.parse_args(argv)

    rows = json.load(open(args.analysis, encoding="utf-8"))["spans"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[dedup_key(row)].append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    blind_path = f"{args.out_dir}/query_span_annotation_blind.csv"
    pred_path = f"{args.out_dir}/query_span_annotation_predictions.csv"

    with open(blind_path, "w", encoding="utf-8", newline="") as blind, open(
        pred_path, "w", encoding="utf-8", newline=""
    ) as predictions:
        blind_writer = csv.writer(blind)
        pred_writer = csv.writer(predictions)
        blind_writer.writerow(
            [
                "annotation_id", "task_id", "question", "question_head_span",
                "answer_role", "answer_target", "span_text", "original_text",
                "local_context", "enclosing_unit", "human_role", "human_boundary",
                "should_enter_query", "annotation_confidence", "notes",
            ]
        )
        # `repair_source` lives here rather than in the blind set: `ner_entity`
        # or `noun_chunk` tells an annotator what an upstream tool concluded
        # about the span, which is the kind of anchoring the split is for.
        pred_writer.writerow(
            [
                "annotation_id", "task_id", "span_text", "occurrences",
                "repair_source",
                "embedding_top_role", "embedding_second_role", "embedding_margin",
                "final_top_role", "final_second_role", "final_margin",
                "selected_role", "argmax_changed_by_heuristic",
                "margin_amplified_by_heuristic", "pushed_over_min_confidence",
                "abstained", "boundary_status", "missing_boundary_content",
                "adjustments", "adjustment_attribution",
                "in_query_occurrences", "in_query_varies_across_runs",
            ]
        )

        for index, (key, group) in enumerate(sorted(groups.items()), start=1):
            first = group[0]
            annotation_id = f"S{index:03d}"
            question = question_of(first["run"], first["task_id"])
            record = span_record(first["run"], first["task_id"], key[1])
            head_span, answer_role, answer_target = question_role_fields(record)
            context = normalise(record.get("context"))
            blind_writer.writerow(
                [
                    annotation_id, first["task_id"], question, head_span,
                    answer_role, answer_target, first["text"],
                    first["original_text"], context, first["enclosing_unit"],
                    "", "", "", "", "",
                ]
            )
            in_query = [bool(r["in_generated_query"]) for r in group]
            pred_writer.writerow(
                [
                    annotation_id, first["task_id"], first["text"], len(group),
                    first["repair_source"],
                    first["embedding_top_role"], first["embedding_second_role"],
                    first["embedding_margin"], first["final_top_role"],
                    first["final_second_role"], first["final_margin"],
                    first["selected_role"], first["argmax_changed_by_heuristic"],
                    first["margin_amplified_by_heuristic"],
                    first["embedding_margin"] < 0.015 <= first["final_margin"],
                    first["abstained"], first["boundary_status"],
                    first["missing_boundary_content"],
                    json.dumps(first["adjustments"], ensure_ascii=False),
                    first["adjustment_attribution"],
                    sum(in_query), len(set(in_query)) > 1,
                ]
            )

    print(f"unique case {len(groups)}（來自 {len(rows)} 個 occurrence）")
    print(f"  blind      {blind_path}")
    print(f"  predictions{pred_path}")
    print(f"\nhuman_role         : {HUMAN_ROLE_OPTIONS}")
    print(f"human_boundary     : {HUMAN_BOUNDARY_OPTIONS}")
    print(f"should_enter_query : {SHOULD_ENTER_QUERY_OPTIONS}")


if __name__ == "__main__":
    sys.exit(main())
