"""Where does a question's answer-format text stop being recognisable as one?

Task 006 asks for authors "(First M. Last)" and searched for
`author:First M Last: (Pie Menus or Linear Menus, Which Is Better?) 2015`. The
format specification became a search entity, and it did so through a chain that
the recorded spans can be walked back through:

    salience picks a candidate     `First`
    boundary repair extends it     `First M`      the unit is `(First M. Last)`
    embedding ranks the roles      source_clue by 0.024 over format_instruction
    heuristics adjust              `_looks_like_title` adds 0.035 to source_clue
    the role decides inclusion     source_clue and constraint become must_include

Four different repairs sit behind those steps, so the loss has to be attributed
before anything is changed:

    candidate_missing        salience never proposed the format text
    boundary_fragmented      it did, and repair left the unit incomplete
    role_misclassified       the whole unit was present and still scored wrong
    query_state_leakage      it was classified as format and reached the query

`similarities` holds the embedding scores and `role_scores` the same scores
after the heuristic table above, both recorded for every span, so the embedding
argmax and the final argmax can be compared without re-running anything.

Two definitions this depends on. `other` is not one outcome: a span abstains
when its final margin falls under `min_confidence`, and is semantically other
when `other` wins outright. And `original_text != text` is not fragmentation --
repair is supposed to extend spans -- so fragmentation is measured against the
enclosing quoted or parenthetical unit in the span's own context.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

ROLES = ("source_clue", "constraint", "answer_target", "format_instruction", "other")

#: `SpanRoleClassifier._role_from_scores`.
MIN_CONFIDENCE = 0.015

#: `SpanRoleClassifier._apply_heuristics`, as of commit 2a096f3.
HEURISTICS = {
    "question_role_overlap": ("answer_target", 0.08),
    "url_shape": ("source_clue", 0.08),
    "source_entity": ("source_clue", 0.045),
    "constraint_entity": ("constraint", 0.035),
    "title_shape": ("source_clue", 0.035),
    "constraint_term": ("constraint", 0.035),
    "format_term": ("format_instruction", 0.05),
    "single_token_other": ("other", 0.025),
}

#: A parenthetical or quoted run in the context is one unit for this purpose.
ENCLOSING = re.compile(r"\(([^()]{2,80})\)|\"([^\"]{2,80})\"|“([^”]{2,80})”")


@dataclass
class SpanAnalysis:
    run: str
    task_id: str
    text: str
    original_text: str
    repair_source: str
    entity_label: str
    embedding_top_role: str = ""
    embedding_second_role: str = ""
    embedding_margin: float = 0.0
    final_top_role: str = ""
    final_second_role: str = ""
    final_margin: float = 0.0
    selected_role: str = ""
    argmax_changed_by_heuristic: bool = False
    margin_amplified_by_heuristic: bool = False
    abstained: bool = False
    semantic_other: bool = False
    adjustments: dict[str, float] = field(default_factory=dict)
    adjustment_attribution: str = "unavailable"
    enclosing_unit: str = ""
    boundary_status: str = "no_enclosing_unit"
    missing_boundary_content: str = ""
    included_in_target: bool = False
    included_in_must_include: bool = False
    in_generated_query: bool = False


def _ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((role, float(scores.get(role) or 0.0)) for role in ROLES),
        key=lambda item: item[1],
        reverse=True,
    )


def _decompose(similarities: dict[str, float], role_scores: dict[str, float]) -> tuple[dict, str]:
    """Split the per-role delta into the heuristics that could have produced it.

    Several adjustments land on `source_clue`, so a delta of 0.08 is either the
    URL rule alone or the entity and title rules together. Where the sum is
    ambiguous the attribution says so rather than guessing.
    """

    deltas = {
        role: round(float(role_scores.get(role) or 0.0) - float(similarities.get(role) or 0.0), 6)
        for role in ROLES
    }
    named: dict[str, float] = {}
    ambiguous = False
    for role, delta in deltas.items():
        if abs(delta) < 1e-6:
            continue
        options = [
            (name, value)
            for name, (target, value) in HEURISTICS.items()
            if target == role
        ]
        exact = [name for name, value in options if abs(value - delta) < 1e-6]
        if len(exact) == 1:
            named[exact[0]] = delta
            continue
        combos = [
            (a, b)
            for i, (a, av) in enumerate(options)
            for b, bv in options[i + 1 :]
            if abs(av + bv - delta) < 1e-6
        ]
        if len(exact) == 0 and len(combos) == 1:
            named[f"{combos[0][0]}+{combos[0][1]}"] = delta
            continue
        named[f"{role}_delta"] = delta
        ambiguous = True
    return named, ("ambiguous" if ambiguous else "exact") if named else "none"


def _boundary(text: str, context: str) -> tuple[str, str, str]:
    """Does the span cover the quoted or parenthetical unit it sits inside?"""

    lowered = text.casefold().strip()
    for match in ENCLOSING.finditer(context or ""):
        unit = next((group for group in match.groups() if group), "").strip()
        if not unit or lowered not in unit.casefold():
            continue
        if unit.casefold() == lowered:
            return unit, "covers_unit", ""
        missing = re.sub(re.escape(text), "", unit, flags=re.IGNORECASE).strip(" .,")
        return unit, "boundary_fragmented", missing
    return "", "no_enclosing_unit", ""


def analyse_run(run: str) -> list[SpanAnalysis]:
    out: list[SpanAnalysis] = []
    for path in sorted(glob.glob(f"c:/SCP/outputs/{run}/tasks/*.json")):
        task = json.loads(open(path, encoding="utf-8").read())
        task_id = os.path.basename(path).split("_")[0]
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
        plan = (raw.get("diagnostics") or {}).get("query_plan") or {}
        spans = plan.get("classified_spans") or []
        queries = " ".join(str(q) for q in (plan.get("queries") or []))
        requests = plan.get("query_requests") or []
        target_text = " ".join(
            str((item.get("source_requirement") or {}).get("source_hint") or "")
            + " "
            + str(item.get("target") or "")
            for item in requests
            if isinstance(item, dict)
        )
        must_include = " ".join(
            str(term)
            for item in requests
            if isinstance(item, dict)
            for term in (item.get("must_include") or [])
        )
        intent = (raw.get("diagnostics") or {}).get("search_intent_plan") or {}
        target_text += " " + str(intent.get("target") or "")
        must_include += " " + " ".join(str(t) for t in (intent.get("must_include") or []))

        for span in spans:
            if not isinstance(span, dict):
                continue
            text = str(span.get("text") or "")
            similarities = span.get("similarities") or {}
            role_scores = span.get("role_scores") or {}
            embedding = _ranked(similarities)
            final = _ranked(role_scores)
            adjustments, attribution = _decompose(similarities, role_scores)
            unit, status, missing = _boundary(text, str(span.get("context") or ""))
            final_margin = round(final[0][1] - final[1][1], 6) if len(final) > 1 else 0.0
            embedding_margin = (
                round(embedding[0][1] - embedding[1][1], 6) if len(embedding) > 1 else 0.0
            )
            selected = str(span.get("role") or "")
            out.append(
                SpanAnalysis(
                    run=run,
                    task_id=task_id,
                    text=text,
                    original_text=str(span.get("original_text") or ""),
                    repair_source=str(span.get("repair_source") or ""),
                    entity_label=str(span.get("entity_label") or ""),
                    embedding_top_role=embedding[0][0],
                    embedding_second_role=embedding[1][0] if len(embedding) > 1 else "",
                    embedding_margin=embedding_margin,
                    final_top_role=final[0][0],
                    final_second_role=final[1][0] if len(final) > 1 else "",
                    final_margin=final_margin,
                    selected_role=selected,
                    argmax_changed_by_heuristic=embedding[0][0] != final[0][0],
                    margin_amplified_by_heuristic=(
                        embedding[0][0] == final[0][0] and final_margin > embedding_margin
                    ),
                    abstained=selected == "other" and final_margin < MIN_CONFIDENCE,
                    semantic_other=(
                        final[0][0] == "other" and final_margin >= MIN_CONFIDENCE
                    ),
                    adjustments=adjustments,
                    adjustment_attribution=attribution,
                    enclosing_unit=unit,
                    boundary_status=status,
                    missing_boundary_content=missing,
                    included_in_target=bool(text) and text.casefold() in target_text.casefold(),
                    included_in_must_include=bool(text)
                    and text.casefold() in must_include.casefold(),
                    in_generated_query=bool(text) and text.casefold() in queries.casefold(),
                )
            )
    return out


def report(rows: list[SpanAnalysis]) -> dict[str, Any]:
    total = len(rows)
    selected = Counter(r.selected_role for r in rows)
    embedding_top = Counter(r.embedding_top_role for r in rows)
    final_top = Counter(r.final_top_role for r in rows)
    changed = [r for r in rows if r.argmax_changed_by_heuristic]
    amplified = [r for r in rows if r.margin_amplified_by_heuristic]
    abstained = [r for r in rows if r.abstained]
    semantic = [r for r in rows if r.semantic_other]
    fragmented = [r for r in rows if r.boundary_status == "boundary_fragmented"]
    covers = [r for r in rows if r.boundary_status == "covers_unit"]
    heuristic_use = Counter(
        name for r in rows for name in r.adjustments if not name.endswith("_delta")
    )
    pushed_over = [
        r
        for r in rows
        if r.embedding_margin < MIN_CONFIDENCE <= r.final_margin and not r.abstained
    ]
    return {
        "spans": total,
        "selected_role_rate": {k: round(v / total, 4) for k, v in selected.most_common()},
        "embedding_argmax_rate": {k: round(v / total, 4) for k, v in embedding_top.most_common()},
        "final_argmax_rate": {k: round(v / total, 4) for k, v in final_top.most_common()},
        "embedding_margin_median": round(median([r.embedding_margin for r in rows]), 6),
        "final_margin_median": round(median([r.final_margin for r in rows]), 6),
        "argmax_changed_by_heuristic": len(changed),
        "margin_amplified_by_heuristic": len(amplified),
        "pushed_over_min_confidence": len(pushed_over),
        "abstained": len(abstained),
        "semantic_other": len(semantic),
        "boundary_fragmented": len(fragmented),
        "covers_enclosing_unit": len(covers),
        "heuristic_fire_counts": dict(heuristic_use.most_common()),
        "attribution": dict(Counter(r.adjustment_attribution for r in rows)),
        "reached_must_include": sum(1 for r in rows if r.included_in_must_include),
        "reached_query": sum(1 for r in rows if r.in_generated_query),
        "fragmented_and_reached_query": sum(
            1 for r in fragmented if r.in_generated_query
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        default="level1_final_13,level1_final_15,level1_final_16,"
        "level_1_final_20,level1_final_21,level1_final_22",
    )
    parser.add_argument("--out-dir", default="c:/SCP/outputs/query_span_analysis")
    args = parser.parse_args(argv)

    rows: list[SpanAnalysis] = []
    for run in args.runs.split(","):
        rows.extend(analyse_run(run.strip()))
    summary = report(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/query_span_role_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "spans": [asdict(r) for r in rows]},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with open(f"{args.out_dir}/query_span_annotation.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run", "task_id", "text", "original_text", "enclosing_unit",
                "boundary_status", "embedding_top", "final_top", "selected_role",
                "embedding_margin", "final_margin", "in_query", "human_label",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.run, r.task_id, r.text, r.original_text, r.enclosing_unit,
                    r.boundary_status, r.embedding_top_role, r.final_top_role,
                    r.selected_role, r.embedding_margin, r.final_margin,
                    r.in_generated_query, "",
                ]
            )
    for key, value in summary.items():
        print(f"{key:<34} {value}")
    print(f"\n寫入 {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
