"""Where does a span lose the rest of its unit, and what should it have been?

Human annotation marked 38 of 133 spans as fragmented; automatic detection found
3, because it looked for parentheses and quotes and only 3 fragments sit inside
either. The rest are cut mid-phrase: `highest number` without `of bird species`,
`11 of Doctor Who` without `Series 9, Episode`, `Nature journal's` without
`Scientific Reports`.

`repair_source` does not separate them -- `noun_chunk+span_rescore` produced 21
fragments and 41 clean spans, `ner_entity+span_rescore` 12 and 43 -- so the
stage cannot be read off the label. What can be read is how `original_text`
relates to `span_text`: on several fragments the *original* span covered more of
the unit than the repaired one, which points at rescore shortening rather than
extraction missing.

Candidate units are proposed mechanically and the shortfall measured on each
side. The gold boundary is not decided here. A target invented by the same code
that will later be judged against it proves nothing, so `gold_complete_span` is
left for a human and must be copied verbatim from the context.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

BASE = "c:/SCP/outputs/query_span_analysis"

#: Death points, as far as the recorded fields can distinguish them.
DEATH_POINTS = (
    "extraction_short",
    "rescore_shortened",
    "rescore_shifted",
    "punctuation_stripped",
    "unit_absent_in_context",
    "unknown",
)

BRACKETED = re.compile(r"\(([^()]{2,90})\)|\"([^\"]{2,90})\"|\u201c([^\u201d]{2,90})\u201d")


@dataclass
class BoundaryCase:
    annotation_id: str
    task_id: str
    span_text: str
    original_text: str
    local_context: str
    human_role: str
    in_query: bool
    candidates: list[str] = field(default_factory=list)
    left_missing_tokens: int = 0
    right_missing_tokens: int = 0
    death_point: str = "unknown"


def tokens(value: str) -> list[str]:
    return [t for t in re.split(r"\s+", str(value or "").strip()) if t]


def propose(span: str, context: str) -> list[str]:
    """Whole units the span sits inside, under several mechanical readings.

    Tightest plausible unit first. Nothing here is authoritative; each entry is
    a starting point for the human who decides what the unit actually was.
    """

    found: list[str] = []
    span_lower = span.casefold()
    start = context.casefold().find(span_lower)
    if start < 0:
        return found
    end = start + len(span)

    for match in BRACKETED.finditer(context):
        unit = next((group for group in match.groups() if group), "")
        if span_lower in unit.casefold():
            found.append(unit.strip())

    left_words = tokens(context[:start])
    right_words = tokens(context[end:])
    span_words = tokens(span)
    for back in range(0, 5):
        for forward in range(0, 5):
            if back == 0 and forward == 0:
                continue
            unit = " ".join(
                left_words[len(left_words) - back:] + span_words + right_words[:forward]
            ).strip(" ,.;:")
            if unit and unit.casefold() != span_lower:
                found.append(unit)

    for part in re.split(r"[,;:?]", context):
        if span_lower in part.casefold() and part.strip():
            found.append(part.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for unit in found:
        key = unit.casefold()
        if key not in seen and len(unit) <= 120:
            seen.add(key)
            unique.append(unit)
    return unique[:8]


def classify(case: BoundaryCase) -> str:
    span_lower = case.span_text.casefold()
    original_lower = case.original_text.casefold()
    context_lower = case.local_context.casefold()
    if span_lower not in context_lower:
        return "unit_absent_in_context"
    if original_lower and original_lower not in context_lower:
        return "unknown"
    if original_lower and span_lower != original_lower:
        if span_lower in original_lower:
            return "rescore_shortened"
        if original_lower not in span_lower:
            return "rescore_shifted"
    if BRACKETED.search(case.local_context) and any(
        span_lower in (next((g for g in m.groups() if g), "") or "").casefold()
        for m in BRACKETED.finditer(case.local_context)
    ):
        return "punctuation_stripped"
    return "extraction_short"


#: Whether a whole unit exists to recover, kept out of the span column so that
#: `human_gold_span` only ever holds text copied from the context. Writing
#: `"unrecoverable"` there would leave grounding to match a status word against
#: the question and find nothing -- or worse, find something.
RECOVERY_STATUS = ("recoverable", "unrecoverable", "drop")

#: `keep` lets an annotator overturn the earlier `fragmented` call rather than
#: being forced to name a repair for a span that did not need one.
REPAIR_DIRECTIONS = ("keep", "left", "right", "both", "replace")

#: What kind of thing the unit is, and how its edges are marked. A quoted paper
#: title is both `title` and `quoted`; collapsing them into one column would
#: force a choice between "use NER because it is an entity" and "use quote
#: recovery because it has quoted edges", which are different repairs.
UNIT_TYPES = ("entity", "title", "date", "noun_phrase", "clause", "other")
BOUNDARY_FORMS = ("plain", "parenthetical", "quoted", "punctuation_delimited", "other")


def ground_gold(path: str) -> list[dict[str, object]]:
    """Locate each annotated gold span in its own context, exactly.

    A gold boundary that cannot be found in the text it came from is not a
    boundary -- it is a paraphrase, and measuring a recovery method against it
    would score the method on reproducing someone's wording. Rows that miss, or
    that match in several places without being told which, are held back rather
    than folded into the reference set.
    """

    out: list[dict[str, object]] = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        gold = (row.get("human_gold_span") or "").strip()
        status = (row.get("recovery_status") or "").strip()
        context = row.get("local_context") or ""
        record: dict[str, object] = {
            "annotation_id": row["annotation_id"],
            "recovery_status": status,
            "human_gold_span": gold,
        }
        if status in ("unrecoverable", "drop"):
            record["grounding"] = "not_applicable"
            record["offsets"] = None
            if gold:
                record["grounding"] = "unexpected_span_for_status"
        elif not gold:
            record["grounding"] = "missing_gold"
            record["offsets"] = None
        else:
            hits = [
                m.start()
                for m in re.finditer(re.escape(gold), context)
            ]
            if len(hits) == 1:
                record["grounding"] = "grounded"
                record["offsets"] = [hits[0], hits[0] + len(gold)]
            elif len(hits) > 1:
                record["grounding"] = "ambiguous_occurrence"
                record["offsets"] = [[h, h + len(gold)] for h in hits]
            else:
                record["grounding"] = "grounding_failed"
                record["offsets"] = None
        out.append(record)
    return out


def main() -> None:
    merged = list(
        csv.DictReader(open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8"))
    )
    predictions = {
        row["annotation_id"]: row
        for row in csv.DictReader(
            open(f"{BASE}/query_span_annotation_predictions.csv", encoding="utf-8")
        )
    }

    cases: list[BoundaryCase] = []
    for row in merged:
        if row["human_boundary"] != "fragmented":
            continue
        case = BoundaryCase(
            annotation_id=row["annotation_id"],
            task_id=row["task_id"],
            span_text=row["span_text"],
            original_text=row["original_text"],
            local_context=row["local_context"],
            human_role=row["human_role"],
            in_query=int(predictions[row["annotation_id"]]["in_query_occurrences"]) > 0,
        )
        case.candidates = propose(case.span_text, case.local_context)
        case.death_point = classify(case)
        widest = max(case.candidates, key=len) if case.candidates else ""
        if widest:
            widest_tokens = tokens(widest)
            span_tokens = tokens(case.span_text)
            joined = " ".join(widest_tokens).casefold()
            offset = joined.find(" ".join(span_tokens).casefold())
            if offset >= 0:
                case.left_missing_tokens = len(tokens(joined[:offset]))
                case.right_missing_tokens = max(
                    0, len(widest_tokens) - len(span_tokens) - case.left_missing_tokens
                )
        cases.append(case)

    # Two files. The proposals are a sliding window with no grammar, so a
    # reviewer shown `highest number of bird species` beside `highest number of`
    # is being offered an answer before being asked the question -- and the gold
    # boundary is what the recovery method will later be judged against, so it
    # cannot be anchored on what the current code happens to suggest.
    #
    # `answer_requirement` is deliberately absent: it is null on all 133 rows,
    # and the three `question_role` fields carry what it was meant to supply.
    by_id = {row["annotation_id"]: row for row in merged}
    gold_path = f"{BASE}/boundary_gold_blind.csv"
    with open(gold_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["annotation_id", "task_id", "question", "question_head_span",
             "answer_role", "answer_target", "span_text", "original_text",
             "local_context", "recovery_status", "human_gold_span",
             "repair_direction", "unit_type", "boundary_form",
             "boundary_confidence", "acceptable_alternative", "notes"]
        )
        for case in cases:
            row = by_id[case.annotation_id]
            writer.writerow(
                [case.annotation_id, case.task_id, row["question"],
                 row["question_head_span"], row["answer_role"],
                 row["answer_target"], case.span_text, case.original_text,
                 case.local_context, "", "", "", "", "", "", "", ""]
            )

    with open(f"{BASE}/boundary_recovery_predictions.csv", "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["annotation_id", "proposed_units", "inferred_death_point",
             "proposed_left_gap", "proposed_right_gap", "in_query", "human_role"]
        )
        for case in cases:
            writer.writerow(
                [case.annotation_id, " | ".join(case.candidates),
                 case.death_point, case.left_missing_tokens,
                 case.right_missing_tokens, case.in_query, case.human_role]
            )

    controls = [r for r in merged if r["human_boundary"] in ("complete", "unrelated")]
    with open(f"{BASE}/boundary_controls.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["annotation_id", "task_id", "span_text", "human_boundary", "human_role"]
        )
        for row in controls:
            writer.writerow(
                [row["annotation_id"], row["task_id"], row["span_text"],
                 row["human_boundary"], row["human_role"]]
            )

    complete = sum(1 for r in controls if r["human_boundary"] == "complete")
    unrelated = sum(1 for r in controls if r["human_boundary"] == "unrelated")
    print(f"fragmented {len(cases)} 筆 -> {gold_path}（無機械候選）")
    print(f"機械提案另存 -> {BASE}/boundary_recovery_predictions.csv")
    print(f"對照組 {len(controls)} 筆（complete {complete}、unrelated {unrelated}）")
    print(f"\n死亡點推定: {dict(Counter(c.death_point for c in cases).most_common())}")
    print(f"有候選單位的: {sum(1 for c in cases if c.candidates)}/{len(cases)}")
    print(f"進入 query 的: {sum(1 for c in cases if c.in_query)}/{len(cases)}")
    print(f"平均缺口（provisional）: "
          f"左 {sum(c.left_missing_tokens for c in cases)/len(cases):.1f}"
          f"、右 {sum(c.right_missing_tokens for c in cases)/len(cases):.1f} token")


if __name__ == "__main__":
    sys.exit(main())
