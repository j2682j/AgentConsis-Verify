"""Which tasks can have a gold answer checked by searching for its text?

The funnel metrics -- did the gold reach the documents, the evidence, the Stage 1
context -- need to find the gold in text. For `Mercedes Sosa` that works. For an
answer of `3`, searching for the digit finds page numbers, counts and years, and
a hit means nothing. Scoring those tasks anyway would put noise in the numerator
of the only metric the ablation turns on.

So eligibility is decided from the question, the gold answer and the answer type,
and decided before either arm runs. Nothing here reads a retrieval result.

Two of the exclusions cannot be settled mechanically. Whether an answer is
derived, and whether it needs several sources combined, are judgements about the
task -- so those columns carry a proposal and the ledger is written for a person
to overrule. A rule that silently guessed would decide the experiment's
denominator on its own.
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
OUT = "c:/SCP/outputs/oracle_boundary_ablation"

EXCLUSION_REASONS = (
    "boolean_answer",
    "too_short_or_ambiguous",
    "gold_present_in_question",
    "derived_answer",
    "multi_source_answer",
    "non_string_representable",
    "eligible",
)

MATCH_MODES = ("exact_phrase", "bounded_numeric", "component_list")

BOOLEAN = {"yes", "no", "true", "false"}

#: A list answer is only found when every part of it is. Hitting one component
#: of `Cornstarch, Freshly ground black pepper` and calling that a hit would
#: score partial retrieval as success.
#:
#: Splitting on ` and ` was wrong and had to go. It cut the book title `Five
#: Hundred Things To Eat Before It's Too Late: and the Very Best Places to Eat
#: Them` into two fragments that could then be matched in different documents.
#: Under-splitting is the safe direction: a genuine `A and B` is treated as one
#: phrase and must appear whole, which is stricter, not looser.
LIST_SEPARATORS = re.compile(r"\s*[,;]\s*")

#: Applied in order, so a task with several disqualifications always reports the
#: same primary one. Structural disqualifications come first: a gold that the
#: question already contains cannot be evidence of retrieval no matter what else
#: is true of it, whereas shortness is only a matching problem.
EXCLUSION_PRECEDENCE = (
    "gold_present_in_question",
    "non_string_representable",
    "derived_answer",
    "multi_source_answer",
    "boolean_answer",
    "too_short_or_ambiguous",
)

#: What kind of check the gold needs, decided per task by a person.
ANSWER_KINDS = (
    "direct_string",
    "component_list_single_source",
    "derived",
    "multi_source",
)


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def gold_answers() -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(
        "c:/SCP/data/gaia/2023/validation/metadata.level1.parquet"
    ).reset_index(drop=True)
    return {
        f"{index + 1:03d}": normalise(frame.loc[index, "Final answer"])
        for index in range(len(frame))
    }


def classify(task: str, question: str, gold: str) -> dict:
    """Propose an eligibility, and say what the proposal rests on.

    Every reason that applies is kept, and one of them is named primary by the
    frozen precedence. Storing only the winner loses the fact that `2` is both
    a counted result and a string too short to match, which two people would
    otherwise label differently and neither would be wrong.
    """

    folded = gold.casefold()
    components = [c for c in LIST_SEPARATORS.split(gold) if c.strip()]
    numeric = bool(re.fullmatch(r"[\d.,]+", gold))
    # Component-level, not whole-string. Q030's answer is five vegetables the
    # question itself lists, and a whole-string test says they are absent while
    # every one of them is right there in the prompt.
    in_question = bool(gold) and (
        folded in question.casefold()
        or (len(components) > 1
            and all(c.casefold() in question.casefold() for c in components))
    )

    reasons: list[str] = []
    if not gold:
        reasons.append("non_string_representable")
    if folded in BOOLEAN:
        reasons.append("boolean_answer")
    if in_question:
        reasons.append("gold_present_in_question")
    if gold and len(gold) <= 2:
        reasons.append("too_short_or_ambiguous")
    if numeric:
        reasons.append("derived_answer")

    primary = next((r for r in EXCLUSION_PRECEDENCE if r in reasons), "eligible")
    needs_review = primary in ("derived_answer", "too_short_or_ambiguous")

    if not reasons:
        eligible = "yes"
    elif needs_review:
        eligible = "REVIEW"
    else:
        eligible = "no"

    if numeric:
        match_mode = "bounded_numeric"
    elif len(components) > 1:
        match_mode = "component_list"
    else:
        match_mode = "exact_phrase"

    notes = []
    if primary == "gold_present_in_question":
        notes.append("gold（或其全部成分）已出現在題目中，檢索命中不構成證據")
    if numeric:
        notes.append("數字答案：需確認是否為推導結果，"
                     "以及 token 邊界是否會被年份／頁碼／計數誤命中")
    if len(components) > 1:
        notes.append("清單答案：所有 required_components 須全部命中；"
                     "並需決定是否要求同一文件")

    return {
        "task_id": task,
        "gold_answer": gold,
        "normalized_gold": normalise(gold),
        "eligible": eligible,
        "exclusion_reasons": reasons or ["eligible"],
        "primary_exclusion_reason": primary,
        "match_mode": match_mode if eligible != "no" else "",
        "required_components": components if len(components) > 1 else [gold],
        "gold_already_in_question": in_question,
        # Proposals only. A person decides these; a rule that guessed would be
        # setting the experiment's denominator by itself.
        "answer_kind": "REVIEW",
        "requires_derivation": "REVIEW" if numeric else "no",
        "requires_multiple_sources": "REVIEW",
        "review_note": "；".join(notes),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(
        open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8")))
    questions = {r["task_id"]: normalise(r["question"]) for r in rows}
    gold = gold_answers()

    ledger = [
        classify(task, questions[task], gold.get(task, ""))
        for task in sorted(questions)
    ]

    path = f"{OUT}/eligibility_ledger.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        for record in ledger:
            row = dict(record)
            row["required_components"] = " | ".join(row["required_components"])
            row["exclusion_reasons"] = " | ".join(row["exclusion_reasons"])
            writer.writerow(row)

    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"eligibility ledger {len(ledger)} 筆 -> {path}")
    print(f"   sha256 {digest}\n")

    by_reason: dict[str, list[str]] = {}
    for record in ledger:
        by_reason.setdefault(record["primary_exclusion_reason"], []).append(record["task_id"])
    for reason in EXCLUSION_REASONS:
        tasks = by_reason.get(reason, [])
        if tasks:
            print(f"   {reason:<26} {len(tasks):>3}  {', '.join(tasks)}")

    auto_yes = sum(1 for r in ledger if r["eligible"] == "yes")
    review = sum(1 for r in ledger if r["eligible"] == "REVIEW")
    print(f"\n   自動判定 eligible {auto_yes}、待人工覆核 {review}、"
          f"自動排除 {len(ledger) - auto_yes - review}")
    print(f"   requires_multiple_sources 全部標記 REVIEW —— 無法機械判定")

    # Printed with the question, because `answer_kind` cannot be judged from a
    # gold string alone: whether `Guatemala` was looked up or worked out is a
    # fact about what the task asked for, not about the answer's shape.
    print(f"\n=== 逐題覆核（answer_kind 四選一："
          f"direct_string / component_list_single_source / derived / multi_source）")
    for record in ledger:
        task = record["task_id"]
        mark = {"yes": " ", "REVIEW": "?", "no": "x"}[record["eligible"]]
        print(f"\n[{mark}] {task}  gold={record['gold_answer'][:50]!r}"
              f"  [{record['match_mode'] or '—'}]")
        print(f"     理由: {' | '.join(record['exclusion_reasons'])}"
              f"  → 主要: {record['primary_exclusion_reason']}")
        if len(record["required_components"]) > 1:
            print(f"     components: {record['required_components']}")
        if record["review_note"]:
            print(f"     註: {record['review_note']}")
        print(f"     題目: {questions[task][:118]}")


if __name__ == "__main__":
    sys.exit(main())
