"""Read every query the system generates, before spending nine hours on retrieval.

The boundary line ran for a long time -- annotation, an oracle, an ablation, four
selector revisions, a freeze protocol, a holdout -- and nobody had looked at a
generated query. The first one read contained `2:009` where the question said
`2009`, a corruption that removes the year constraint entirely. Twenty-five of
these twenty-nine questions carry a year, a date or a number.

So the queries get read first. A paired retrieval experiment run against a
baseline with a known systematic defect measures the defect as much as anything
else, and nine hours of results would be stale the moment the next one is found.

The checks are fixed in advance and split by what can be decided objectively.
Corruption of a literal the question contains is a fact; whether a query is any
good is not, and this file only records the second kind. Repairing anything that
lacks an unambiguous invariant would be tuning the query generator against these
twenty-nine tasks, which is the trap the boundary work already fell into.

No search runs. Query generation only.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, r"c:/SCP")

from tools.search_result_builder.query.query_literal_integrity import (
    protected_literals,
    repair_query,
)

BASE = "c:/SCP/outputs/query_span_analysis"
OUT = "c:/SCP/outputs/query_audit"

QUOTED = re.compile(r'"([^"]{3,120})"|\u201c([^\u201d]{3,120})\u201d')

#: Objective: each is a property of the question and the query, not a judgement
#: about quality.
OBJECTIVE_CHECKS = (
    "protected_literal_corrupted",
    "quoted_title_corrupted",
    "format_instruction_in_query",
    "question_echo",
    "duplicate_query",
    "query_at_length_cap",
    "entity_not_in_question",
)

#: Recorded only. `answer_target_absent` is not a defect on its own -- plenty of
#: good queries leave the sought thing out -- and `relation_sent_whole` needs a
#: reading of what the relation was for.
OBSERVATIONAL_CHECKS = ("answer_target_absent", "relation_sent_whole")

#: `_clean_query` drops anything longer. A query sitting on the cap is either
#: truncated or about to be discarded, and both are worth seeing.
LENGTH_CAP = 220

STOPWORDS = frozenset(
    "a an the of in on at to for and or is was were be been being with from by "
    "what which who whom whose how many much when where why that this these "
    "those it its as into during between about above after before".split()
)


def normalise(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def words(text: str) -> set[str]:
    return {
        w for w in re.split(r"[^\w']+", text.casefold())
        if w and w not in STOPWORDS and len(w) > 2
    }


def quoted_titles(question: str) -> list[str]:
    out = []
    for match in QUOTED.finditer(question):
        title = next((g for g in match.groups() if g), "")
        if title:
            out.append(title.strip())
    return out


def audit_query(query: str, question: str, context: dict) -> list[str]:
    """Every check this query trips, named."""

    tripped: list[str] = []
    folded = query.casefold()

    if repair_query(query, question).changed:
        tripped.append("protected_literal_corrupted")

    for title in quoted_titles(question):
        # Only a query that means to carry the title is judged on carrying it.
        overlap = words(title) & words(query)
        if overlap and len(overlap) >= 2 and title.casefold() not in folded:
            tripped.append("quoted_title_corrupted")
            break

    for instruction in context.get("format_instructions", []):
        if instruction and instruction.casefold() in folded:
            tripped.append("format_instruction_in_query")
            break

    question_words = words(question)
    query_words = words(query)
    if query_words and len(query_words & question_words) / len(query_words) > 0.95:
        if len(query_words) >= max(6, len(question_words) * 0.8):
            tripped.append("question_echo")

    if len(query) >= LENGTH_CAP - 5:
        tripped.append("query_at_length_cap")

    novel = query_words - question_words
    if novel:
        tripped.append("entity_not_in_question")

    answer_target = normalise(context.get("answer_target"))
    if answer_target and answer_target.casefold() not in folded:
        tripped.append("answer_target_absent")

    for goal in context.get("relation_goals", []):
        phrase = normalise(
            f"{goal.get('subject','')} {goal.get('relation','')} {goal.get('target','')}"
        )
        if phrase and phrase.casefold() in folded:
            tripped.append("relation_sent_whole")
            break

    return tripped


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(
        open(f"{BASE}/query_span_annotation_merged.csv", encoding="utf-8")))
    questions = {r["task_id"]: normalise(r["question"]) for r in rows}
    formats: dict[str, list[str]] = {}
    targets: dict[str, str] = {}
    for row in rows:
        if row["human_role"] == "format_instruction":
            formats.setdefault(row["task_id"], []).append(normalise(row["span_text"]))
        targets.setdefault(row["task_id"], normalise(row["answer_target"]))

    from tools.search_result_builder.query.mask_salience_query import (
        MaskSalienceQueryGenerator,
    )

    generator = MaskSalienceQueryGenerator()
    ledger = f"{OUT}/query_audit.jsonl"
    done = set()
    if os.path.exists(ledger):
        for line in open(ledger, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["task_id"])

    with open(ledger, "a", encoding="utf-8") as handle:
        for task in sorted(questions):
            if task in done:
                continue
            question = questions[task]
            try:
                candidates = generator.generate(question, num_candidates=3)
                queries = [c.query for c in candidates]
                relation = generator.last_relation_plan
                goals = [
                    g if isinstance(g, dict) else getattr(g, "__dict__", {})
                    for g in (getattr(relation, "goals", []) or [])
                ]
                error = None
            except Exception as exc:
                queries, goals, error = [], [], f"{type(exc).__name__}: {exc}"

            context = {
                "format_instructions": formats.get(task, []),
                "answer_target": targets.get(task, ""),
                "relation_goals": goals,
            }
            seen: set[str] = set()
            per_query = []
            for query in queries:
                tripped = audit_query(query, question, context)
                if query.casefold() in seen:
                    tripped.append("duplicate_query")
                seen.add(query.casefold())
                per_query.append({"query": query, "checks": tripped})

            record = {
                "task_id": task,
                "question": question,
                "queries": per_query,
                "repairs": list(generator.last_query_repairs),
                "protected_literals": sorted(protected_literals(question)),
                "spans": [s.text for s in generator.last_salient_spans],
                "error": error,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            flags = sorted({c for q in per_query for c in q["checks"]})
            print(f"   {task}  {len(queries)} query"
                  f"  修復 {len(generator.last_query_repairs)}"
                  f"  {'、'.join(flags) if flags else '—'}")

    report(ledger)


def report(ledger: str) -> None:
    records = [json.loads(l) for l in open(ledger, encoding="utf-8") if l.strip()]
    total_tasks = len(records)
    total_queries = sum(len(r["queries"]) for r in records)

    print(f"\n=== {total_tasks} task、{total_queries} query")
    print(f"{'check':<32}{'task 觸發':>12}{'query 觸發':>14}")
    for check in OBJECTIVE_CHECKS + OBSERVATIONAL_CHECKS:
        tasks = sum(
            1 for r in records if any(check in q["checks"] for q in r["queries"])
        )
        queries = sum(
            1 for r in records for q in r["queries"] if check in q["checks"]
        )
        kind = "" if check in OBJECTIVE_CHECKS else "  (僅記錄)"
        print(f"{check:<32}{tasks:>6}/{total_tasks}{queries:>8}/{total_queries}{kind}")

    repaired = [r for r in records if r["repairs"]]
    print(f"\n=== protected literal 修復 {len(repaired)}/{total_tasks} task")
    for record in repaired:
        for repair in record["repairs"]:
            for item in repair["repairs"]:
                print(f"   {record['task_id']}  {item['before']!r} -> "
                      f"{item['after']!r}  ({item['reason']})")

    empty = [r["task_id"] for r in records if not r["queries"]]
    if empty:
        print(f"\n=== 未產出任何 query 的 task {len(empty)}: {empty}")

    print(f"\n=== 逐題 query")
    for record in records:
        print(f"\n{record['task_id']}  literals={record['protected_literals']}")
        for entry in record["queries"]:
            marks = f"  [{'、'.join(entry['checks'])}]" if entry["checks"] else ""
            print(f"     {entry['query']!r}{marks}")


if __name__ == "__main__":
    sys.exit(main())
