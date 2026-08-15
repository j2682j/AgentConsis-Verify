"""Trace where the answer stops surviving, on fixed recorded retrieval.

Live reruns cannot attribute a change: measured across level1_final_13, _14 and
_15 with the same seed, no task retrieved the same document set twice and the
median Jaccard overlap was 0.33. Anything compared across runs is therefore
confounded by two thirds of the corpus turning over.

This reads one recorded run and walks the answer through the stages it has to
survive, all of which are already in the trace except the last, which is
rebuilt with the real context budget:

    retrieved document -> useful span -> classified span -> semantic fact
    -> direct contract -> grounded evidence -> unverified reference
    -> the search block Stage 1 actually sees

Every task is assigned the stage where the answer was last seen, so the output
says which step to work on rather than that something upstream is wrong.

Two measurement defects were corrected after this script's conclusions had to be
withdrawn, and both are worth knowing about before trusting any number here:

* `classified_span` read `doc["span_roles"]`, populated on 26 of 2339 documents.
  The classifier writes to `labeler_diagnostics.span_role_classifier.span_roles`,
  populated on 820. The stage looked dead and is not: nothing is lost there.
* presence of the gold string is not evidence that the passage answers the
  question. Task 007's "THE CASTLE" matched the prose *"the castle appears
  deserted"* while the question asks for a script's scene heading, and that
  false positive was read as proof of a model limit. `gold_string_present` and
  `relation_support_present` are now reported separately -- the second requires
  the gold to sit in a fact's object or in a span the classifier called
  ANSWER_SUPPORT, i.e. structurally an answer rather than an incidental word.

Composed answers are tracked by component. `Braintree, Honolulu` never appears
as one string on any page, so whole-string matching reports a recall failure
where the real question is whether the parts were retrieved.

    .\\venv312\\Scripts\\python.exe scripts/replay_evidence_funnel.py \\
      --run outputs/level1_final_15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unicodedata import normalize as un

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context.context_budget import ContextBudgetManager
from context.context_builder import ContextBuilder
from core.evidence_runner import EvidenceRunner

RENDERER = EvidenceRunner.__new__(EvidenceRunner)
BUDGET = ContextBudgetManager()
COMPRESSOR = ContextBuilder()

STAGES = (
    "retrieved_document",
    "useful_span",
    "classified_span",
    "semantic_fact",
    "direct_contract",
    "grounded_evidence",
    "unverified_reference",
    "stage1_context",
)


def normalise(value: Any) -> str:
    without_urls = re.sub(r"https?://\S+", " ", str(value or ""))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", un("NFKC", without_urls).casefold()).split())


def carries(haystack: Any, gold: str) -> bool:
    """Whether the answer appears in the text as a whole token, ignoring URLs.

    Word boundaries are required at every length, not only for short answers.
    Substring matching put `CUB` inside `Cuba` and selected a table of Swedish
    medals as the oracle passage for a question about athlete counts, which was
    then read as evidence that the models could not use perfect evidence.
    """

    needle = normalise(gold)
    if not needle:
        return False
    text = normalise(haystack)
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def components(gold: str) -> list[str]:
    """The parts of a composed answer, or the answer itself when it is one value.

    `Braintree, Honolulu` and `132, 133, 134, 197, 245` are assembled from facts
    that never appear adjacent, so asking whether the whole string was retrieved
    always answers no and hides whether the parts were there.
    """

    parts = [part.strip() for part in re.split(r"[,;]| and ", str(gold or "")) if part.strip()]
    return parts if len(parts) > 1 else [str(gold or "").strip()]


def _blob(items: Any) -> str:
    if isinstance(items, list):
        return "\n".join(json.dumps(item, ensure_ascii=False, default=str) if not isinstance(item, str) else item for item in items)
    return str(items or "")


def span_roles(document: dict) -> list[dict]:
    """Classifier output, from where it is actually written."""

    diagnostics = document.get("labeler_diagnostics")
    if isinstance(diagnostics, dict):
        classifier = diagnostics.get("span_role_classifier")
        if isinstance(classifier, dict) and classifier.get("span_roles"):
            return [item for item in classifier["span_roles"] if isinstance(item, dict)]
    return [item for item in document.get("span_roles") or [] if isinstance(item, dict)]


def semantic_facts(document: dict) -> list[dict]:
    facts = [item for item in document.get("semantic_facts") or [] if isinstance(item, dict)]
    for item in span_roles(document):
        facts.extend(fact for fact in item.get("semantic_facts") or [] if isinstance(fact, dict))
    return facts


class _Contract:
    def __init__(self, requirement: str, target: str) -> None:
        self.answer_requirement = requirement
        self.answer_target = target


def _contract(summary: dict) -> _Contract:
    refs = summary.get("unverified_references") or []
    return _Contract(
        next((str(r.get("answer_requirement") or "") for r in refs if r.get("answer_requirement")), ""),
        next((str(r.get("answer_target") or "") for r in refs if r.get("answer_target")), ""),
    )


def stage1_context(summary: dict) -> str:
    """The search block after the budget, which is what Stage 1 reads."""

    rendered = EvidenceRunner._render_web_retrieval_evidence(
        RENDERER,
        evidence_items=list(summary.get("evidence_items") or []),
        unverified_references=list(summary.get("unverified_references") or []),
        answer_candidates=list(summary.get("answer_candidates") or []),
        contract=_contract(summary),
    )
    compacted, _dropped = BUDGET._compact_search_evidence(
        COMPRESSOR._compress_multiline_text(rendered)
    )
    return compacted


def documents_of(summary: dict) -> list[dict]:
    return [
        doc
        for round_trace in summary.get("retrieval_rounds") or []
        for doc in round_trace.get("documents") or []
        if isinstance(doc, dict)
    ]


def survival(task: dict) -> dict[str, bool] | None:
    gold = str(task.get("expected") or "")
    summary = task.get("search_summary") or {}
    if not (summary.get("retrieval_rounds") or []) or not (3 <= len(gold) <= 60):
        return None

    documents = documents_of(summary)
    return {
        "retrieved_document": any(carries(doc.get("text"), gold) for doc in documents),
        "useful_span": any(carries(span, gold) for doc in documents for span in doc.get("useful_spans") or []),
        "classified_span": any(
            carries(_blob([item]), gold) for doc in documents for item in span_roles(doc)
        ),
        "semantic_fact": any(
            carries(_blob([fact]), gold) for doc in documents for fact in semantic_facts(doc)
        ),
        "direct_contract": any(carries(_blob(doc.get("direct_contracts")), gold) for doc in documents),
        "grounded_evidence": any(carries(item.get("text"), gold) for item in summary.get("evidence_items") or []),
        "unverified_reference": any(
            carries(item.get("text"), gold) for item in summary.get("unverified_references") or []
        ),
        "stage1_context": carries(stage1_context(summary), gold),
    }


def relation_support(task: dict) -> bool:
    """Does the gold sit somewhere that claims it answers the question?

    A fact's object, or a span the classifier called ANSWER_SUPPORT. Anything
    else is the word appearing in passing, which is how a passage about a castle
    being deserted was accepted as evidence for a scene heading.
    """

    gold = str(task.get("expected") or "")
    summary = task.get("search_summary") or {}
    for document in documents_of(summary):
        for fact in semantic_facts(document):
            if carries(fact.get("object"), gold):
                return True
        for item in span_roles(document):
            if str(item.get("role") or "").upper() == "ANSWER_SUPPORT" and carries(
                item.get("text") or item.get("span"), gold
            ):
                return True
    return False


def component_recall(task: dict) -> tuple[int, int]:
    """How many parts of a composed answer reached any retrieved document."""

    gold = str(task.get("expected") or "")
    parts = components(gold)
    corpus = " ".join(str(doc.get("text") or "") for doc in documents_of(task.get("search_summary") or {}))
    return sum(1 for part in parts if carries(corpus, part)), len(parts)


def delivery_trace(task: dict) -> dict[str, Any]:
    """Why the answer stopped: its document's rank, and what analysis it had.

    A gold document ranked 90th of 96 does not fail for the same reason as one
    ranked 8th, and the fix differs. `BestEffortReferenceSelector` orders on
    retrieval score alone, so the rank is what decided admission.
    """

    gold = str(task.get("expected") or "")
    summary = task.get("search_summary") or {}
    documents = documents_of(summary)
    scores = sorted((float(doc.get("retrieval_score") or 0.0) for doc in documents), reverse=True)
    for document in documents:
        if not carries(document.get("text"), gold):
            continue
        score = float(document.get("retrieval_score") or 0.0)
        roles = [str(item.get("role") or "") for item in span_roles(document)]
        return {
            "rank": sum(1 for value in scores if value > score) + 1,
            "documents": len(documents),
            "spans": len(document.get("useful_spans") or []),
            "roles": roles[:3],
            "facts": len(semantic_facts(document)),
            "duplicate": bool(document.get("duplicate")),
        }
    return {}


def reference_position(task: dict) -> int:
    gold = str(task.get("expected") or "")
    refs = (task.get("search_summary") or {}).get("unverified_references") or []
    for index, item in enumerate(refs, start=1):
        if carries(item.get("text"), gold):
            return index
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="An outputs/<log_name> directory.")
    parser.add_argument("--json", default="", help="Write the per-task trace to this path.")
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.run, "tasks", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            task = json.load(handle)
        seen = survival(task)
        if seen is None:
            continue
        found, total = component_recall(task)
        rows.append(
            {
                "task": os.path.basename(path)[:3],
                "gold": str(task.get("expected") or ""),
                "seen": seen,
                "reference_position": reference_position(task),
                "exact": bool(task.get("exact_match")),
                "relation_support": relation_support(task),
                "components_found": found,
                "components_total": total,
                "delivery": delivery_trace(task),
            }
        )

    header = "%-5s %-18s " % ("task", "gold") + " ".join("%-6s" % s[:6] for s in STAGES)
    print(header + " %5s %5s %6s %5s" % ("refN", "exact", "supp", "parts"))
    for row in rows:
        marks = " ".join("%-6s" % ("o" if row["seen"][stage] else ".") for stage in STAGES)
        print(
            "%-5s %-18s %s %5s %5s %6s %5s"
            % (
                row["task"],
                row["gold"][:18],
                marks,
                row["reference_position"] or "-",
                "O" if row["exact"] else "X",
                "O" if row["relation_support"] else ".",
                "%d/%d" % (row["components_found"], row["components_total"]),
            )
        )

    print("\n每個階段仍存活的題數（共 %d 題有檢索且 gold 可比對）" % len(rows))
    for stage in STAGES:
        print("  %-22s %3d" % (stage, sum(1 for row in rows if row["seen"][stage])))

    print("\n答案最後出現在哪個階段（即下一步在哪裡消失）")
    death: Counter = Counter()
    for row in rows:
        last = ""
        for stage in STAGES:
            if row["seen"][stage]:
                last = stage
        death[last or "(從未出現)"] += 1
    for stage, count in death.most_common():
        print("  %-22s %3d" % (stage, count))

    strings = sum(1 for row in rows if row["seen"]["retrieved_document"])
    supported = sum(1 for row in rows if row["relation_support"])
    print("\ngold 字串出現在文件中: %d   其中構成答案支持關係的: %d" % (strings, supported))
    print("  差額是字面命中但不構成證據的題目，不能當成召回成功")

    composed = [row for row in rows if row["components_total"] > 1]
    if composed:
        print("\n組合型答案（完整字串不會出現在任何頁面，改看元件召回）")
        for row in composed:
            print(
                "  %-5s %-30s 元件 %d/%d"
                % (row["task"], row["gold"][:30], row["components_found"], row["components_total"])
            )

    blocked = [row for row in rows if row["delivery"] and not row["seen"]["unverified_reference"]]
    if blocked:
        print("\ngold 文件存在但沒進參考池 —— 排序由 retrieval_score 決定")
        for row in blocked:
            info = row["delivery"]
            print(
                "  %-5s rank %3d/%-4d spans=%-2d facts=%-2d roles=%s"
                % (row["task"], info["rank"], info["documents"], info["spans"], info["facts"], info["roles"])
            )

    positions = sorted(row["reference_position"] for row in rows if row["reference_position"])
    if positions:
        print(
            "\ngold 所在的參考序位：%s   （預算通常只保留前 3-4 筆）"
            % ", ".join(str(position) for position in positions)
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print("\n[OK] %s" % args.json)


if __name__ == "__main__":
    main()
