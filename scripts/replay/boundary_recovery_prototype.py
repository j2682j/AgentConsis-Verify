"""Can a dependency parse recover the boundary a span was cut from?

Human annotation of the 38 fragmented spans put 25 of them in `noun_phrase` and
32 under a `plain` boundary -- no bracket or quote to key on. That points at
dependency parsing, but only as a hypothesis: a phrase a person calls a noun
phrase is not necessarily a span spaCy's `noun_chunks` will draw the same edges
around, and the point of this prototype is to find out which.

Three arms, deliberately without NER or bracket rules, so what is measured is
what dependency parsing alone contributes:

    A   the smallest `noun_chunk` containing the span
    B   the smallest contiguous subtree of the span's dependency head
    C   A, falling back to B when A finds nothing

Three populations, because recovery that only ever expands is not recovery:
the 38 fragments measure whether the right boundary is found, the 90 complete
spans measure whether correct ones are left alone, and the 5 unrelated spans
measure whether something that should have been dropped gets grown instead.

Every candidate has to be a verbatim, contiguous run of the canonical context
and has to contain the span it came from. Where the parser produces nothing the
span is returned unchanged rather than guessed at.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

BASE = "c:/SCP/outputs/query_span_analysis"


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def collapse(value: str) -> str:
    return re.sub(r"\s+", " ", nfc(value)).strip()


@dataclass
class GoldCase:
    annotation_id: str
    span_text: str
    gold_span: str
    context: str
    unit_type: str
    boundary_form: str
    repair_direction: str
    acceptable_alternative: str = ""
    gold_repair_source: str = "verbatim"
    span_offsets: tuple[int, int] | None = None
    gold_offsets: tuple[int, int] | None = None


def align_to_canonical(text: str, context: str) -> tuple[str, str]:
    """Recover an annotation typed against mojibake, without silent substitution.

    Two gold spans came back containing `Taish?` because the editor that saved
    the sheet destroyed `Taishō`. Replacing every non-ASCII character with `?`
    on both sides would make them match, and would also make any two spans
    differing only in accented characters match. Instead the damaged run is
    located by its intact neighbours and replaced with what the canonical text
    actually says, and the row records that this happened.
    """

    if collapse(text) in collapse(context):
        return text, "verbatim"
    words = collapse(text).split()
    clean = [w for w in words if "?" not in w]
    if len(clean) < 2:
        return text, "unaligned"
    head, tail = clean[0], clean[-1]
    pattern = re.escape(head) + r".{0,80}?" + re.escape(tail)
    match = re.search(pattern, collapse(context), re.DOTALL)
    if not match:
        return text, "unaligned"
    return match.group(0), "canonical_alignment"


def load_gold() -> list[GoldCase]:
    raw = open(f"{BASE}/boundary_gold_blind.csv", "rb").read().decode("cp1252")
    rows = list(csv.DictReader(io.StringIO(raw)))
    canonical = {
        r["annotation_id"]: r
        for r in csv.DictReader(
            open(f"{BASE}/_canonical/query_span_annotation_blind.csv", encoding="utf-8")
        )
    }
    cases: list[GoldCase] = []
    for row in rows:
        clean = canonical[row["annotation_id"]]
        context = collapse(clean["local_context"]) or collapse(clean["question"])
        gold, source = align_to_canonical(row["human_gold_span"], context)
        if source == "unaligned":
            gold, source = align_to_canonical(row["human_gold_span"], collapse(clean["question"]))
            if source != "unaligned":
                context = collapse(clean["question"])
        case = GoldCase(
            annotation_id=row["annotation_id"],
            span_text=collapse(clean["span_text"]),
            gold_span=collapse(gold),
            context=context,
            unit_type=row["unit_type"].strip(),
            boundary_form=row["boundary_form"].strip(),
            repair_direction=row["repair_direction"].strip(),
            acceptable_alternative=collapse(row.get("acceptable_alternative")),
            gold_repair_source=source,
        )
        span_at = case.context.casefold().find(case.span_text.casefold())
        gold_at = case.context.casefold().find(case.gold_span.casefold())
        if span_at >= 0:
            case.span_offsets = (span_at, span_at + len(case.span_text))
        if gold_at >= 0:
            case.gold_offsets = (gold_at, gold_at + len(case.gold_span))
        cases.append(case)
    return cases


def load_controls() -> list[tuple[str, str, str, str]]:
    controls = list(csv.DictReader(open(f"{BASE}/boundary_controls.csv", encoding="utf-8")))
    canonical = {
        r["annotation_id"]: r
        for r in csv.DictReader(
            open(f"{BASE}/_canonical/query_span_annotation_blind.csv", encoding="utf-8")
        )
    }
    out = []
    for row in controls:
        clean = canonical[row["annotation_id"]]
        out.append(
            (
                row["annotation_id"],
                collapse(clean["span_text"]),
                collapse(clean["local_context"]) or collapse(clean["question"]),
                row["human_boundary"],
            )
        )
    return out


class Recovery:
    """Arms A, B and C over one loaded parser."""

    def __init__(self) -> None:
        import spacy

        self.nlp = spacy.load("en_core_web_md")

    def _span_char_range(self, doc, span_text: str) -> tuple[int, int] | None:
        at = doc.text.casefold().find(span_text.casefold())
        return (at, at + len(span_text)) if at >= 0 else None

    def arm_a(self, doc, span_text: str) -> str:
        """Smallest noun chunk containing the span."""

        window = self._span_char_range(doc, span_text)
        if not window:
            return ""
        start, end = window
        containing = [
            chunk
            for chunk in doc.noun_chunks
            if chunk.start_char <= start and chunk.end_char >= end
        ]
        if not containing:
            return ""
        return min(containing, key=lambda c: c.end_char - c.start_char).text

    def arm_b(self, doc, span_text: str) -> str:
        """Smallest contiguous subtree of the span's dependency head."""

        window = self._span_char_range(doc, span_text)
        if not window:
            return ""
        start, end = window
        inside = [t for t in doc if t.idx >= start and t.idx < end]
        if not inside:
            return ""
        head = inside[0]
        # Compared by token index, not identity. `Token.head` builds a fresh
        # wrapper each access, so `token.head is token` is False even at the
        # root -- the loop it was meant to terminate ran forever instead.
        while head.head.i != head.i:
            parent = head.head
            subtree = sorted(parent.subtree, key=lambda t: t.i)
            first, last = subtree[0], subtree[-1]
            if not (first.idx <= start and last.idx + len(last.text) >= end):
                break
            if len(subtree) > len(inside) + 12:
                break
            head = parent
        subtree = sorted(head.subtree, key=lambda t: t.idx)
        if not subtree:
            return ""
        first, last = subtree[0], subtree[-1]
        text = doc.text[first.idx : last.idx + len(last.text)]
        return text if start >= first.idx and end <= last.idx + len(last.text) else ""

    def arm_c(self, doc, span_text: str) -> str:
        return self.arm_a(doc, span_text) or self.arm_b(doc, span_text)


def tokens(value: str) -> list[str]:
    return [t for t in re.split(r"[\s]+", collapse(value).casefold().strip(" .,;:?\"'")) if t]


def token_prf(predicted: str, gold: str) -> tuple[float, float, float]:
    p_tokens, g_tokens = Counter(tokens(predicted)), Counter(tokens(gold))
    overlap = sum((p_tokens & g_tokens).values())
    precision = overlap / max(sum(p_tokens.values()), 1)
    recall = overlap / max(sum(g_tokens.values()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate() -> None:
    gold_cases = load_gold()
    controls = load_controls()
    recovery = Recovery()

    print(f"gold {len(gold_cases)} 筆")
    print(f"  grounding: {dict(Counter(c.gold_repair_source for c in gold_cases))}")
    print(f"  有 offsets: span {sum(1 for c in gold_cases if c.span_offsets)}"
          f"、gold {sum(1 for c in gold_cases if c.gold_offsets)}\n")

    docs = {c.annotation_id: recovery.nlp(c.context) for c in gold_cases}
    arms = {"A": recovery.arm_a, "B": recovery.arm_b, "C": recovery.arm_c}
    by_unit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for name, fn in arms.items():
        exact = alt = grounded = contains = 0
        under = over = parser_miss = 0
        precisions, recalls, f1s = [], [], []
        for case in gold_cases:
            produced = collapse(fn(docs[case.annotation_id], case.span_text))
            if not produced:
                parser_miss += 1
                produced = case.span_text
            if produced.casefold() in case.context.casefold():
                grounded += 1
            if case.span_text.casefold() in produced.casefold():
                contains += 1
            hit = produced.casefold() == case.gold_span.casefold()
            exact += hit
            by_unit[case.unit_type][name] += hit
            by_unit[case.unit_type]["n"] = by_unit[case.unit_type].get("n", 0)
            if case.acceptable_alternative and produced.casefold() == case.acceptable_alternative.casefold():
                alt += 1
            p, r, f = token_prf(produced, case.gold_span)
            precisions.append(p); recalls.append(r); f1s.append(f)
            if not hit:
                if len(tokens(produced)) < len(tokens(case.gold_span)):
                    under += 1
                elif len(tokens(produced)) > len(tokens(case.gold_span)):
                    over += 1
        n = len(gold_cases)
        print(f"=== Arm {name}")
        print(f"    gold exact match     {exact}/{n} = {exact/n:.3f}")
        print(f"    acceptable alt match {alt}")
        print(f"    token P/R/F1         {sum(precisions)/n:.3f} / {sum(recalls)/n:.3f} / {sum(f1s)/n:.3f}")
        print(f"    under / over         {under} / {over}")
        print(f"    parser 無輸出         {parser_miss}")
        print(f"    grounded / 含原 span  {grounded}/{n} / {contains}/{n}")

    print(f"\n=== 對照組（不該被改動）")
    control_docs = {}
    for annotation_id, span_text, context, kind in controls:
        control_docs[annotation_id] = (recovery.nlp(context), span_text, kind)
    for name, fn in arms.items():
        mutated = Counter()
        for annotation_id, (doc, span_text, kind) in control_docs.items():
            produced = collapse(fn(doc, span_text))
            if produced and produced.casefold() != span_text.casefold():
                mutated[kind] += 1
        complete_n = sum(1 for _, _, _, k in controls if k == "complete")
        unrelated_n = sum(1 for _, _, _, k in controls if k == "unrelated")
        print(f"    Arm {name}: complete 被改動 {mutated['complete']}/{complete_n}"
              f"、unrelated 被擴張 {mutated['unrelated']}/{unrelated_n}")

    print(f"\n=== 依 unit_type 的 exact match")
    counts = Counter(c.unit_type for c in gold_cases)
    print(f"{'unit_type':<14}{'n':>4}" + "".join(f"{'Arm '+a:>8}" for a in arms))
    for unit, total in counts.most_common():
        print(f"{unit:<14}{total:>4}" + "".join(f"{by_unit[unit][a]:>8}" for a in arms))


if __name__ == "__main__":
    sys.exit(evaluate())
