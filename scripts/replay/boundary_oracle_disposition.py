"""Before and after, per case, for the six gold spans the first oracle missed.

The repair is only worth what it is claimed to be worth if each of the six is
accounted for individually. A union recall that moves from 0.842 to 0.974 could
in principle come from five of the six plus one unrelated case flipping, so this
replays the old string-based path alongside the offset-based one and prints the
disposition of every case that changed in either direction.

The legacy path is reproduced here rather than kept in the live oracle: it is
the thing being measured against, and leaving it importable would invite it back
into use.
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, r"c:/SCP")

from scripts.replay.boundary_candidate_oracle import (
    MAX_EXPANSION_TOKENS,
    WEAK_TOKENS,
    _substantive,
    candidates_for,
)
from scripts.replay.boundary_recovery_prototype import Recovery, collapse, load_gold

BRACKETED = re.compile(
    r"\(([^()]{1,120})\)|\"([^\"]{1,120})\"|\u201c([^\u201d]{1,120})\u201d"
)


def _range(text: str, span: str) -> tuple[int, int] | None:
    at = text.casefold().find(span.casefold())
    return (at, at + len(span)) if at >= 0 else None


def legacy_union(doc, context: str, span: str) -> set[str]:
    """The pre-fix candidate set: whitespace joins, then a rewriting filter."""

    produced: dict[str, list[str]] = {"keep": [span]}

    window = _range(doc.text, span)
    if window:
        start, end = window
        produced["noun_chunk"] = [
            c.text for c in doc.noun_chunks if c.start_char <= start and c.end_char >= end
        ]
        produced["ner"] = [
            e.text for e in doc.ents if e.start_char <= end and e.end_char >= start
        ]
        inside = [t for t in doc if t.idx >= start and t.idx < end]
        subtree_out: list[str] = []
        if inside:
            node, seen = inside[0], set()
            while node.i not in seen:
                seen.add(node.i)
                ordered = sorted(node.subtree, key=lambda t: t.i)
                if ordered:
                    first, last = ordered[0], ordered[-1]
                    if first.idx <= start and last.idx + len(last.text) >= end:
                        subtree_out.append(doc.text[first.idx : last.idx + len(last.text)])
                if node.head.i == node.i:
                    break
                node = node.head
        produced["subtree"] = subtree_out

        left_tokens, right_tokens = context[:start].split(), context[end:].split()
        span_tokens = span.split()
        expansion: list[str] = []
        for back in range(0, MAX_EXPANSION_TOKENS + 1):
            for forward in range(0, MAX_EXPANSION_TOKENS + 1):
                if back == 0 and forward == 0:
                    continue
                unit = " ".join(
                    left_tokens[len(left_tokens) - back :] + span_tokens + right_tokens[:forward]
                ).strip()
                if unit and unit.casefold() in context.casefold():
                    expansion.append(unit)
        produced["expansion"] = expansion
    else:
        produced.update({"noun_chunk": [], "ner": [], "subtree": [], "expansion": []})

    bracket: list[str] = []
    for match in BRACKETED.finditer(context):
        unit = next((g for g in match.groups() if g), "")
        if unit and span.casefold() in unit.casefold():
            bracket.append(unit.strip())
    for part in re.split(r"[,;:?]", context):
        if span.casefold() in part.casefold() and part.strip():
            bracket.append(part.strip())
    produced["bracket"] = bracket

    words = span.split()
    produced["contraction"] = [
        " ".join(words[a:b])
        for a in range(len(words))
        for b in range(a + 1, len(words) + 1)
        if b - a != len(words) and _substantive(" ".join(words[a:b]))
    ]

    overlap: list[str] = []
    span_words, span_tokens_set = span.split(), _substantive(span)
    for drop_left in range(1, min(len(span_words), 4)):
        head = " ".join(span_words[drop_left:])
        at = collapse(context).casefold().find(head.casefold()) if head else -1
        if at < 0:
            continue
        tail = collapse(context)[at + len(head) :].split()
        for forward in range(1, 7):
            unit = " ".join([head] + tail[:forward]).strip()
            if unit and _substantive(unit) & span_tokens_set:
                overlap.append(unit)
    produced["overlap_replacement"] = overlap

    produced["sentence"] = [s.text for s in doc.sents if span.casefold() in s.text.casefold()]

    hyphen: list[str] = []
    for match in re.finditer(r"[\w']+(?:-[\w']+)+", context):
        if span.casefold() in match.group(0).casefold():
            hyphen.append(match.group(0))
            tail = context[match.end() :].split()
            for forward in range(1, 4):
                hyphen.append(" ".join([match.group(0)] + tail[:forward]))
    produced["hyphen"] = hyphen

    exempt = {"contraction", "overlap_replacement", "hyphen"}
    union: set[str] = set()
    for name, items in produced.items():
        for item in items:
            value = collapse(item).strip(" ,.;:")   # the rewriting filter
            key = value.casefold()
            if not value or key not in collapse(context).casefold():
                continue
            if name not in exempt and span.casefold() not in key:
                continue
            if name in exempt and not (_substantive(value) & _substantive(span)):
                continue
            union.add(key)
    return union


def main() -> None:
    recovery = Recovery()
    rows = []
    for case in load_gold():
        doc = recovery.nlp(case.context)
        target = collapse(case.gold_span).casefold()
        before = target in legacy_union(doc, case.context, case.span_text)
        after = target in {
            c.text(case.context).casefold()
            for c in candidates_for(doc, case.context, case.span_text)
        }
        rows.append((case, before, after))

    gained = [r for r in rows if not r[1] and r[2]]
    lost = [r for r in rows if r[1] and not r[2]]
    still = [r for r in rows if not r[1] and not r[2]]

    print(f"before {sum(1 for r in rows if r[1])}/{len(rows)}"
          f"  ->  after {sum(1 for r in rows if r[2])}/{len(rows)}\n")
    print(f"新命中 {len(gained)} 筆:")
    for case, _, _ in gained:
        print(f"   {case.annotation_id}  span={case.span_text[:30]!r}")
        print(f"        gold={case.gold_span[:64]!r}")
    print(f"\n退步 {len(lost)} 筆:")
    for case, _, _ in lost:
        print(f"   {case.annotation_id}  span={case.span_text[:30]!r} gold={case.gold_span[:50]!r}")
    print(f"\n仍未命中 {len(still)} 筆:")
    for case, _, _ in still:
        print(f"   {case.annotation_id}  span={case.span_text[:30]!r}")
        print(f"        gold={case.gold_span[:70]!r}")


if __name__ == "__main__":
    sys.exit(main())
