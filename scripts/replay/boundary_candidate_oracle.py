"""Is the correct boundary even among the candidates a selector could pick from?

The three recovery arms each committed to one reading of the span and reached
23.7% exact match at best. That measures those three readings, not what is
reachable: a selector cannot choose a boundary no generator proposes, and a
detector that perfectly identifies which spans need repair still leaves 29 of 38
unrepaired if the right span is never on the list.

So this asks the prior question. Every generator contributes candidates, the
union is scored against the human gold, and the ceiling that produces is what
any downstream selector is bounded by. If the union is low, more selector work
is wasted; if it is high, the problem reduces to choosing among candidates,
which is one decision rather than two.

Every candidate is a contiguous verbatim run of the canonical context, addressed
by character offset rather than by copied text. The first version of this file
built candidates by joining whitespace tokens and then ran `.strip(" ,.;:")`
over the results, which attached punctuation during generation and removed it
during filtering; six gold spans were unreachable as a direct consequence. See
`boundary_candidates` for why the repair is more candidates rather than a looser
comparison. Matching stays exact.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

sys.path.insert(0, r"c:/SCP")

from scripts.replay.boundary_candidates import Candidate, finalise, merge
from scripts.replay.boundary_recovery_prototype import (
    Recovery,
    collapse,
    load_controls,
    load_gold,
)

BRACKETED = re.compile(
    r"\(([^()]{1,120})\)|\"([^\"]{1,120})\"|\u201c([^\u201d]{1,120})\u201d"
)
#: Raised from 6 after the first oracle run: three gold units were 8 to 11
#: tokens long and could not be reached. Derived from the failures it is being
#: measured on, so it is a development choice, not a value shown to generalise.
MAX_EXPANSION_TOKENS = 12

#: An overlap made only of these is not an overlap worth keeping -- a candidate
#: sharing nothing but `the` with the original span is a different phrase.
WEAK_TOKENS = frozenset(
    {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is",
     "was", "that", "this", "with", "from", "by", "s", "'s"}
)

Span = tuple[int, int]


def word_spans(context: str, lo: int = 0, hi: int | None = None) -> list[Span]:
    """Whitespace-delimited runs, as offsets rather than as copied strings.

    Offsets are what makes `page?` and `page` both expressible: the token run
    ends where it ends, and dropping the question mark is a second candidate
    rather than an edit applied to the first.
    """

    hi = len(context) if hi is None else hi
    return [m.span() for m in re.finditer(r"\S+", context) if m.start() >= lo and m.end() <= hi]


def _range(text: str, span: str) -> Span | None:
    at = text.casefold().find(span.casefold())
    return (at, at + len(span)) if at >= 0 else None


def _substantive(text: str) -> set[str]:
    return {
        t for t in re.split(r"[^\w']+", text.casefold())
        if t and t not in WEAK_TOKENS
    }


def keep(context: str, window: Span) -> list[Span]:
    return [window]


def noun_chunks(doc, window: Span) -> list[Span]:
    start, end = window
    return [
        (chunk.start_char, chunk.end_char)
        for chunk in doc.noun_chunks
        if chunk.start_char <= start and chunk.end_char >= end
    ]


def subtrees(doc, window: Span) -> list[Span]:
    """Every ancestor's subtree that still contains the span, not just one."""

    start, end = window
    inside = [t for t in doc if t.idx >= start and t.idx < end]
    if not inside:
        return []
    out: list[Span] = []
    node = inside[0]
    seen_indices: set[int] = set()
    while node.i not in seen_indices:
        seen_indices.add(node.i)
        ordered = sorted(node.subtree, key=lambda t: t.i)
        if ordered:
            first, last = ordered[0], ordered[-1]
            if first.idx <= start and last.idx + len(last.text) >= end:
                out.append((first.idx, last.idx + len(last.text)))
        if node.head.i == node.i:
            break
        node = node.head
    return out


def entities(doc, window: Span) -> list[Span]:
    start, end = window
    return [
        (ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.start_char <= end and ent.end_char >= start
    ]


def bracket_units(context: str, window: Span) -> list[Span]:
    start, end = window
    out: list[Span] = []
    for match in BRACKETED.finditer(context):
        for group in range(1, 4):
            if match.group(group) is not None:
                out.append(match.span(group))
    cut = 0
    for delimiter in re.finditer(r"[,;:?]", context):
        out.append((cut, delimiter.start()))
        cut = delimiter.end()
    out.append((cut, len(context)))
    return [(a, b) for a, b in out if a <= start and b >= end and a < b]


def token_expansions(context: str, window: Span) -> list[Span]:
    """Bounded left, right and both-sided growth, in whole tokens.

    Growth moves an offset onto a token edge; whether the resulting edge carries
    punctuation is not decided here. Both readings reach the candidate set.
    """

    start, end = window
    left = word_spans(context, hi=start)
    right = word_spans(context, lo=end)
    out: list[Span] = []
    for back in range(0, min(MAX_EXPANSION_TOKENS, len(left)) + 1):
        new_start = left[len(left) - back][0] if back else start
        for forward in range(0, min(MAX_EXPANSION_TOKENS, len(right)) + 1):
            if back == 0 and forward == 0:
                continue
            new_end = right[forward - 1][1] if forward else end
            out.append((new_start, new_end))
    return out


def contraction(context: str, window: Span) -> list[Span]:
    """Shorter runs inside the span itself.

    Two gold units are *narrower* than the span they came from -- `Emily
    Midkiff's` should have been `Emily Midkiff` -- and every generator above
    requires the candidate to contain the span, which rules those out by
    construction rather than by evidence. The possessive case is not reachable
    by dropping whole tokens either; `boundary_candidates` handles that edge.
    """

    start, end = window
    inner = word_spans(context, start, end)
    out: list[Span] = []
    for first in range(len(inner)):
        for last in range(first + 1, len(inner) + 1):
            if last - first == len(inner):
                continue
            a, b = inner[first][0], inner[last - 1][1]
            if _substantive(context[a:b]):
                out.append((a, b))
    return out


def overlap_replacement(context: str, window: Span) -> list[Span]:
    """Windows that shed part of the span on one side and grow on the other.

    `Taishō Tamai's` needs to become `Tamai's number`: the left edge moves in
    while the right edge moves out, which neither expansion nor contraction can
    produce on its own. At least one substantive token of the original has to
    survive, or the result is simply a different phrase.
    """

    start, end = window
    inner = word_spans(context, start, end)
    after = word_spans(context, lo=end)
    original = _substantive(context[start:end])
    out: list[Span] = []
    for drop_left in range(1, min(len(inner), 4)):
        new_start = inner[drop_left][0]
        for forward in range(1, min(7, len(after) + 1)):
            new_end = after[forward - 1][1]
            if _substantive(context[new_start:new_end]) & original:
                out.append((new_start, new_end))
    return out


def sentence_or_clause(doc, window: Span) -> list[Span]:
    """The whole sentence, for instructions that only make sense entire.

    These end in a full stop, and that stop is part of one gold span. It stays
    on because nothing downstream rewrites a candidate.
    """

    start, end = window
    return [
        (sentence.start_char, sentence.end_char)
        for sentence in doc.sents
        if sentence.start_char <= start and sentence.end_char >= end
    ]


def hyphen_aware(context: str, window: Span) -> list[Span]:
    """Runs that keep a hyphenated compound intact.

    `Polish` sits inside `Polish-language`, and every token-based expansion
    treats the hyphen as a boundary, so `Polish-language version` was
    unreachable.
    """

    start, end = window
    out: list[Span] = []
    for match in re.finditer(r"[\w']+(?:-[\w']+)+", context):
        if not (match.start() <= start and match.end() >= end):
            if not (start <= match.start() and end >= match.end()):
                continue
        out.append(match.span())
        tail = word_spans(context, lo=match.end())
        for forward in range(1, min(4, len(tail) + 1)):
            out.append((match.start(), tail[forward - 1][1]))
    return out


GENERATORS = (
    "keep", "noun_chunk", "subtree", "ner", "bracket", "expansion",
    "contraction", "overlap_replacement", "sentence", "hyphen",
)


def candidates_for(doc, context: str, span: str) -> list[Candidate]:
    """Every generator's offsets, deduplicated by boundary.

    There is no filtering pass. Each generator's contract is enforced where the
    offsets are produced, so nothing here has to reconstruct intent from a
    string -- which is what the old `.strip()` filter was doing when it silently
    rewrote the one gold span that needed its full stop.
    """

    window = _range(context, span)
    if not window:
        return []
    produced: dict[str, list[Span]] = {
        "keep": keep(context, window),
        "noun_chunk": noun_chunks(doc, window),
        "subtree": subtrees(doc, window),
        "ner": entities(doc, window),
        "bracket": bracket_units(context, window),
        "expansion": token_expansions(context, window),
        "contraction": contraction(context, window),
        "overlap_replacement": overlap_replacement(context, window),
        "sentence": sentence_or_clause(doc, window),
        "hyphen": hyphen_aware(context, window),
    }
    collected: dict[Span, set[str]] = {}
    for name, spans in produced.items():
        merge(collected, name, context, spans)
    return finalise(collected, context)


def by_generator(candidates: list[Candidate], context: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for name in candidate.generators:
            out[name].add(candidate.text(context).casefold())
    return out


def main() -> None:
    gold_cases = load_gold()
    controls = load_controls()
    recovery = Recovery()

    hits: dict[str, int] = defaultdict(int)
    union_hits = 0
    alt_hits = 0
    sizes: list[int] = []
    by_unit: dict[str, list[int]] = defaultdict(list)
    missing: list[tuple[str, str, str]] = []
    unique_credit: dict[str, int] = defaultdict(int)

    for case in gold_cases:
        doc = recovery.nlp(case.context)
        candidates = candidates_for(doc, case.context, case.span_text)
        sources = by_generator(candidates, case.context)
        target = collapse(case.gold_span).casefold()
        alternative = collapse(case.acceptable_alternative).casefold()
        for name, items in sources.items():
            if target in items:
                hits[name] += 1
        union = {c.text(case.context).casefold() for c in candidates}
        sizes.append(len(union))
        found = target in union
        union_hits += found
        if found:
            owners = [n for n in GENERATORS if target in sources.get(n, set())]
            if len(owners) == 1:
                unique_credit[owners[0]] += 1
        if alternative and alternative in union:
            alt_hits += 1
        by_unit[case.unit_type].append(int(found))
        if not found:
            missing.append((case.annotation_id, case.span_text, case.gold_span))

    total = len(gold_cases)
    ordered_sizes = sorted(sizes)
    print(f"fragmented {total} 筆\n")
    print(f"{'generator':<20}{'oracle recall':>16}{'唯一命中':>10}")
    for name in GENERATORS:
        print(f"{name:<20}{hits[name]}/{total} = {hits[name]/total:>8.3f}"
              f"{unique_credit[name]:>10}")
    print(f"{'union':<20}{union_hits}/{total} = {union_hits/total:>8.3f}")
    print(f"{'（含替代解）':<18}{union_hits + alt_hits}/{total}")
    print(f"\n候選數量: 平均 {sum(sizes)/total:.1f}、"
          f"中位數 {ordered_sizes[total//2]}、"
          f"P95 {ordered_sizes[int(total*0.95)-1]}、最大 {ordered_sizes[-1]}")

    print(f"\n依 unit_type 的 union oracle recall:")
    for unit, flags in sorted(by_unit.items(), key=lambda kv: -len(kv[1])):
        print(f"   {unit:<14} {sum(flags)}/{len(flags)} = {sum(flags)/len(flags):.3f}")

    print(f"\n候選集合未涵蓋 gold 的 {len(missing)} 筆:")
    for annotation_id, span, gold in missing[:12]:
        print(f"   {annotation_id}  span={span[:26]!r}")
        print(f"        gold={gold[:60]!r}")

    keep_ok = 0
    for annotation_id, span, context, kind in controls:
        doc = recovery.nlp(context)
        candidates = candidates_for(doc, context, span)
        sources = by_generator(candidates, context)
        if span.casefold() in sources.get("keep", set()):
            keep_ok += 1
    print(f"\n對照組 {len(controls)} 筆: KEEP 候選存在 {keep_ok}/{len(controls)}")
    print("DROP 為 action，不需候選；5 筆 unrelated 皆可用")


if __name__ == "__main__":
    sys.exit(main())
