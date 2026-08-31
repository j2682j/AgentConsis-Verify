"""Candidate boundaries as character offsets into the canonical context.

The first oracle missed six gold spans, and none of them needed a new strategy.
Three failed because expansion grew by whitespace tokens, so `page?` came back
where the gold said `page`; one failed the opposite way, because the filter ran
`.strip(" ,.;:")` and removed the sentence-final full stop the gold contained.
The same punctuation was glued on at generation and shaved off at filtering.

Making the oracle tolerant of punctuation would hide that rather than fix it,
and would let any candidate differing only in trailing marks count as exact. So
candidates are offsets instead, and a run whose edge sits on punctuation yields
*both* real substrings -- `Legume Wikipedia page` and `Legume Wikipedia page?`
are separate candidates because both occur in the text. Nothing is rewritten,
and `context[start:end]` reproduces every candidate exactly.

The sixth failure is a different fault: `Emily Midkiff's` should have become
`Emily Midkiff`, and splitting on whitespace cannot express that boundary. It is
handled by character offset, leaving names like `O'Connor` intact.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

#: Marks that may sit on a boundary without belonging to the unit. Both readings
#: are kept, so `page?` and `page` are each a candidate.
BOUNDARY_PUNCTUATION = ",.;:?!\"'“”‘’()[]"

#: Whitespace is different. A leading space is not a second reading of where the
#: unit begins, it is an offset taken from the wrong side -- `bracket_units`
#: splits on `[,;:?]`, so every following segment starts on the space after the
#: delimiter. Whitespace edges are pulled in rather than kept alongside, or the
#: set fills with pairs whose only difference is a space no gold span contains.
WHITESPACE = string.whitespace

#: `'s`, `’s`, and the bare apostrophe of a plural possessive. Anchored to the
#: end of the span, which is what keeps `O'Connor` out of it while still
#: reducing `O'Connor's` to `O'Connor`.
POSSESSIVE = re.compile(r"(?:['’]s|['’])$", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    """One boundary, addressed by offset so its text cannot drift."""

    start: int
    end: int
    generators: tuple[str, ...] = field(default_factory=tuple)

    def text(self, context: str) -> str:
        return context[self.start : self.end]


def _trim(context: str, start: int, end: int, marks: str) -> tuple[int, int]:
    start = max(0, min(start, len(context)))
    end = max(0, min(end, len(context)))
    while start < end and context[start] in marks:
        start += 1
    while end > start and context[end - 1] in marks:
        end -= 1
    return start, end


def trim_whitespace(context: str, start: int, end: int) -> tuple[int, int]:
    """Pull the edges in off whitespace only."""

    return _trim(context, start, end, WHITESPACE)


def trim_boundary(context: str, start: int, end: int) -> tuple[int, int]:
    """Pull the edges in off punctuation and whitespace, without rewriting."""

    return _trim(context, start, end, BOUNDARY_PUNCTUATION + WHITESPACE)


def possessive_trim(context: str, start: int, end: int) -> tuple[int, int] | None:
    """Drop a possessive ending, unless the apostrophe is inside a name.

    `Emily Midkiff's` -> `Emily Midkiff`. Names keep their own apostrophes for
    free: the pattern is anchored to the end of the span, so `O'Connor` does not
    match it, while `O'Connor's` does and correctly leaves `O'Connor` behind.
    """

    text = context[start:end]
    match = POSSESSIVE.search(text)
    if not match:
        return None
    head = text[: match.start()]
    if not head.strip():
        return None
    return start, start + len(head)


def with_punctuation_variants(
    context: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Each span, plus the same span with boundary punctuation removed.

    Both are genuine substrings of the context, so a gold answer that keeps its
    full stop and one that drops its question mark are each reachable without
    the comparison being loosened for either.
    """

    out: list[tuple[int, int]] = []
    for raw_start, raw_end in spans:
        if raw_start >= raw_end or raw_start < 0 or raw_end > len(context):
            continue
        start, end = trim_whitespace(context, raw_start, raw_end)
        if start >= end:
            continue
        out.append((start, end))
        trimmed = trim_boundary(context, start, end)
        if trimmed != (start, end) and trimmed[0] < trimmed[1]:
            out.append(trimmed)
        possessive = possessive_trim(context, *trimmed)
        if possessive and possessive[0] < possessive[1]:
            out.append(possessive)
    return out


def merge(
    collected: dict[tuple[int, int], set[str]],
    generator: str,
    context: str,
    spans: list[tuple[int, int]],
) -> None:
    """Record spans under one generator, deduplicating by offset.

    Provenance is a set: the same boundary found by several generators is one
    candidate with several sources, which is what a later ablation needs in
    order to tell a generator that adds coverage from one that only agrees.
    """

    for start, end in with_punctuation_variants(context, spans):
        if 0 <= start < end <= len(context):
            collected.setdefault((start, end), set()).add(generator)


def finalise(
    collected: dict[tuple[int, int], set[str]], context: str
) -> list[Candidate]:
    return [
        Candidate(start=start, end=end, generators=tuple(sorted(sources)))
        for (start, end), sources in sorted(collected.items())
        if context[start:end].strip()
    ]


__all__ = [
    "BOUNDARY_PUNCTUATION",
    "WHITESPACE",
    "Candidate",
    "finalise",
    "merge",
    "possessive_trim",
    "trim_boundary",
    "trim_whitespace",
    "with_punctuation_variants",
]
