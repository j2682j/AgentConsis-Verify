"""The singular of a plural answer, when the corpus writes it that way.

Corpus attestation reserves a candidate the fetched pages never state. It
compares exact surface forms, so task 034's `Rockhopper penguins` counted zero
while page-005-000 says `rockhopper penguin` at characters 496-514 -- the same
species, singular. A three-run specific answer was reserved and a one-run
generic one won.

One direction only. Attestation asks whether the corpus states the candidate,
and the case that arises is a plural candidate against a corpus using the
singular. Generating plurals as well adds no coverage and invents surfaces that
match by accident: `R` becomes `Rs` and matches an unrelated token, `class`
becomes `classs`, `Saint Petersburg` becomes `Saint Petersburgs`.

Not a stemmer. Only the final word's inflection changes and every preceding word
survives, so `Rockhopper penguins` can reach `Rockhopper penguin` and never
`penguin` -- dropping the modifier would merge a species with its genus, which
is the mistake this exists to avoid.
"""

from __future__ import annotations

#: Below this a plural is not worth reducing: `Rs` matches unrelated tokens.
MIN_INFLECTED_WORD_CHARS = 4

_ES_STEMS = ("s", "x", "z", "ch", "sh")


def singular_variants(answer: str) -> list[str]:
    """`["Rockhopper penguin"]` for `"Rockhopper penguins"`, else `[]`."""

    text = str(answer or "").strip()
    if not text:
        return []
    words = text.split()
    last = words[-1]
    lowered = last.casefold()
    if len(last) < MIN_INFLECTED_WORD_CHARS or not lowered.endswith("s"):
        return []
    if lowered.endswith("ss"):
        return []
    if lowered.endswith("es") and lowered[:-2].endswith(_ES_STEMS):
        singular = last[:-2]
    else:
        singular = last[:-1]
    variant = " ".join(words[:-1] + [singular])
    return [variant] if variant.casefold() != text.casefold() else []


__all__ = ["MIN_INFLECTED_WORD_CHARS", "singular_variants"]
