"""Keep the numbers a question asked about intact on their way into a query.

Asked how many albums Mercedes Sosa released between 2000 and 2009, the query
model returned `... between 2000 and 2:009`. The colon is the model's own: the
same reply carried `2009` correctly in its relation goal, so nothing downstream
mangled it. A search for `2:009` does not constrain the year at all, and the
task is lost before any document is fetched.

The narrow fix would be to delete stray colons. That would leave every other
shape of the same fault -- a comma in a year, a stray full stop, a dropped
separator -- to be discovered one benchmark at a time. So the rule is stated
over what must be preserved rather than over what went wrong: a number, year or
date written in the question is a protected literal, and a query that means to
carry it must carry it exactly.

Two limits keep this from becoming a rewriter. Only damage that aligns to
exactly one protected literal is repaired, so an ambiguous token is left as the
model wrote it. And nothing is inserted: a query that simply omits a number is
not corrected, because deciding which numbers a query *should* mention is query
planning, not integrity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: A run of digits with the separators that legitimately appear inside numbers,
#: dates and ranges. Bounded by digits so trailing punctuation stays outside.
NUMERIC_RUN = re.compile(r"\d(?:[\d,./:\-–—]*\d)?")

#: Separators a model has been seen to insert or drop. Comparison happens on
#: digits alone, so this is only used to describe what changed.
SEPARATORS = ",./:-–—"


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


@dataclass(frozen=True)
class LiteralRepair:
    """One replacement, kept so the change can be audited rather than trusted."""

    before: str
    after: str
    reason: str


@dataclass
class RepairedQuery:
    """The query as written and as corrected, never one without the other."""

    raw: str
    repaired: str
    repairs: tuple[LiteralRepair, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return self.raw != self.repaired

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw,
            "repaired_query": self.repaired,
            "repairs": [
                {"before": r.before, "after": r.after, "reason": r.reason}
                for r in self.repairs
            ],
        }


def protected_literals(question: str) -> set[str]:
    """Every number, year, date and range the question actually contains."""

    text = unicodedata.normalize("NFC", str(question or ""))
    return {match.group(0) for match in NUMERIC_RUN.finditer(text) if _digits(match.group(0))}


def repair_query(query: str, question: str) -> RepairedQuery:
    """Restore protected literals the query damaged, and nothing else.

    A token is only touched when its digits match exactly one protected literal
    and its written form differs. `2:009` becomes `2009` because the question
    says `2009` and no other literal has those digits; `2000-2009` is left alone
    because its digits match no single literal, which is the correct outcome for
    a range the model assembled itself.
    """

    raw = str(query or "")
    literals = protected_literals(question)
    if not literals or not raw:
        return RepairedQuery(raw=raw, repaired=raw)

    by_digits: dict[str, set[str]] = {}
    for literal in literals:
        by_digits.setdefault(_digits(literal), set()).add(literal)

    repairs: list[LiteralRepair] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in literals:
            return token
        candidates = by_digits.get(_digits(token), set())
        # Exactly one, or the repair would be a guess between two readings.
        if len(candidates) != 1:
            return token
        target = next(iter(candidates))
        if target == token:
            return token
        reason = (
            "separator_inserted"
            if _digits(token) == _digits(target) and len(token) > len(target)
            else "separator_dropped"
            if len(token) < len(target)
            else "separator_changed"
        )
        repairs.append(LiteralRepair(before=token, after=target, reason=reason))
        return target

    repaired = NUMERIC_RUN.sub(replace, raw)
    return RepairedQuery(raw=raw, repaired=repaired, repairs=tuple(repairs))


__all__ = [
    "LiteralRepair",
    "NUMERIC_RUN",
    "RepairedQuery",
    "protected_literals",
    "repair_query",
]
