"""Which tasks the gold-string funnel is computed over, fixed before any run.

Two lists, because the classification is genuinely contested and the sample is
too small for the argument to be settled by argument. Three tasks separate them:
`014` (a country identified from a flag), `040` (cities enumerated then compared
by longitude) and `050` (a minimum with an alphabetical tie-break). Each is a
defensible call in either direction.

Reporting both costs nothing at analysis time and answers the question the debate
was really about. If the two denominators agree, the classification does not
matter and the argument was moot. If they disagree, that is a finding -- the
result depends on how eligibility is drawn -- and it gets reported as one rather
than resolved by whoever writes the paper.

What matters is that membership is fixed now. Moving a task after seeing which
way it went is how a sensitivity analysis becomes a way of choosing an answer.
"""

from __future__ import annotations

#: Answers that appear verbatim in a single end document, checkable by searching
#: for their text. Multi-hop tasks stay in: needing three lookups to reach a page
#: says nothing about whether the answer is written on it, and excluding them
#: would drop the denominator to nine and end the experiment.
STRICT = (
    "006", "007", "013", "016", "018", "019", "020", "027", "029",
    "032", "033", "034", "039", "042", "046", "048", "051", "053",
)

#: The three contested tasks, each arguably derived or non-textual rather than
#: retrieved as a string.
CONTESTED = ("014", "040", "050")

LENIENT = tuple(sorted(STRICT + CONTESTED))

#: Excluded under both, and not contested by anyone: seven numeric answers whose
#: digits match years, page numbers and counts on any page, plus `030`, whose
#: five gold components are listed in the question itself.
EXCLUDED = ("001", "002", "004", "005", "030", "043", "044", "047")

#: Kept apart from `answer_realization` deliberately. An answer reached through
#: three hops can still be one string on one page, and conflating the two would
#: exclude most of the eligible set for a property of the search rather than of
#: the answer.
ANSWER_REALIZATIONS = (
    "direct_string",
    "component_list_single_source",
    "derived",
    "nontext",
)
RETRIEVAL_PATHS = ("single_source", "multi_hop", "multi_source")


def denominators() -> dict[str, tuple[str, ...]]:
    return {"strict": STRICT, "lenient": LENIENT}


__all__ = [
    "ANSWER_REALIZATIONS",
    "CONTESTED",
    "EXCLUDED",
    "LENIENT",
    "RETRIEVAL_PATHS",
    "STRICT",
    "denominators",
]
