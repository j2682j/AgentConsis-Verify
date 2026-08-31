"""Candidate boundaries must be real substrings, punctuation and all.

Six gold spans were unreachable in the first oracle, and the cause was one
inconsistency applied twice: expansion grew by whitespace tokens and returned
`Legume Wikipedia page?`, while the filter ran `.strip(" ,.;:")` and turned
`... if there is one.` into `... if there is one`. Punctuation was attached
during generation and removed during filtering, so a gold with a full stop and a
gold without a question mark both missed.

The repair is not a tolerant comparison -- that would let any two spans
differing only in trailing marks count as equal, and exact match would stop
meaning anything. Both forms are emitted as separate candidates because both
genuinely occur, and every candidate is addressed by offset so its text is
whatever `context[start:end]` says it is.

A seventh case, `Emily Midkiff's` where the gold is `Emily Midkiff`, is a
different fault: whitespace splitting cannot express that boundary at all.
"""

from __future__ import annotations

import unittest

from scripts.replay.boundary_candidates import (
    Candidate,
    finalise,
    merge,
    possessive_trim,
    trim_boundary,
    with_punctuation_variants,
)


class PunctuationVariantTest(unittest.TestCase):
    CONTEXT = "logs on the Legume Wikipedia page?"

    def test_a_span_ending_in_punctuation_yields_both_forms(self) -> None:
        start = self.CONTEXT.index("Legume")
        variants = with_punctuation_variants(self.CONTEXT, [(start, len(self.CONTEXT))])

        produced = {self.CONTEXT[a:b] for a, b in variants}
        self.assertIn("Legume Wikipedia page?", produced)
        self.assertIn("Legume Wikipedia page", produced)

    def test_both_forms_are_real_substrings(self) -> None:
        start = self.CONTEXT.index("Legume")
        for a, b in with_punctuation_variants(self.CONTEXT, [(start, len(self.CONTEXT))]):
            with self.subTest(span=self.CONTEXT[a:b]):
                self.assertIn(self.CONTEXT[a:b], self.CONTEXT)

    def test_a_sentence_final_stop_is_not_stripped_away(self) -> None:
        """The gold for task 032 ends in a full stop; it has to stay reachable."""

        context = "received a bug fix? Just give the name, not a path."
        start = context.index("Just give")
        produced = {
            context[a:b]
            for a, b in with_punctuation_variants(context, [(start, len(context))])
        }

        self.assertIn("Just give the name, not a path.", produced)
        self.assertIn("Just give the name, not a path", produced)

    def test_trimming_only_moves_the_edges(self) -> None:
        context = '  "quoted phrase",  '
        start, end = trim_boundary(context, 0, len(context))

        self.assertEqual(context[start:end], "quoted phrase")


class PossessiveTest(unittest.TestCase):
    def test_a_possessive_ending_can_be_dropped(self) -> None:
        context = "In Emily Midkiff's June 2014 article"
        start = context.index("Emily")
        end = context.index("'s") + 2

        trimmed = possessive_trim(context, start, end)

        self.assertIsNotNone(trimmed)
        self.assertEqual(context[trimmed[0] : trimmed[1]], "Emily Midkiff")

    def test_a_typographic_apostrophe_is_handled_too(self) -> None:
        context = "In Emily Midkiff’s June 2014 article"
        start = context.index("Emily")
        end = context.index("’s") + 2

        trimmed = possessive_trim(context, start, end)

        self.assertEqual(context[trimmed[0] : trimmed[1]], "Emily Midkiff")

    def test_a_name_containing_an_apostrophe_is_left_alone(self) -> None:
        """`O'Connor` is a name, not a possessive."""

        context = "written by O'Connor in 1955"
        start = context.index("O'Connor")
        end = start + len("O'Connor")

        self.assertIsNone(possessive_trim(context, start, end))

    def test_a_bare_possessive_apostrophe_is_dropped(self) -> None:
        context = "the players' union"
        start = context.index("players'")
        end = start + len("players'")

        trimmed = possessive_trim(context, start, end)

        self.assertEqual(context[trimmed[0] : trimmed[1]], "players")

    def test_an_apostrophe_alone_produces_nothing(self) -> None:
        context = "a ' b"

        self.assertIsNone(possessive_trim(context, 2, 3))


class MergeTest(unittest.TestCase):
    CONTEXT = "the Legume Wikipedia page?"

    def test_the_same_boundary_from_two_generators_is_one_candidate(self) -> None:
        collected: dict[tuple[int, int], set[str]] = {}
        span = (self.CONTEXT.index("Legume"), self.CONTEXT.index("page") + 4)
        merge(collected, "expansion", self.CONTEXT, [span])
        merge(collected, "noun_chunk", self.CONTEXT, [span])

        candidates = finalise(collected, self.CONTEXT)
        match = next(c for c in candidates if c.text(self.CONTEXT) == "Legume Wikipedia page")

        self.assertEqual(match.generators, ("expansion", "noun_chunk"))

    def test_every_candidate_reproduces_from_its_offsets(self) -> None:
        collected: dict[tuple[int, int], set[str]] = {}
        merge(collected, "expansion", self.CONTEXT, [(0, len(self.CONTEXT))])

        for candidate in finalise(collected, self.CONTEXT):
            with self.subTest(candidate=candidate):
                self.assertEqual(
                    candidate.text(self.CONTEXT),
                    self.CONTEXT[candidate.start : candidate.end],
                )

    def test_offsets_outside_the_context_are_refused(self) -> None:
        collected: dict[tuple[int, int], set[str]] = {}
        merge(collected, "expansion", self.CONTEXT, [(5, len(self.CONTEXT) + 10)])

        self.assertEqual(finalise(collected, self.CONTEXT), [])

    def test_blank_candidates_are_dropped(self) -> None:
        context = "a   b"
        collected: dict[tuple[int, int], set[str]] = {}
        merge(collected, "expansion", context, [(1, 4)])

        self.assertEqual(finalise(collected, context), [])


if __name__ == "__main__":
    unittest.main()


class OracleIntegrationTest(unittest.TestCase):
    """The same two guarantees, over the real 38 gold cases rather than fixtures.

    Unit fixtures show the offsets behave on the shapes they were written for.
    These run the whole generator set over the annotated contexts, because the
    defect being fixed was invisible at the unit level -- each generator looked
    correct on its own, and the loss happened where generation met filtering.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from scripts.replay.boundary_candidate_oracle import candidates_for
            from scripts.replay.boundary_recovery_prototype import Recovery, load_gold
        except Exception as exc:  # spaCy or the annotation files are absent
            raise unittest.SkipTest(f"oracle inputs unavailable: {exc}")
        cls.cases = load_gold()
        recovery = Recovery()
        cls.produced = {
            case.annotation_id: candidates_for(
                recovery.nlp(case.context), case.context, case.span_text
            )
            for case in cls.cases
        }

    def test_every_candidate_is_a_contiguous_run_of_its_context(self) -> None:
        for case in self.cases:
            for candidate in self.produced[case.annotation_id]:
                text = candidate.text(case.context)
                with self.subTest(case=case.annotation_id, candidate=text[:40]):
                    self.assertEqual(
                        case.context[candidate.start : candidate.end], text
                    )
                    self.assertIn(text, case.context)

    def test_no_candidate_carries_leading_or_trailing_whitespace(self) -> None:
        """A whitespace edge means an offset was taken from the wrong side."""

        for case in self.cases:
            for candidate in self.produced[case.annotation_id]:
                text = candidate.text(case.context)
                with self.subTest(case=case.annotation_id, candidate=text[:40]):
                    self.assertEqual(text, text.strip())

    def test_the_candidate_count_stays_within_the_measured_envelope(self) -> None:
        """Emitting both punctuation readings roughly doubles nothing.

        The pre-fix run had P95 172. Both readings of a boundary are only
        emitted where the edge actually sits on punctuation, so the set grows by
        a margin rather than by a factor; a regression past 220 would mean
        variants are being produced where no punctuation exists.
        """

        sizes = sorted(
            len({c.text(case.context).casefold() for c in self.produced[case.annotation_id]})
            for case in self.cases
        )
        p95 = sizes[int(len(sizes) * 0.95) - 1]

        self.assertLessEqual(p95, 220, f"P95 候選數 {p95}，超出量測範圍")

    def test_the_generators_that_found_a_gold_span_are_recorded(self) -> None:
        """Provenance has to survive dedup, or the ablation measures nothing."""

        for case in self.cases:
            for candidate in self.produced[case.annotation_id]:
                with self.subTest(case=case.annotation_id):
                    self.assertTrue(candidate.generators)
