"""Pin what a morphology-aware attestation count is allowed to match.

`_drop_unattested_candidates` reserves a candidate the fetched corpus never
states. Task 034's `Rockhopper penguins` scored zero mentions while the corpus
states `rockhopper penguin` -- the same species in the singular -- so a
three-run specific answer lost to a one-run generic one.

Widening the count is a change to a gate that measures 8 helps against 2 hurts
over five runs, so what the widening can reach has to stay small and provable.
These tests hold the two properties the safety argument rests on: the modifier
is never dropped, and nothing but a plural's own singular is generated.

The second property is one-directional on purpose. While variants were
generated both ways, the candidate `R` produced `Rs` and matched an unrelated
token in the corpus, which would have readmitted a candidate the corpus does
not state at all.
"""

from __future__ import annotations

import unittest

from scripts.replay.corpus_attestation_diagnostics import (
    CandidateAttestation,
    classify,
    morphological_variants,
)


class MorphologicalVariantTest(unittest.TestCase):
    def test_a_plural_reaches_its_own_singular(self) -> None:
        self.assertEqual(
            morphological_variants("Rockhopper penguins"), ["Rockhopper penguin"]
        )
        self.assertEqual(morphological_variants("penguins"), ["penguin"])

    def test_the_modifier_is_never_dropped(self) -> None:
        """`Rockhopper penguins` must not reach the genus on its own."""

        for answer in ("Rockhopper penguins", "Emperor penguins", "Saint Petersburgs"):
            with self.subTest(answer=answer):
                for variant in morphological_variants(answer):
                    self.assertEqual(
                        variant.split()[:-1],
                        answer.split()[:-1],
                        "every word but the last must survive",
                    )
                    self.assertEqual(len(variant.split()), len(answer.split()))

    def test_no_plurals_are_invented(self) -> None:
        """A singular candidate generates nothing; that direction only misfires."""

        for answer in ("penguin", "Saint Petersburg", "R", "Guava"):
            with self.subTest(answer=answer):
                self.assertEqual(morphological_variants(answer), [])

    def test_short_words_are_left_alone(self) -> None:
        """`R` used to yield `Rs`, which matched an unrelated corpus token."""

        for answer in ("R", "Rs", "As", "is"):
            with self.subTest(answer=answer):
                self.assertEqual(morphological_variants(answer), [])

    def test_double_s_is_not_a_plural(self) -> None:
        for answer in ("class", "glass", "address"):
            with self.subTest(answer=answer):
                self.assertEqual(morphological_variants(answer), [])

    def test_es_plurals_lose_both_letters(self) -> None:
        self.assertEqual(morphological_variants("classes"), ["class"])
        self.assertEqual(morphological_variants("boxes"), ["box"])
        self.assertEqual(morphological_variants("branches"), ["branch"])


class ClassificationTest(unittest.TestCase):
    def _row(self, **kwargs) -> CandidateAttestation:
        base = dict(
            candidate="x",
            production_mentions=0,
            full_corpus_exact_mentions=0,
            attestation_window_exact_mentions=0,
            canonical_morph_mentions=0,
        )
        base.update(kwargs)
        return CandidateAttestation(**base)

    def test_task_034_is_morphology_blind(self) -> None:
        row = self._row(canonical_morph_mentions=5)

        self.assertEqual(classify(row), "MORPHOLOGY_BLIND")

    def test_a_form_past_the_character_cap_is_window_blind(self) -> None:
        row = self._row(full_corpus_exact_mentions=7)

        self.assertEqual(classify(row), "WINDOW_BLIND")

    def test_a_candidate_the_corpus_never_states_stays_unattested(self) -> None:
        """The category the widening must never move."""

        self.assertEqual(classify(self._row()), "TRULY_UNATTESTED")

    def test_a_count_production_missed_is_an_implementation_bug(self) -> None:
        row = self._row(attestation_window_exact_mentions=4, production_mentions=0)

        self.assertEqual(classify(row), "COUNTING_IMPLEMENTATION_BUG")


if __name__ == "__main__":
    unittest.main()
