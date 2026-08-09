"""Pin how a fact's subject is matched against a goal's subject.

Subjects used to be compared as raw substrings, so anything short bound
anything longer that happened to spell it. On level1_final_14 a contract with
subject "I" bound the goal subject "Wikipedia" -- "i" is inside "wikipedia" --
and the resulting fact, "I nominated_by this particular article", counted as a
direct answer to a question asking *who* nominated an article. That marked the
task sufficient and stopped retrieval two rounds early on a wrong answer.

Comparing tokens instead keeps the shortenings that matter (a surname reaching
a full name) and drops the coincidences. Replayed over every contract-goal pair
recorded in level1_final_13 and _14, exactly one of seven bindings changes: the
"I" / "Wikipedia" one.
"""

from __future__ import annotations

import unittest

from tools.evidence.fact_extraction.fact_goal_binding_validator import (
    FactGoalBindingValidator,
)


class SubjectEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = FactGoalBindingValidator()

    def _equivalent(self, first: str, second: str) -> bool:
        return self.validator._entity_equivalent(first, second)

    def test_a_letter_does_not_bind_a_name_that_contains_it(self) -> None:
        """The level1_final_14 defect, in one line."""

        self.assertFalse(self._equivalent("I", "Wikipedia"))

    def test_short_pronouns_do_not_bind(self) -> None:
        for subject, goal in (("it", "Italy"), ("he", "Helsinki"), ("a", "Anchorage")):
            with self.subTest(subject=subject):
                self.assertFalse(self._equivalent(subject, goal))

    def test_a_partial_name_still_binds_the_full_name(self) -> None:
        """Losing this would cost the bindings the change is meant to keep."""

        self.assertTrue(self._equivalent("Claus", "Claus Peter Flor"))
        self.assertTrue(self._equivalent("Enrollment Count", "actual enrollment count"))
        self.assertTrue(self._equivalent("Dimond Center", "Dimond Center"))

    def test_matching_ignores_case_punctuation_and_order(self) -> None:
        self.assertTrue(self._equivalent("Teal'c", "teal c"))
        self.assertTrue(self._equivalent("Petersen, Carolyn", "Carolyn Petersen"))

    def test_unrelated_names_do_not_bind(self) -> None:
        self.assertFalse(self._equivalent("Carolyn Collins Petersen", "R. G. Arendt"))

    def test_an_empty_subject_binds_nothing(self) -> None:
        self.assertFalse(self._equivalent("", "Wikipedia"))
        self.assertFalse(self._equivalent("Wikipedia", ""))

    def test_overlap_alone_is_not_enough(self) -> None:
        """Sharing one token does not make two subjects the same entity."""

        self.assertFalse(self._equivalent("Boston Marathon", "Marathon County"))


class BindingUsesTheSubjectCheckTest(unittest.TestCase):
    """The comparison has to be reached through `validate`, not just exist."""

    def setUp(self) -> None:
        self.validator = FactGoalBindingValidator()

    def _validate(self, subject: str):
        from types import SimpleNamespace

        return self.validator.validate(
            fact={
                "fact_id": "F1",
                "goal_id": "G1",
                "subject": subject,
                "relation": "nominated by",
                "object": "FunkMonk",
                "grounding_status": "grounded",
            },
            goal=SimpleNamespace(
                goal_id="G1",
                subject="Wikipedia",
                relation="nominated by",
                target="nominator",
            ),
            effective_subjects=["Wikipedia"],
            answer_role="nominator",
        )

    def test_a_letter_subject_is_rejected_as_a_mismatch(self) -> None:
        result = self._validate("I")

        self.assertFalse(result.bound)
        self.assertEqual(result.status, "subject_mismatch")

    def test_the_real_subject_still_binds(self) -> None:
        self.assertTrue(self._validate("Wikipedia").bound)


if __name__ == "__main__":
    unittest.main()
