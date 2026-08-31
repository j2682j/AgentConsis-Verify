"""A year the question asked about has to survive into the query.

The query model, asked about albums released between 2000 and 2009, wrote
`between 2000 and 2:009`. The colon came from the model itself -- the same reply
spelled the year correctly elsewhere -- and a search for `2:009` places no
constraint on the year at all.

These tests are written from both sides. The damage has to be repaired, and the
things that merely look like damage -- a range the model assembled, a thousands
separator, a number the query never mentioned -- have to be left exactly as they
are. A repairer that rewrites more than it was asked to is worse than the fault
it fixes, because a wrong query is at least visible.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.query.query_literal_integrity import (
    protected_literals,
    repair_query,
)

QUESTION = (
    "How many studio albums were published by Mercedes Sosa between 2000 and "
    "2009 (included)? You can use the latest 2022 version of english wikipedia."
)


class ProtectedLiteralTest(unittest.TestCase):
    def test_years_in_the_question_are_protected(self) -> None:
        self.assertLessEqual({"2000", "2009", "2022"}, protected_literals(QUESTION))

    def test_a_date_is_one_literal_not_three(self) -> None:
        found = protected_literals("as compiled 08/21/2023 by the office")

        self.assertIn("08/21/2023", found)

    def test_a_question_without_numbers_protects_nothing(self) -> None:
        self.assertEqual(protected_literals("Who wrote this book?"), set())


class RepairTest(unittest.TestCase):
    def test_the_observed_corruption_is_repaired(self) -> None:
        result = repair_query(
            "Mercedes Sosa studio albums released between 2000 and 2:009", QUESTION
        )

        self.assertEqual(
            result.repaired, "Mercedes Sosa studio albums released between 2000 and 2009"
        )
        self.assertTrue(result.changed)

    def test_the_raw_query_is_kept_beside_the_repair(self) -> None:
        """A correction nobody can inspect afterwards is a correction on trust."""

        result = repair_query("albums between 2000 and 2:009", QUESTION)

        self.assertIn("2:009", result.raw)
        self.assertNotIn("2:009", result.repaired)
        self.assertEqual(result.repairs[0].before, "2:009")
        self.assertEqual(result.repairs[0].after, "2009")
        self.assertTrue(result.repairs[0].reason)

    def test_an_undamaged_query_is_returned_untouched(self) -> None:
        query = "Mercedes Sosa discography 2000-2009"
        result = repair_query(query, QUESTION)

        self.assertEqual(result.repaired, query)
        self.assertFalse(result.changed)
        self.assertEqual(result.repairs, ())

    def test_a_range_the_model_assembled_is_not_split(self) -> None:
        """`2000-2009` has the digits of neither literal, so it stays whole."""

        result = repair_query("albums 2000-2009 studio", QUESTION)

        self.assertIn("2000-2009", result.repaired)

    def test_a_number_the_question_never_mentioned_is_left_alone(self) -> None:
        result = repair_query("Mercedes Sosa top 5 albums", QUESTION)

        self.assertIn("5", result.repaired)
        self.assertEqual(result.repairs, ())

    def test_nothing_is_inserted_into_a_query_that_omits_a_year(self) -> None:
        """Deciding which numbers belong in a query is planning, not integrity."""

        result = repair_query("Mercedes Sosa studio albums", QUESTION)

        self.assertEqual(result.repaired, "Mercedes Sosa studio albums")

    def test_an_ambiguous_token_is_not_guessed_at(self) -> None:
        """Two literals share the digits, so there is no single right answer."""

        question = "Compare the 2009 edition with the 20:09 timestamp."
        result = repair_query("edition 2:009 details", question)

        self.assertEqual(result.repaired, "edition 2:009 details")
        self.assertEqual(result.repairs, ())

    def test_a_damaged_date_is_restored_whole(self) -> None:
        question = "the list as compiled 08/21/2023 by the office"
        result = repair_query("list compiled 08.21.2023", question)

        self.assertEqual(result.repaired, "list compiled 08/21/2023")

    def test_repair_is_idempotent(self) -> None:
        once = repair_query("albums between 2000 and 2:009", QUESTION).repaired
        twice = repair_query(once, QUESTION).repaired

        self.assertEqual(once, twice)

    def test_only_the_damaged_token_moves(self) -> None:
        result = repair_query(
            "albums 2000 and 2:009 from the 2022 version", QUESTION
        )

        self.assertEqual(
            result.repaired, "albums 2000 and 2009 from the 2022 version"
        )
        self.assertEqual(len(result.repairs), 1)


if __name__ == "__main__":
    unittest.main()
