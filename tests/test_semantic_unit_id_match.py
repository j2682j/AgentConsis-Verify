"""Pin that a model echoing the prompt's unit label does not lose its facts.

`_build_prompt` lists every source as `Unit T1`, so qwen3:4b answers with
`"unit_id": "Unit T1"` about as often as `"T1"`. The reply parser looked the
value up verbatim against ids of the form `T1`, so those replies were dropped
whole with `unknown_unit_id`.

On level1_final_16 task 031 that cost the entire extraction: five clean
transcript segments in, 848 completion tokens of correctly extracted ingredients
out, `fact_count: 0` — the ingredients were found and then discarded on the
label the prompt itself had introduced. Matching after stripping that prefix
took it to `fact_count: 5`.
"""

from __future__ import annotations

import unittest

from tools.evidence.fact_extraction.semantic_fact_extractor import SemanticFactExtractor


class SemanticUnitKeyTest(unittest.TestCase):
    def test_the_prompt_label_matches_the_bare_id(self) -> None:
        key = SemanticFactExtractor._unit_key

        self.assertEqual(key("Unit T1"), key("T1"))
        self.assertEqual(key("unit t1"), key("T1"))
        self.assertEqual(key("UNIT  T1"), key("T1"))

    def test_distinct_units_stay_distinct(self) -> None:
        key = SemanticFactExtractor._unit_key

        self.assertNotEqual(key("Unit T1"), key("Unit T2"))
        self.assertNotEqual(key("T1"), key("V1"))

    def test_an_id_that_merely_starts_with_unit_is_untouched(self) -> None:
        """Only the `Unit ` label is stripped, not a leading word `unit`."""

        key = SemanticFactExtractor._unit_key

        self.assertEqual(key("unittest"), "unittest")
        self.assertNotEqual(key("unittest"), key("test"))

    def test_blank_input_collapses_to_empty(self) -> None:
        key = SemanticFactExtractor._unit_key

        self.assertEqual(key(""), "")
        self.assertEqual(key("   "), "")

    def test_a_label_with_no_id_matches_nothing(self) -> None:
        """`Unit` alone is not a reference, and must not collide with a real id."""

        key = SemanticFactExtractor._unit_key

        self.assertNotEqual(key("Unit"), key("T1"))
        self.assertNotEqual(key("Unit   "), "")


if __name__ == "__main__":
    unittest.main()
