"""Pin the anti-fabrication rules the planner prompt must carry.

On level1_final_06 the planner produced 'NCT03822452' for task a0068077 and
'MLB' (instead of NPB) for task a0c07678 — both were confident-looking guesses
the search backend then anchored on, so the correct pages never surfaced. These
tests keep the guardrail rules inside USER_TEMPLATE so future prompt edits do
not silently drop them, without requiring an LLM to run.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.query.mask_salience_query import (
    MaskSalienceQueryGenerator,
)


class PlannerPromptAntiFabricationTest(unittest.TestCase):
    def test_prompt_forbids_inventing_specific_identifiers(self) -> None:
        template = MaskSalienceQueryGenerator.USER_TEMPLATE
        # The rule must name enough concrete identifier types that the model
        # cannot rationalise a novel case as "not covered".
        for keyword in ("NCT", "DOI", "identifiers"):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, template)
        self.assertIn("Copy such identifiers only when", template)

    def test_prompt_forbids_guessing_league_or_source(self) -> None:
        template = MaskSalienceQueryGenerator.USER_TEMPLATE
        self.assertIn("league", template)
        self.assertIn("Do not guess", template)
        self.assertIn("Keep entity names as written", template)


if __name__ == "__main__":
    unittest.main()
