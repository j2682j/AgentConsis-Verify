from __future__ import annotations

import unittest

from tools.deterministic_handlers import DeterministicHandlerRouter


class DeterministicHandlerExpansionTests(unittest.TestCase):
    def test_registry_exposes_new_high_value_roles(self):
        router = DeterministicHandlerRouter()
        for role in (
            "probability_simulation",
            "logic_equivalence",
            "multi_step_counting",
            "chess_tactics",
        ):
            self.assertTrue(router.registry.find_by_role(role), role)

    def test_logic_equivalence_handler_solves_truth_table_task(self):
        result = DeterministicHandlerRouter().run(
            question='Are "not (A and B)" and "not A or not B" logically equivalent?',
            required_handler_role="logic_equivalence",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "yes")
        self.assertEqual(result.output_type, "final_answer")

    def test_probability_handler_solves_dice_sum_task(self):
        result = DeterministicHandlerRouter().run(
            question="If rolling 2d6, what is the probability that the sum is exactly 7?",
            required_handler_role="probability_simulation",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "1/6")
        self.assertEqual(result.output_type, "final_answer")


if __name__ == "__main__":
    unittest.main()
