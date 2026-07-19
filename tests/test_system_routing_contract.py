from __future__ import annotations

import unittest

from tools.system_routing_contract import SystemRoutingContract


class SystemRoutingContractTests(unittest.TestCase):
    def test_attachment_metadata_takes_priority_over_search_terms(self):
        decision = SystemRoutingContract().route(
            question="According to the attached spreadsheet, which company had the highest revenue in 2020?",
            stage="stage1_round0",
            has_attachment=True,
            attachment_type="xlsx",
        )

        self.assertEqual(decision.initial_route, "attachment_first")
        self.assertTrue(decision.use_attachment)
        self.assertFalse(decision.use_search)
        self.assertFalse(decision.search_allowed)
        self.assertEqual(decision.task_type, "attachment_deterministic_solver")

    def test_factual_question_without_attachment_uses_search_first(self):
        decision = SystemRoutingContract().route(
            question="According to GitHub, when was Regression added to the oldest closed numpy.polynomial issue?",
            stage="stage1_round0",
        )

        self.assertEqual(decision.initial_route, "search_first")
        self.assertTrue(decision.use_search)
        self.assertTrue(decision.search_allowed)
        self.assertEqual(decision.task_type, "factual_search")

    def test_code_question_uses_deterministic_first(self):
        decision = SystemRoutingContract().route(
            question='In Unlambda, what exact character needs to be added to correct the following code to output "For penguins"? `r```.F`',
            stage="stage1_round0",
        )

        self.assertEqual(decision.initial_route, "deterministic_first")
        self.assertTrue(decision.use_deterministic_solver)
        self.assertFalse(decision.use_search)
        self.assertFalse(decision.search_allowed)


if __name__ == "__main__":
    unittest.main()
