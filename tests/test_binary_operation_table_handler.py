from __future__ import annotations

import unittest

from tools.deterministic_handlers.handlers.binary_operation_table import (
    BinaryOperationTableRouterHandler,
)
from tools.deterministic_handlers.router import DeterministicHandlerRouter


QUESTION = """Given this table defining * on the set S = {a, b, c, d, e}

|*|a|b|c|d|e|
|---|---|---|---|---|---|
|a|a|b|c|b|d|
|b|b|c|a|e|c|
|c|c|a|b|b|a|
|d|b|e|b|e|d|
|e|d|b|a|d|c|

provide the subset of S involved in any possible counter-examples that prove *
is not commutative. Provide your answer as a comma separated list of the
elements in the set in alphabetical order."""


class BinaryOperationTableHandlerTests(unittest.TestCase):
    def test_reports_elements_in_commutativity_counterexamples(self):
        handler = BinaryOperationTableRouterHandler()
        inputs = handler.build_input(type("Input", (), {
            "question": QUESTION,
            "attachment_result": "",
            "adapted_inputs": lambda self: {},
        })())

        result = handler.run(inputs)

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "b, e")
        self.assertEqual(
            result.structured_result["counterexamples"],
            [
                {
                    "pair": ["b", "e"],
                    "forward": "c",
                    "reverse": "b",
                    "forward_expression": "b*e=c",
                    "reverse_expression": "e*b=b",
                }
            ],
        )

    def test_router_prefers_binary_operation_over_list_formatting(self):
        router = DeterministicHandlerRouter(similarity_fn=lambda left, right: 0.0)

        result = router.run(question=QUESTION)

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "binary_operation_table")
        self.assertEqual(result.answer, "b, e")


if __name__ == "__main__":
    unittest.main()
