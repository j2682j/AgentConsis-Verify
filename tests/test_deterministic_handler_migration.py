from __future__ import annotations

import unittest

from tools.deterministic_handlers import DeterministicHandlerRouter
from tools.deterministic_solver_tool import DeterministicSolverTool


def similarity_for(target: str):
    def _similarity(left: str, right: str) -> float:
        del left
        return 1.0 if target.lower() in right.lower() else -1.0

    return _similarity


class DeterministicHandlerMigrationTests(unittest.TestCase):
    def test_boggle_handler_runs_through_router(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("boggle"),
        )

        result = router.run(
            question='Find all words in this Boggle letter grid: {"grid":["CAT","RRE","DOG"],"words":["CAT","CAR","DOG","TREE"]}'
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "boggle_dfs")
        self.assertEqual(result.answer, "CAT, CAR, DOG")

    def test_sexagesimal_handler_runs_through_router(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("sexagesimal"),
        )

        result = router.run(
            question="Convert 12 degrees 30 minutes 0 seconds from sexagesimal to decimal degrees."
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "sexagesimal_conversion")
        self.assertEqual(result.answer, "12.5")

    def test_unit_handler_runs_through_router(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("convert units"),
        )

        result = router.run(question="Convert 250 cm to m.")

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "unit_conversion")
        self.assertEqual(result.answer, "2.5")

    def test_deterministic_solver_tool_uses_handler_router(self):
        tool = DeterministicSolverTool()
        tool.router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("boggle"),
        )

        result = tool.run(
            {
                "input": 'Find all words in this Boggle letter grid: {"grid":["CAT","RRE","DOG"],"words":["CAT","CAR","DOG","TREE"]}'
            }
        )

        self.assertTrue(result["used_deterministic_solver"])
        self.assertEqual(result["task_type"], "boggle_dfs")
        self.assertEqual(result["answer_text"], "CAT, CAR, DOG")
        self.assertEqual(result["evidence_source"], "deterministic_handler_router")


if __name__ == "__main__":
    unittest.main()
