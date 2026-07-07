from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from core.evidence_runner import EvidenceRunner
from tools.deterministic_handlers import (
    DeterministicHandlerRouter,
    HandlerResult,
    default_deterministic_registry,
)


def similarity_for(target: str):
    def _similarity(left: str, right: str) -> float:
        return 1.0 if target in right else -1.0

    return _similarity


class DeterministicHandlerRouterTests(unittest.TestCase):
    def test_registry_lists_default_handlers(self):
        registry = default_deterministic_registry()
        names = {handler.name for handler in registry.list_handlers()}

        self.assertIn("coordinate_distance", names)
        self.assertIn("graph_shortest_path", names)
        self.assertIn("table_aggregation", names)
        self.assertIn("table_exact_operations", names)
        self.assertIn("boggle_dfs", names)
        self.assertIn("sexagesimal_conversion", names)
        self.assertIn("unit_conversion", names)
        self.assertIn("string_transform", names)
        self.assertIn("list_operations", names)
        self.assertIn("simple_math", names)

    def test_no_match_returns_no_match(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=lambda left, right: -1.0,
        )

        result = router.run(question="Who won the award?")

        self.assertEqual(result.status, "no_match")
        self.assertFalse(result.ok)
        self.assertIn("matches", result.structured_result)

    def test_coordinate_distance_handler_runs(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("coordinate"),
        )

        result = router.run(
            question="Compute the distance between coordinates (0, 0) and (3, 4)."
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "coordinate_distance")
        self.assertEqual(result.answer, "5")

    def test_graph_shortest_path_reads_edge_list_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text("from,to\nA,B\nB,C\nC,D\n", encoding="utf-8")
            router = DeterministicHandlerRouter(
                threshold=0.9,
                similarity_fn=similarity_for("shortest path"),
            )

            result = router.run(
                question='How many stops are needed from "A" to "D"?',
                attachment={"file_path": str(path), "extension": ".csv"},
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "graph_shortest_path")
        self.assertEqual(result.answer, "3")
        self.assertEqual(result.structured_result["path"], ["A", "B", "C", "D"])

    def test_graph_shortest_path_reports_missing_inputs(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("shortest path"),
        )

        result = router.run(question="Find the shortest route from A to D.")

        self.assertEqual(result.status, "missing_inputs")
        self.assertIn("edges", result.missing_inputs)

    def test_table_aggregation_reads_csv_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("name,score\nAda,8\nLin,13\nMax,5\n", encoding="utf-8")
            router = DeterministicHandlerRouter(
                threshold=0.9,
                similarity_fn=similarity_for("CSV or table aggregation"),
            )

            result = router.run(
                question="What is the max score in the table?",
                attachment={"file_path": str(path), "extension": ".csv"},
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "table_aggregation")
        self.assertEqual(result.answer, "13")

    def test_boggle_handler_runs_through_router(self):
        router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("Boggle"),
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
            similarity_fn=similarity_for("Convert units"),
        )

        result = router.run(question="Convert 250 cm to m.")

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "unit_conversion")
        self.assertEqual(result.answer, "2.5")

    def test_deterministic_solver_tool_uses_handler_router(self):
        from tools.deterministic_solver_tool import DeterministicSolverTool

        tool = DeterministicSolverTool()
        tool.router = DeterministicHandlerRouter(
            threshold=0.9,
            similarity_fn=similarity_for("Boggle"),
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

    def test_evidence_runner_appends_deterministic_handler_evidence(self):
        class FakeRouter:
            def run(self, **kwargs):
                return HandlerResult(
                    handler_name="simple_math",
                    status="ok",
                    answer="42",
                    evidence_text="Deterministic handler evidence:\nAnswer: 42",
                    structured_result={"source": "fake"},
                )

        runner = EvidenceRunner(
            question="Compute 40 + 2.",
            search_result="provided search",
            attachment_result="provided attachment",
            deterministic_handler_router=FakeRouter(),
            enable_deterministic_handler_router=True,
        )

        evidence = runner.run()

        self.assertIn("Answer: 42", evidence["solver_result"])
        usage = [
            item
            for item in evidence["tool_usage"]
            if item.get("tool_name") == "deterministic_handler_router"
        ]
        self.assertEqual(len(usage), 1)
        self.assertTrue(usage[0]["ok"])


if __name__ == "__main__":
    unittest.main()
