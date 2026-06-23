from __future__ import annotations

import json
import unittest

from tools.deterministic_solver import DeterministicSolver


class DeterministicHandlerTests(unittest.TestCase):
    def setUp(self):
        self.solver = DeterministicSolver()

    def test_boggle_dfs(self):
        payload = {
            "grid": ["CAT", "RRE", "DOG"],
            "words": ["CAT", "CAR", "DOG", "TREE"],
        }
        result = self.solver.solve(
            f"Find all words in this Boggle letter grid: {json.dumps(payload)}"
        )

        self.assertTrue(result.used_deterministic_solver)
        self.assertEqual(result.task_type, "boggle_dfs")
        self.assertEqual(result.answer_text, "CAT, CAR, DOG")

    def test_graph_shortest_path_and_hop_count(self):
        payload = {
            "edges": [["A", "B"], ["B", "C"], ["A", "D"], ["D", "E"], ["E", "C"]],
            "start": "A",
            "end": "C",
        }
        result = self.solver.solve(
            f"How many stops are on the shortest route in this graph? {json.dumps(payload)}"
        )

        self.assertTrue(result.used_deterministic_solver)
        self.assertEqual(result.task_type, "graph_hop_count")
        self.assertEqual(result.answer_text, "2")

    def test_coordinate_euclidean_distance(self):
        result = self.solver.solve(
            "Calculate the coordinate distance between (0, 0) and (3, 4)."
        )

        self.assertTrue(result.used_deterministic_solver)
        self.assertEqual(result.task_type, "coordinate_euclidean_distance")
        self.assertEqual(result.answer_text, "5")

    def test_sexagesimal_to_decimal(self):
        result = self.solver.solve(
            "Convert 12 degrees 30 minutes 0 seconds from sexagesimal to decimal degrees."
        )

        self.assertTrue(result.used_deterministic_solver)
        self.assertEqual(result.task_type, "sexagesimal_to_decimal")
        self.assertEqual(result.answer_text, "12.5")

    def test_table_filter_and_average(self):
        table_data = [
            {"Name": "A", "Age": "20", "Score": "70"},
            {"Name": "B", "Age": "35", "Score": "80"},
            {"Name": "C", "Age": "40", "Score": "90"},
        ]
        result = self.solver.solve(
            "In the spreadsheet, calculate the average Score where Age > 30.",
            table_data=table_data,
        )

        self.assertTrue(result.used_deterministic_solver)
        self.assertEqual(result.task_type, "table_average")
        self.assertEqual(result.answer_text, "85")


if __name__ == "__main__":
    unittest.main()
