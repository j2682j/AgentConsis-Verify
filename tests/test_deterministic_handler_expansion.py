from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.deterministic_handlers import DeterministicHandlerRouter


class DeterministicHandlerExpansionTests(unittest.TestCase):
    def test_table_multi_filter_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text(
                "Name,Age,City,Score\n"
                "Ada,20,Boston,70\n"
                "Lin,35,Boston,80\n"
                "Max,40,Boston,90\n"
                "Eve,45,Taipei,100\n",
                encoding="utf-8",
            )
            router = DeterministicHandlerRouter(
                threshold=0.62,
                similarity_fn=lambda left, right: None,
            )

            result = router.run(
                question="What is the average Score where Age > 30 and City = Boston?",
                attachment={"file_path": str(path), "extension": ".csv"},
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "table_exact_operations")
        self.assertEqual(result.answer, "85")
        self.assertEqual(result.structured_result["matched_row_count"], 2)
        self.assertEqual(len(result.structured_result["filters"]), 2)

    def test_graph_weighted_shortest_path_total_weight(self):
        router = DeterministicHandlerRouter(
            threshold=0.62,
            similarity_fn=lambda left, right: None,
        )

        result = router.run(
            question="In this graph, what is the total distance of the shortest path from A to C? A-B: 5, B-C: 2, A-C: 10"
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "graph_shortest_path")
        self.assertEqual(result.answer, "7")
        self.assertEqual(result.structured_result["path"], ["A", "B", "C"])
        self.assertTrue(result.structured_result["weighted"])

    def test_date_difference(self):
        router = DeterministicHandlerRouter(
            threshold=0.62,
            similarity_fn=lambda left, right: None,
        )

        result = router.run(
            question="How many days elapsed between March 4, 2019 and March 14, 2019?"
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "date_time")
        self.assertEqual(result.answer, "10")

    def test_numeric_percentage_change(self):
        router = DeterministicHandlerRouter(
            threshold=0.62,
            similarity_fn=lambda left, right: None,
        )

        result = router.run(question="What is the percentage change from 80 to 100?")

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "numeric_reasoning")
        self.assertEqual(result.answer, "25")

    def test_text_extraction_nth_word_from_attachment_result(self):
        router = DeterministicHandlerRouter(
            threshold=0.62,
            similarity_fn=lambda left, right: None,
        )

        result = router.run(
            question="What is the third word in the text?",
            attachment_result="alpha beta gamma delta",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "text_extraction")
        self.assertEqual(result.answer, "gamma")

    def test_coordinate_dms_haversine(self):
        router = DeterministicHandlerRouter(
            threshold=0.62,
            similarity_fn=lambda left, right: None,
        )

        result = router.run(
            question=(
                "Compute the haversine distance between "
                "0 degrees 0 minutes 0 seconds N, 0 degrees 0 minutes 0 seconds E "
                "and 0 degrees 0 minutes 0 seconds N, 1 degrees 0 minutes 0 seconds E."
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.handler_name, "coordinate_distance")
        self.assertTrue(result.answer.endswith("km"))


if __name__ == "__main__":
    unittest.main()
