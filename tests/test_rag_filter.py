from __future__ import annotations

import unittest
from pathlib import Path

from tools.search_result_builder.config import EvidenceItem
from tools.search_result_builder.next_hop_query.rag_filter import (
    PROJECT_FILTER_CHECKPOINT,
    EfficientRAGFilterAdapter,
)


class RAGFilterTests(unittest.TestCase):
    def test_default_checkpoint_uses_project_models_directory(self):
        rag_filter = EfficientRAGFilterAdapter()

        self.assertEqual(
            Path(rag_filter.filter_checkpoint),
            PROJECT_FILTER_CHECKPOINT,
        )
        self.assertEqual(rag_filter.device, "cpu")
        self.assertTrue(
            Path(rag_filter.filter_checkpoint, "config.json").exists()
        )

    def test_missing_checkpoint_uses_explicit_fallback(self):
        rag_filter = EfficientRAGFilterAdapter(
            filter_checkpoint="missing-filter-checkpoint",
        )
        evidence = [
            EvidenceItem(
                evidence_id="E1",
                source_id="S1",
                query_id="Q1",
                text="Moon minimum perigee distance is 356400 km.",
                matched_terms=["minimum", "perigee", "356400", "km"],
            )
        ]

        result = rag_filter.build_query(
            question="What is the Moon minimum perigee distance?",
            evidence_items=evidence,
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(
            result.metadata["method"],
            "efficientrag_filter_fallback",
        )
        self.assertIn("FileNotFoundError", result.metadata["model_error"])


if __name__ == "__main__":
    unittest.main()
