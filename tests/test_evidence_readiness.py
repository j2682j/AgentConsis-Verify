from __future__ import annotations

import unittest

from tools.evidence.evidence_readiness import (
    EvidenceReadinessEvaluator,
    EvidenceReadinessStatus,
)
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore


class EvidenceReadinessEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = EvidenceReadinessEvaluator()

    def test_nonempty_attachment_is_partial_not_direct_supported(self):
        result = self.evaluator.evaluate(
            fact_store=TaskFactStore(),
            tool_usage=[{"ok": True, "tool_name": "attachment_reader"}],
            routing={"primary_route": "attachment", "search_policy": "deferred"},
            attachment_result="Parsed attachment text",
        )

        self.assertEqual(result.status, EvidenceReadinessStatus.PARTIAL)
        self.assertFalse(result.is_sufficient)

    def test_verifiable_fact_is_direct_supported(self):
        store = TaskFactStore()
        store.add(
            EvidenceFact(
                fact_id="fact-1",
                subject="album",
                relation="studio album count",
                object="3",
                qualifiers={"answer_binding": "direct"},
                role="ANSWER_SUPPORT",
                evidence_spans=["released three studio albums"],
                context="The group released three studio albums.",
                source_id="doc-1",
                source_type="search",
                grounding_status="grounded",
                extraction_method="semantic_model",
            )
        )

        result = self.evaluator.evaluate(
            fact_store=store,
            tool_usage=[],
            routing={"primary_route": "factual_search", "search_policy": "immediate"},
        )

        self.assertEqual(result.status, EvidenceReadinessStatus.DIRECT_SUPPORTED)
        self.assertEqual(result.direct_fact_ids, ["fact-1"])

    def test_explicit_search_gap_requests_search(self):
        result = self.evaluator.evaluate(
            fact_store=TaskFactStore(),
            tool_usage=[
                {
                    "ok": False,
                    "tool_name": "deterministic_handler_router",
                    "missing_inputs": ["source_text"],
                    "next_capability": "search",
                }
            ],
            routing={"primary_route": "deterministic", "search_policy": "deferred"},
        )

        self.assertEqual(result.status, EvidenceReadinessStatus.NEEDS_EXTERNAL)
        self.assertEqual(result.next_capability, "search")

    def test_unknown_route_without_evidence_uses_fallback_search(self):
        result = self.evaluator.evaluate(
            fact_store=TaskFactStore(),
            tool_usage=[],
            routing={"primary_route": "unknown", "search_policy": "fallback"},
        )

        self.assertEqual(result.status, EvidenceReadinessStatus.NEEDS_EXTERNAL)
        self.assertEqual(result.next_capability, "search")


if __name__ == "__main__":
    unittest.main()
