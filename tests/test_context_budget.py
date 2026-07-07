from __future__ import annotations

import unittest

from context.context_budget import ContextBudget, ContextBudgetManager
from context.context_builder import ContextPacket
from context.stage1_context import Stage1ContextBuilder


class ContextBudgetTests(unittest.TestCase):
    def test_search_evidence_keeps_configured_number_of_items(self):
        evidence = ["Evidence:"]
        for index in range(1, 8):
            evidence.extend(
                [
                    f"[E{index}]",
                    f"Source Title: Source {index}",
                    "Evidence: " + ("answer supporting sentence " * 80),
                ]
            )
        manager = ContextBudgetManager(
            ContextBudget(
                max_total_chars=4000,
                max_search_evidence_items=5,
                max_search_evidence_chars=120,
            )
        )

        result = manager.apply({"question": "Q", "search_result": "\n".join(evidence)})

        self.assertIn("[E5]", result.sections["search_result"])
        self.assertNotIn("[E6]", result.sections["search_result"])
        self.assertEqual(result.diagnostics.dropped_evidence_count, 2)
        self.assertTrue(result.diagnostics.truncation_applied)

    def test_stage1_context_builder_returns_budget_diagnostics(self):
        builder = Stage1ContextBuilder(
            budget_manager=ContextBudgetManager(
                ContextBudget(
                    max_total_chars=2400,
                    max_search_evidence_items=2,
                    max_search_evidence_chars=100,
                    max_attachment_chars=120,
                )
            )
        )
        evidence_packets = [
            ContextPacket(
                packet_type="search_result",
                content="\n".join(
                    [
                        "Evidence:",
                        "[E1]",
                        "Source Title: A",
                        "Evidence: " + ("alpha " * 100),
                        "[E2]",
                        "Source Title: B",
                        "Evidence: " + ("beta " * 100),
                        "[E3]",
                        "Source Title: C",
                        "Evidence: " + ("gamma " * 100),
                    ]
                ),
            ),
            ContextPacket(
                packet_type="attachment_result",
                content="attachment " * 100,
            ),
        ]

        messages, diagnostics = builder.build_with_diagnostics(
            question="What is the answer?",
            evidence_packets=evidence_packets,
        )

        self.assertEqual(len(messages), 2)
        self.assertTrue(diagnostics["truncation_applied"])
        self.assertEqual(diagnostics["dropped_evidence_count"], 1)
        self.assertIn("search_result", diagnostics["truncated_sections"])
        self.assertIn("attachment_result", diagnostics["truncated_sections"])


if __name__ == "__main__":
    unittest.main()
