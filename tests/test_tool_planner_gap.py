from __future__ import annotations

import unittest

from tools.tool_planner import (
    SLMToolPlanner,
    ToolCandidate,
    ToolCandidateRouter,
    ToolPlanningRunner,
)


class ToolPlannerGapTests(unittest.TestCase):
    def test_candidate_router_uses_deterministic_gap_to_add_recovery_tools(self):
        candidates = ToolCandidateRouter().route(
            question="What is the average Score?",
            attachment={"file_path": "scores.csv", "extension": ".csv"},
            routing={
                "use_search": False,
                "use_attachment": False,
                "deterministic_tool_gap": {
                    "handler_name": "table_exact_operations",
                    "missing_inputs": ["table_rows"],
                    "next_action_hint": "Use attachment_reader or provide CSV rows.",
                },
            },
        )

        names = [candidate.tool_name for candidate in candidates]
        self.assertIn("attachment_reader", names)
        self.assertIn("deterministic_handler", names)
        attachment = next(
            candidate for candidate in candidates if candidate.tool_name == "attachment_reader"
        )
        self.assertTrue(attachment.required)
        self.assertIn("attachment_reader", attachment.priority_hint)

    def test_fallback_planner_runs_recovery_before_deterministic_handler(self):
        candidates = [
            ToolCandidate("attachment_reader", "read"),
            ToolCandidate("deterministic_handler", "compute"),
        ]

        plan = ToolPlanningRunner(
            slm_planner=SLMToolPlanner(model_name=""),
        ).fallback_planner.plan(
            candidates=candidates,
            routing={
                "deterministic_tool_gap": {
                    "missing_inputs": ["table_rows"],
                    "next_action_hint": "Use attachment_reader.",
                }
            },
            deterministic_handler_requested=True,
        )

        self.assertEqual(
            [step.tool_name for step in plan.tool_sequence],
            ["attachment_reader", "deterministic_handler"],
        )
        self.assertEqual(
            plan.tool_sequence[0].purpose,
            "recover deterministic handler missing inputs",
        )


if __name__ == "__main__":
    unittest.main()
