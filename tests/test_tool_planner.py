from __future__ import annotations

import unittest

from core.evidence_runner import EvidenceRunner
from tools.tool_planner import (
    HandlerPlan,
    ToolCandidate,
    ToolCandidateRouter,
    ToolPlan,
    ToolPlanParser,
    ToolPlanResult,
    ToolPlanStep,
    ToolPlanValidator,
    ToolPlanningRunner,
    SLMToolPlanner,
)
from tools.deterministic_handlers import HandlerMatch


class ToolPlannerTests(unittest.TestCase):
    def test_candidate_router_includes_attachment_search_and_deterministic(self):
        candidates = ToolCandidateRouter().route(
            question="test",
            attachment={"file_path": "data.csv", "extension": ".csv"},
            routing={"use_search": True, "use_attachment": True},
        )
        names = [candidate.tool_name for candidate in candidates]

        self.assertIn("attachment_reader", names)
        self.assertIn("search", names)
        self.assertIn("deterministic_handler", names)
        self.assertTrue(next(candidate for candidate in candidates if candidate.tool_name == "attachment_reader").required)

    def test_candidate_router_includes_video_transcript_for_youtube_url(self):
        candidates = ToolCandidateRouter().route(
            question="In https://www.youtube.com/watch?v=L1vXCYZAYYM, what is said?",
            attachment={},
            routing={"use_search": True},
        )
        video = next(candidate for candidate in candidates if candidate.tool_name == "video_transcript")

        self.assertTrue(video.required)
        self.assertIn("transcript", video.capability)

    def test_parser_repairs_markdown_and_aliases(self):
        raw = """
        ```json
        {"requires_tools": true, "tools": [{"tool": "search", "depends": []}], "stop_condition": "done"}
        ```
        """
        plan = ToolPlanParser().parse(raw)

        self.assertTrue(plan.requires_tools)
        self.assertEqual(plan.tool_sequence[0].tool_name, "search")
        self.assertIn("tools_to_tool_sequence", plan.repair_actions)
        self.assertIn("tool_to_tool_name", plan.repair_actions)

    def test_parser_reads_handler_plans(self):
        raw = """
        {"requires_tools": true,
         "tool_sequence": [{"tool_name": "deterministic_handler"}],
         "handler_plans": [{
           "handler_name": "simple_math",
           "status": "ready",
           "required_inputs": ["question"],
           "confidence": 0.9
         }]}
        """
        plan = ToolPlanParser().parse(raw)

        self.assertEqual(plan.handler_plans[0].handler_name, "simple_math")
        self.assertEqual(plan.handler_plans[0].status, "ready")
        self.assertEqual(plan.handler_plans[0].required_inputs, ["question"])

    def test_validator_removes_unknown_and_duplicate_tools(self):
        candidates = [
            ToolCandidate("attachment_reader", "read", required=True),
            ToolCandidate("search", "search"),
        ]
        plan = ToolPlan(
            requires_tools=True,
            tool_sequence=[
                ToolPlanStep("made_up_tool"),
                ToolPlanStep("search"),
                ToolPlanStep("search"),
            ],
            planner_source="slm",
        )

        validated = ToolPlanValidator().validate(plan, candidates)

        self.assertEqual([step.tool_name for step in validated.tool_sequence], ["attachment_reader", "search"])
        self.assertIn("unknown_tool_removed:made_up_tool", validated.validation_errors)
        self.assertIn("duplicate_tool_removed:search", validated.validation_errors)
        self.assertIn("insert_required_tool:attachment_reader", validated.repair_actions)

    def test_planning_runner_falls_back_when_slm_unavailable(self):
        runner = ToolPlanningRunner(slm_planner=SLMToolPlanner(model_name=""))
        result = runner.plan(
            question="Search this question.",
            attachment={},
            routing={"use_search": True},
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.validated_plan.tool_sequence[0].tool_name, "search")

    def test_planning_runner_attaches_deterministic_handler_plan(self):
        class FakeHandlerRouter:
            threshold = 0.5

            def match_handlers(self, handler_input):
                del handler_input
                return [
                    HandlerMatch(
                        handler_name="simple_math",
                        matched=True,
                        confidence=0.95,
                        reason="math signal",
                        required_inputs=["question"],
                    )
                ]

        runner = ToolPlanningRunner(
            slm_planner=SLMToolPlanner(model_name=""),
            deterministic_handler_router=FakeHandlerRouter(),
        )

        result = runner.plan(
            question="Compute 40 + 2.",
            attachment={},
            routing={"use_deterministic_solver": True},
        )

        self.assertEqual(result.validated_plan.handler_plans[0].handler_name, "simple_math")
        self.assertEqual(result.validated_plan.handler_plans[0].status, "ready")

    def test_planning_runner_uses_capability_registry_for_video(self):
        runner = ToolPlanningRunner(slm_planner=SLMToolPlanner(model_name=""))
        result = runner.plan(
            question="In https://www.youtube.com/watch?v=L1vXCYZAYYM, what is said?",
            attachment={},
            routing={"use_search": True},
        )

        self.assertIn(
            "video_transcript",
            [step.tool_name for step in result.validated_plan.tool_sequence],
        )
        self.assertTrue(result.validated_plan.tool_needs)

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
        attachment = next(candidate for candidate in candidates if candidate.tool_name == "attachment_reader")
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
        self.assertEqual(plan.tool_sequence[0].purpose, "recover deterministic handler missing inputs")

    def test_evidence_runner_uses_plan_and_can_skip_search(self):
        class FakePlanningRunner:
            def plan(self, **kwargs):
                plan = ToolPlan(
                    requires_tools=False,
                    tool_sequence=[],
                    planner_source="test",
                )
                return ToolPlanResult(
                    candidate_tools=[],
                    raw_planner_reply="",
                    parsed_plan=plan,
                    validated_plan=plan,
                    fallback_used=False,
                )

        runner = EvidenceRunner(
            question="Who is the president?",
            tool_planning_runner=FakePlanningRunner(),
            enable_tool_planner=True,
        )

        evidence = runner.run()

        self.assertEqual(evidence["search_result"], "")
        self.assertEqual(evidence["attachment_result"], "")
        self.assertEqual(evidence["solver_result"], "")
        self.assertEqual([item["tool_name"] for item in evidence["tool_usage"]], ["tool_planner"])
        self.assertTrue(evidence["routing"]["tool_planner_enabled"])

    def test_evidence_runner_drops_invalid_attachment_output_from_context(self):
        class FakePlanningRunner:
            def plan(self, **kwargs):
                plan = ToolPlan(
                    requires_tools=True,
                    tool_sequence=[ToolPlanStep("attachment_reader")],
                    planner_source="test",
                )
                return ToolPlanResult(
                    candidate_tools=[],
                    raw_planner_reply="",
                    parsed_plan=plan,
                    validated_plan=plan,
                    fallback_used=False,
                )

        class FakeAttachmentBuilder:
            def build(self, question, attachment):
                del question, attachment
                return {
                    "context": "",
                    "tool_usage": [
                        {
                            "ok": True,
                            "tool_name": "attachment_reader",
                            "output_text": "",
                            "raw_result": {},
                            "error": None,
                        }
                    ],
                }

        runner = EvidenceRunner(
            question="Read the attachment.",
            attachment={"file_path": "fake.txt"},
            tool_planning_runner=FakePlanningRunner(),
            attachment_evidence_builder=FakeAttachmentBuilder(),
            enable_tool_planner=True,
        )

        evidence = runner.run()

        self.assertEqual(evidence["attachment_result"], "")
        usage = next(item for item in evidence["tool_usage"] if item["tool_name"] == "attachment_reader")
        self.assertFalse(usage["validation"]["valid"])
        self.assertEqual(usage["validation"]["status"], "empty_output")

    def test_evidence_runner_executes_planned_deterministic_handler(self):
        class FakePlanningRunner:
            def plan(self, **kwargs):
                plan = ToolPlan(
                    requires_tools=True,
                    tool_sequence=[ToolPlanStep("deterministic_handler")],
                    handler_plans=[
                        HandlerPlan(
                            handler_name="simple_math",
                            status="ready",
                            required_inputs=["question"],
                            confidence=0.9,
                        )
                    ],
                    planner_source="test",
                )
                return ToolPlanResult(
                    candidate_tools=[],
                    raw_planner_reply="",
                    parsed_plan=plan,
                    validated_plan=plan,
                    fallback_used=False,
                )

        class FakeDeterministicRouter:
            def run(self, **kwargs):
                from tools.deterministic_handlers import HandlerResult

                self.handler_name = kwargs.get("handler_name")
                return HandlerResult(
                    handler_name="simple_math",
                    status="ok",
                    answer="42",
                    evidence_text="Deterministic handler evidence:\nAnswer: 42",
                    structured_result={
                        "handler_role": "simple_math",
                        "output_contract": {
                            "schema_version": "deterministic-handler-v1",
                            "required_outputs": ["answer"],
                        },
                    },
                    input_summary={"question": "Compute 40 + 2."},
                    output_type="final_answer",
                    semantic_role="simple_math_answer",
                    supporting_inputs=["question"],
                )

        runner = EvidenceRunner(
            question="Compute 40 + 2.",
            tool_planning_runner=FakePlanningRunner(),
            deterministic_handler_router=FakeDeterministicRouter(),
            enable_tool_planner=True,
            enable_deterministic_handler_router=True,
        )

        evidence = runner.run()

        self.assertIn("Answer: 42", evidence["solver_result"])
        self.assertIn(
            "deterministic_handler_router",
            [item["tool_name"] for item in evidence["tool_usage"]],
        )
        deterministic_usage = next(
            item
            for item in evidence["tool_usage"]
            if item["tool_name"] == "deterministic_handler_router"
        )
        self.assertEqual(deterministic_usage["planned_handler_name"], "simple_math")
        self.assertEqual(deterministic_usage["handler_plan"]["handler_name"], "simple_math")
        self.assertTrue(deterministic_usage["handler_trust"]["trusted"])

    def test_evidence_runner_blocks_untrusted_deterministic_handler_result(self):
        class FakePlanningRunner:
            def plan(self, **kwargs):
                plan = ToolPlan(
                    requires_tools=True,
                    tool_sequence=[ToolPlanStep("deterministic_handler")],
                    handler_plans=[
                        HandlerPlan(
                            handler_name="simple_math",
                            status="ready",
                            required_inputs=["question"],
                            confidence=0.9,
                        )
                    ],
                    planner_source="test",
                )
                return ToolPlanResult(
                    candidate_tools=[],
                    raw_planner_reply="",
                    parsed_plan=plan,
                    validated_plan=plan,
                    fallback_used=False,
                )

        class FakeDeterministicRouter:
            def run(self, **kwargs):
                from tools.deterministic_handlers import HandlerResult

                return HandlerResult(
                    handler_name="simple_math",
                    status="ok",
                    answer="unknown",
                    evidence_text="Deterministic handler evidence:\nAnswer: unknown",
                )

        runner = EvidenceRunner(
            question="Compute 40 + 2.",
            tool_planning_runner=FakePlanningRunner(),
            deterministic_handler_router=FakeDeterministicRouter(),
            enable_tool_planner=True,
            enable_deterministic_handler_router=True,
        )

        evidence = runner.run()

        self.assertEqual(evidence["solver_result"], "")
        deterministic_usage = next(
            item
            for item in evidence["tool_usage"]
            if item["tool_name"] == "deterministic_handler_router"
        )
        self.assertFalse(deterministic_usage["handler_trust"]["trusted"])
        self.assertEqual(deterministic_usage["handler_trust"]["status"], "invalid_candidate")

    def test_evidence_runner_executes_planned_video_transcript(self):
        class FakePlanningRunner:
            def plan(self, **kwargs):
                plan = ToolPlan(
                    requires_tools=True,
                    tool_sequence=[ToolPlanStep("video_transcript")],
                    planner_source="test",
                )
                return ToolPlanResult(
                    candidate_tools=[],
                    raw_planner_reply="",
                    parsed_plan=plan,
                    validated_plan=plan,
                    fallback_used=False,
                )

        class FakeVideoTool:
            def run(self, parameters):
                return "Video transcript source: test\n[00:01] hello from video"

        runner = EvidenceRunner(
            question="In https://www.youtube.com/watch?v=L1vXCYZAYYM, what is said?",
            tool_planning_runner=FakePlanningRunner(),
            enable_tool_planner=True,
        )
        runner.video_transcript_tool = FakeVideoTool()

        evidence = runner.run()

        self.assertIn("Video Evidence", evidence["attachment_result"])
        self.assertIn("hello from video", evidence["attachment_result"])
        self.assertIn("video_transcript", [item["tool_name"] for item in evidence["tool_usage"]])


if __name__ == "__main__":
    unittest.main()
