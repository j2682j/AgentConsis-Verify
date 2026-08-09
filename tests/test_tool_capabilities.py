from __future__ import annotations

import json
import unittest

from core.config import AgentConfig
from core.stage1_tool_use_runner import Stage1ToolUseRunner
from tools.base import Tool, ToolParameter
from tools.tool_manager import ToolManager


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="dynamic_echo",
            description="Echo a value for dynamic registry testing.",
            capabilities={"test.echo"},
            deterministic=True,
        )

    def run(self, parameters):
        return {"echo": str(parameters.get("input", ""))}

    def get_parameters(self):
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Value to echo.",
                required=True,
            )
        ]


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def invoke_with_usage(self, messages, **_overrides):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            reply = {
                "type": "tool_request",
                "reasoning_step": "step 1. use the registered echo capability",
                "tool_name": "dynamic_echo",
                "tool_args": {"input": "hello"},
            }
        else:
            reply = {
                "type": "final_answer",
                "reasoning": "step 1. The dynamic tool returned the requested value.",
                "final_answer": "hello",
            }
        return json.dumps(reply), 10, 5


class ToolCapabilityTests(unittest.TestCase):
    def test_gap_detector_matches_new_specialized_capabilities(self):
        manager = ToolManager()

        report = manager.detect_tool_gap(
            "Use Boggle DFS to find words in this letter grid, then calculate "
            "the shortest path between stations."
        )

        self.assertIn("grid.word_search", report["matched"])
        self.assertIn("graph.shortest_path", report["matched"])
        self.assertFalse(report["has_gap"])

    def test_gap_detector_matches_existing_math_and_attachment_capabilities(self):
        manager = ToolManager()

        report = manager.detect_tool_gap(
            "Calculate the average values in the spreadsheet.",
            attachment_type="xlsx",
        )

        self.assertIn("math.statistics", report["matched"])
        self.assertIn("attachment.table", report["matched"])
        self.assertIn("table.statistics", report["matched"])

    def test_runtime_registered_tool_is_available_to_stage1_without_allowlist_change(self):
        manager = ToolManager()
        manager.register_tool(EchoTool())
        manager.enabled_tools.add("dynamic_echo")
        runner = Stage1ToolUseRunner(
            tool_manager=manager,
            max_tool_turns=1,
        )
        agent = FakeAgent()

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Echo hello using the available tool.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(reply.final_answer, "hello")
        self.assertEqual(reply.tool_results[0]["tool_name"], "dynamic_echo")
        self.assertEqual(reply.tool_results[0]["status"], "success")
        first_prompt = agent.messages[0][1]["content"]
        self.assertIn("dynamic_echo", first_prompt)
        self.assertIn("test.echo", first_prompt)

    def test_unknown_requested_tool_returns_capability_gap(self):
        manager = ToolManager()

        result = manager.execute_tool(
            "determine_density",
            {"input": "determine density from mass and volume"},
            agent_id="a1",
            stage="stage1",
        )

        self.assertEqual(result["status"], "unsupported")
        self.assertIn("physics.density", result["raw_result"]["tool_gap"]["missing"])


if __name__ == "__main__":
    unittest.main()
