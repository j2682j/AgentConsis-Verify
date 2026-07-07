from __future__ import annotations

import json
import unittest

from core.config import AgentConfig
from core.stage1_trajectory_runner import Stage1TrajectoryRunner
from core.tool_turn_policy import AdaptiveToolTurnPolicy


class ScriptedAgent:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def invoke_with_usage(self, messages):
        self.messages.append(messages)
        reply = self.replies.pop(0)
        return json.dumps(reply), 10, 5


class ScriptedToolManager:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def describe_enabled_tools(self):
        return "- search: test search capabilities=[web.search] args=[input:string*]"

    def format_tool_gap(self, question, *, attachment_type=None):
        del question, attachment_type
        return "Required capabilities: ['web.search']\nMissing capabilities: ['none']"

    def execute_tool(self, tool_name, parameters, *, agent_id, stage):
        del parameters, agent_id, stage
        self.calls += 1
        result = dict(self.results.pop(0))
        result.setdefault("tool_name", tool_name)
        result.setdefault("cache_hit", False)
        result.setdefault("duplicate_request", False)
        return result


def tool_request(index):
    return {
        "type": "tool_request",
        "reasoning_step": f"step {index}. gather new evidence",
        "tool_name": "search",
        "tool_args": {"input": f"query {index}"},
    }


def final_answer():
    return {
        "type": "final_answer",
        "reasoning": "step 1. Use the accumulated evidence.",
        "final_answer": "done",
    }


def success_result(index):
    return {
        "ok": True,
        "status": "success",
        "output_text": f"new evidence {index}",
        "raw_result": {"results": [{"title": f"result {index}"}]},
        "error": None,
        "error_code": "",
        "error_message": "",
        "retryable": False,
        "retry_hint": "",
        "evidence_valid": True,
    }


def no_progress_result():
    return {
        "ok": True,
        "status": "partial",
        "output_text": "",
        "raw_result": {"results": []},
        "error": "search returned no results",
        "error_code": "search_no_results",
        "error_message": "search returned no results",
        "retryable": True,
        "retry_hint": "Change the query terms before retrying.",
        "evidence_valid": False,
    }


class AdaptiveToolTurnTests(unittest.TestCase):
    def test_policy_extends_budget_when_new_evidence_arrives(self):
        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)

        self.assertTrue(policy.observe(success_result(1)))
        self.assertEqual(policy.allowed_budget, 3)
        self.assertTrue(policy.observe(success_result(2)))
        self.assertEqual(policy.allowed_budget, 4)
        self.assertFalse(policy.force_final)

        policy.observe(success_result(3))
        policy.observe(success_result(4))
        self.assertTrue(policy.force_final)
        self.assertEqual(policy.stop_reason, "hard_tool_turn_limit")

    def test_policy_treats_repeated_content_as_no_progress(self):
        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)
        repeated = success_result(1)

        self.assertTrue(policy.observe(repeated))
        self.assertFalse(policy.observe(repeated))
        self.assertEqual(policy.no_progress_streak, 1)

    def test_trajectory_extends_beyond_base_budget_on_progress(self):
        manager = ScriptedToolManager(
            [success_result(1), success_result(2), success_result(3)]
        )
        agent = ScriptedAgent(
            [tool_request(1), tool_request(2), tool_request(3), final_answer()]
        )
        runner = Stage1TrajectoryRunner(
            tool_manager=manager,
            max_tool_turns=2,
            hard_max_tool_turns=4,
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find information using search.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(manager.calls, 3)
        self.assertEqual(reply.final_answer, "done")
        self.assertEqual(
            reply.tool_results[-1]["tool_turn_policy"]["allowed_budget"],
            4,
        )

    def test_trajectory_stops_after_two_no_progress_results(self):
        manager = ScriptedToolManager([no_progress_result(), no_progress_result()])
        agent = ScriptedAgent(
            [tool_request(1), tool_request(2), final_answer()]
        )
        runner = Stage1TrajectoryRunner(
            tool_manager=manager,
            max_tool_turns=2,
            hard_max_tool_turns=4,
            no_progress_limit=2,
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find information using search.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(manager.calls, 2)
        policy = reply.tool_results[-1]["tool_turn_policy"]
        self.assertTrue(policy["force_final"])
        self.assertEqual(policy["stop_reason"], "consecutive_no_progress")
        final_prompt = agent.messages[-1][1]["content"]
        self.assertIn("Return final_answer now", final_prompt)

    def test_trajectory_retries_final_answer_after_invalid_reply(self):
        manager = ScriptedToolManager([])
        agent = ScriptedAgent([{}, final_answer()])
        runner = Stage1TrajectoryRunner(
            tool_manager=manager,
            max_tool_turns=2,
            hard_max_tool_turns=4,
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Answer directly.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(reply.final_answer, "done")
        self.assertEqual(len(agent.messages), 2)
        self.assertTrue(reply.trajectory[0]["retry_final_answer"])
        final_prompt = agent.messages[-1][1]["content"]
        self.assertIn("Return final_answer now", final_prompt)

    def test_trajectory_repairs_invalid_final_answer_once(self):
        manager = ScriptedToolManager([])
        agent = ScriptedAgent(
            [
                {
                    "type": "final_answer",
                    "reasoning_steps": ["step 1. I do not have enough evidence."],
                    "final_answer": "unknown",
                    "tool_request": None,
                },
                {
                    "type": "final_answer",
                    "reasoning_steps": ["step 1. Use the available evidence."],
                    "final_answer": "Tokyo",
                    "tool_request": None,
                },
            ]
        )
        runner = Stage1TrajectoryRunner(
            tool_manager=manager,
            max_tool_turns=2,
            hard_max_tool_turns=4,
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Answer directly.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(reply.final_answer, "Tokyo")
        self.assertEqual(reply.final_answer_source, "repair_turn")
        self.assertTrue(reply.repair_metadata["attempted"])
        self.assertTrue(reply.repair_metadata["success"])
        self.assertIn("refusal_like_final_answer", reply.repair_metadata["reason"])
        final_prompt = agent.messages[-1][1]["content"]
        self.assertIn("Previous_Invalid_Reply", final_prompt)

    def test_repair_turn_tool_request_is_not_treated_as_final_answer(self):
        manager = ScriptedToolManager([no_progress_result()])
        agent = ScriptedAgent([tool_request(1), tool_request(2)])
        runner = Stage1TrajectoryRunner(
            tool_manager=manager,
            max_tool_turns=2,
            hard_max_tool_turns=4,
            no_progress_limit=1,
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find information using search.",
            evidence_packets=[],
            run_index=1,
        )

        self.assertFalse(reply.parse_completed)
        self.assertEqual(reply.final_answer, "")
        self.assertEqual(reply.final_answer_source, "none")
        self.assertIn("tool_trajectory_no_final_answer", reply.validity_labels)
        self.assertIn("final_answer_repair_failed", reply.validity_labels)
        self.assertEqual(manager.calls, 1)


if __name__ == "__main__":
    unittest.main()
