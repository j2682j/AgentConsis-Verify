from __future__ import annotations

import json
import unittest

from core.config import AgentConfig
from core.stage1_search_gate import Stage1SearchAccessState
from core.stage1_runner import Stage1Runner
from core.stage1_tool_use_runner import Stage1ToolUseRunner


def prepared_evidence() -> dict:
    return {
        "search_result": "[E1] Source: Example\nEvidence: prepared fact",
        "routing": {"use_search": True},
        "tool_usage": [
            {
                "tool_name": "search",
                "ok": True,
                "status": "success",
                "evidence_valid": True,
                "output_text": "[E1] prepared fact",
                "raw_result": {
                    "queries": ["prepared query"],
                    "evidence_items": [
                        {"evidence_id": "E1", "content": "prepared fact"}
                    ],
                    "retrieval": {"searched_queries": ["prepared query"]},
                },
            }
        ],
    }


def tool_request(query: str, missing_information: str = "") -> dict:
    tool_args = {"input": query, "mode": "text"}
    if missing_information:
        tool_args["missing_information"] = missing_information
    return {
        "type": "tool_request",
        "reasoning_step": "step 1. Obtain the one missing fact.",
        "tool_name": "search",
        "tool_args": tool_args,
    }


def final_answer(answer: str = "done") -> dict:
    return {
        "type": "final_answer",
        "reasoning_steps": ["step 1. Use the available evidence."],
        "final_answer": answer,
        "confidence": 0.8,
        "used_evidence_ids": ["E1"],
        "answer_type": "short_text",
        "tool_request": None,
    }


class ScriptedAgent:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.messages: list[list[dict[str, str]]] = []
        self.sampling_overrides: list[dict] = []

    def invoke_with_usage(self, messages, **overrides):
        self.messages.append(messages)
        self.sampling_overrides.append(overrides)
        return json.dumps(self.replies.pop(0)), 10, 5


class CountingToolManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def describe_enabled_tools(self):
        return (
            "- search: test search capabilities=[web.search] args=[input:string*]\n"
            "- python_calculator: calculator capabilities=[math] args=[input:string*]"
        )

    def format_tool_gap(self, question, *, attachment_type=None):
        del question, attachment_type
        return "Missing capabilities: ['none']"

    def execute_tool(self, tool_name, parameters, *, agent_id, stage):
        del agent_id, stage
        self.calls.append((tool_name, dict(parameters)))
        return {
            "ok": True,
            "tool_name": tool_name,
            "status": "success",
            "output_text": f"supplemental evidence for {parameters.get('input', '')}",
            "raw_result": {"results": [{"title": "result"}]},
            "error": None,
            "error_code": "",
            "error_message": "",
            "retryable": False,
            "retry_hint": "",
            "evidence_valid": True,
            "cache_hit": False,
            "duplicate_request": False,
        }


class Stage1SearchGateTests(unittest.TestCase):
    def test_builds_usable_state_from_prepared_search_contract(self):
        state = Stage1SearchAccessState.from_evidence(prepared_evidence())

        self.assertEqual(state.prepared_status, "prepared_usable")
        self.assertTrue(state.prepared_evidence_available)
        self.assertEqual(state.prepared_queries, ["prepared query"])
        self.assertEqual(state.prepared_evidence_ids, ["E1"])

    def test_prepared_query_is_blocked_without_backend_call(self):
        state = Stage1SearchAccessState.from_evidence(prepared_evidence())
        manager = CountingToolManager()
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)
        agent = ScriptedAgent(
            [tool_request("prepared query", "the missing date"), final_answer()]
        )

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(manager.calls, [])
        self.assertEqual(reply.tool_results[0]["status"], "already_available")
        self.assertEqual(state.snapshot()["blocked_reasons"], {"query_already_prepared": 1})

    def test_missing_information_is_required_for_refinement(self):
        state = Stage1SearchAccessState.from_evidence(prepared_evidence())
        manager = CountingToolManager()
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)
        agent = ScriptedAgent([tool_request("new query"), final_answer()])

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual(manager.calls, [])
        self.assertEqual(reply.tool_results[0]["error_code"], "missing_information_required")

    def test_one_refinement_is_shared_across_agents_and_later_context(self):
        """The shared pool itself, with the per-agent floor stood down.

        By default each Agent now holds a reserved refinement before the shared
        pool applies, so two Agents asking once each are both granted; that
        behaviour is pinned in test_search_gate_per_agent_floor. This case keeps
        the shared pool under test on its own, including the supplemental
        evidence a blocked Agent inherits from the one that got through.
        """

        state = Stage1SearchAccessState.from_evidence(
            prepared_evidence(),
            refinement_budget=1,
            per_agent_refinement_floor=0,
        )
        manager = CountingToolManager()
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)
        first = ScriptedAgent(
            [tool_request("new query one", "the missing year"), final_answer("first")]
        )
        second = ScriptedAgent(
            [tool_request("new query two", "the missing location"), final_answer("second")]
        )

        first_reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=first,
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )
        second_reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a2", model_name="fake"),
            agent=second,
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )

        self.assertTrue(first_reply.parse_completed)
        self.assertTrue(second_reply.parse_completed)
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(second_reply.tool_results[0]["error_code"], "refinement_budget_exhausted")
        self.assertIn("supplemental evidence for new query one", second.messages[0][1]["content"])
        snapshot = state.snapshot()
        self.assertEqual(snapshot["physical_execution_count"], 1)
        self.assertEqual(snapshot["refinement_used"], 1)

    def test_same_supplemental_query_uses_shared_result(self):
        state = Stage1SearchAccessState.from_evidence(prepared_evidence(), refinement_budget=1)
        manager = CountingToolManager()
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)
        first = ScriptedAgent(
            [tool_request("new query", "the missing year"), final_answer("first")]
        )
        second = ScriptedAgent(
            [tool_request("new query", "the missing year"), final_answer("second")]
        )

        for index, agent in enumerate((first, second), start=1):
            runner.run(
                config=AgentConfig(agent_id=f"a{index}", model_name="fake"),
                agent=agent,
                question="Find the answer.",
                evidence_packets=[],
                run_index=1,
                search_access_state=state,
            )

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(state.snapshot()["cache_hit_count"], 1)

    def test_no_prepared_evidence_keeps_existing_search_behavior(self):
        state = Stage1SearchAccessState.from_evidence(
            {"search_result": "", "routing": {}, "tool_usage": []}
        )
        manager = CountingToolManager()
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)
        agent = ScriptedAgent([tool_request("normal query"), final_answer()])

        runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(state.snapshot()["physical_execution_count"], 1)

    def test_stage1_runner_builds_one_task_scoped_gate_from_evidence(self):
        manager = CountingToolManager()
        agent = ScriptedAgent(
            [tool_request("new query", "the missing year"), final_answer()]
        )
        config = AgentConfig(agent_id="a1", model_name="fake")
        runner = Stage1Runner(
            question="Find the answer.",
            agents=[config],
            get_agent=lambda _: agent,
            record_token_usage=lambda **_: None,
            stage1_runs_per_agent=1,
            max_workers=1,
            enable_tool_use=True,
            max_tool_turns=2,
            tool_manager=manager,
            unload_previous_slm_on_switch=False,
            prepared_search_refinement_budget=0,
        )

        summaries = runner.run(prepared_evidence())

        self.assertEqual(len(summaries), 1)
        self.assertEqual(manager.calls, [])
        metadata = runner.search_gate_metadata()
        self.assertEqual(metadata["prepared_status"], "prepared_usable")
        self.assertEqual(metadata["refinement_budget"], 0)
        self.assertEqual(metadata["blocked_reasons"], {"refinement_budget_exhausted": 1})

    def test_search_gate_does_not_block_non_search_tools(self):
        state = Stage1SearchAccessState.from_evidence(
            prepared_evidence(),
            refinement_budget=0,
        )
        manager = CountingToolManager()
        agent = ScriptedAgent(
            [
                {
                    "type": "tool_request",
                    "reasoning_step": "step 1. Calculate the required value.",
                    "tool_name": "python_calculator",
                    "tool_args": {"input": "2 + 2"},
                },
                final_answer("4"),
            ]
        )
        runner = Stage1ToolUseRunner(tool_manager=manager, max_tool_turns=2)

        reply, _, _ = runner.run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=agent,
            question="Calculate the answer.",
            evidence_packets=[],
            run_index=1,
            search_access_state=state,
        )

        self.assertTrue(reply.parse_completed)
        self.assertEqual([name for name, _ in manager.calls], ["python_calculator"])
        self.assertEqual(state.snapshot()["request_count"], 0)

    def test_negative_budget_disables_gate(self):
        state = Stage1SearchAccessState.from_evidence(
            prepared_evidence(),
            refinement_budget=-1,
        )

        decision = state.authorize(query="prepared query", missing_information="")

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "legacy_search_access")
        self.assertFalse(state.snapshot()["enabled"])


if __name__ == "__main__":
    unittest.main()
