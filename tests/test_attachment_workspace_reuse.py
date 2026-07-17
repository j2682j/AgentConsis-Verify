from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.attachment_reader_tool import AttachmentReaderTool
from tools.attachment_strategy import AttachmentStrategy, AttachmentStrategyExecutor
from tools.attachment_tool_coordinator import AttachmentToolCoordinator
from tools.attachment_workspace import AttachmentWorkspace
from tools.base import Tool
from tools.tool_manager import ToolManager
from core.config import AgentConfig
from core.network import Network


class StaticPlanner:
    def __init__(self, strategy: AttachmentStrategy) -> None:
        self.strategy = strategy
        self.calls = 0

    def plan(self, *, question, attachment_profile, allowed_handlers):
        del question, attachment_profile, allowed_handlers
        self.calls += 1
        return self.strategy, '{"planned":true}'


class FailingAttachmentBuilder:
    def build(self, question, attachment):
        del question, attachment
        raise AssertionError("prepared payload should prevent a second reader call")


class ContextAwareTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="context_aware", description="test")
        self.seen_context = None

    def run(self, parameters):
        raise AssertionError("run_with_context should be preferred")

    def run_with_context(self, parameters, runtime_context):
        self.seen_context = runtime_context
        return {"value": parameters.get("input")}


class AttachmentWorkspaceReuseTests(unittest.TestCase):
    def _xlsx_artifact(self):
        from openpyxl import Workbook

        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "records.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["name", "value"])
        sheet.append(["A", 10])
        sheet.append(["B", 20])
        workbook.save(path)
        attachment = {"file_path": str(path), "extension": ".xlsx"}
        read_result = AttachmentEvidenceBuilder().build(
            "How many records are in the table?",
            attachment,
        )
        workspace = AttachmentWorkspace(attachment)
        workspace.seed_from_read_result(read_result)
        return temporary, attachment, read_result, workspace

    def test_workspace_keeps_full_typed_payload_but_snapshot_is_compact(self):
        temporary, _, read_result, workspace = self._xlsx_artifact()
        try:
            artifact = workspace.artifact()
            self.assertIsNotNone(artifact)
            self.assertEqual(
                artifact.parsed_payload["tables"][0]["rows"],
                read_result["parsed_payload"]["tables"][0]["rows"],
            )
            snapshot = workspace.snapshot()
            self.assertTrue(snapshot["prepared_available"])
            self.assertIn("table", snapshot["available_inputs"])
            self.assertNotIn("parsed_payload", snapshot)
            self.assertEqual(snapshot["reader_execution_count"], 1)
        finally:
            temporary.cleanup()

    def test_stage1_coordinator_reuses_xlsx_payload_and_handler_result(self):
        temporary, attachment, _, workspace = self._xlsx_artifact()
        planner = StaticPlanner(
            AttachmentStrategy(
                information_need="Count table records.",
                required_handler="table_exact_operations",
                required_inputs=["rows"],
                expected_answer="record count",
            )
        )
        executor = AttachmentStrategyExecutor(
            attachment_builder=FailingAttachmentBuilder(),
            planner=planner,
        )
        coordinator = AttachmentToolCoordinator(strategy_executor=executor)
        try:
            first = coordinator.run(
                question="How many records are in the table?",
                information_need="the number of table records",
                attachment=attachment,
                workspace=workspace,
            )
            second = coordinator.run(
                question="How many records are in the table?",
                information_need="the number of table records",
                attachment=attachment,
                workspace=workspace,
            )
        finally:
            temporary.cleanup()

        self.assertTrue(first["evidence_valid"])
        self.assertEqual(first["handler_name"], "table_exact_operations")
        self.assertEqual(first["value"], "2")
        self.assertEqual(first["attachment_reuse"]["action"], "reuse_prepared_payload")
        self.assertEqual(second["status"], "already_available")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(planner.calls, 1)
        snapshot = workspace.snapshot()
        self.assertEqual(snapshot["reader_execution_count"], 1)
        self.assertEqual(snapshot["payload_reuse_count"], 1)
        self.assertEqual(snapshot["result_cache_hit_count"], 1)
        self.assertEqual(snapshot["handler_execution_count"], 1)

    def test_attachment_reader_tool_uses_workspace_runtime_context(self):
        temporary, attachment, _, workspace = self._xlsx_artifact()
        planner = StaticPlanner(
            AttachmentStrategy(
                information_need="Count table records.",
                required_handler="table_exact_operations",
                required_inputs=["rows"],
                expected_answer="record count",
            )
        )
        tool = AttachmentReaderTool(
            builder=FailingAttachmentBuilder(),
            coordinator=AttachmentToolCoordinator(
                strategy_executor=AttachmentStrategyExecutor(
                    attachment_builder=FailingAttachmentBuilder(),
                    planner=planner,
                )
            ),
        )
        try:
            result = tool.run_with_context(
                {"information_need": "the number of table records", "attachment": attachment},
                {
                    "attachment_workspace": workspace,
                    "question": "How many records are in the table?",
                },
            )
        finally:
            temporary.cleanup()

        self.assertTrue(result["evidence_valid"])
        self.assertEqual(result["value"], "2")

    def test_tool_manager_passes_runtime_context_only_to_contextual_tools(self):
        manager = ToolManager()
        tool = ContextAwareTool()
        manager.register_tool(tool)
        manager.enabled_tools.add(tool.name)
        runtime_context = {"task": "one"}

        result = manager.execute_tool(
            tool.name,
            {"input": "hello"},
            agent_id="a1",
            stage="stage1",
            runtime_context=runtime_context,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(tool.seen_context, runtime_context)

    def test_workspaces_do_not_share_cached_results(self):
        first = AttachmentWorkspace({"file_path": "first.txt"})
        second = AttachmentWorkspace({"file_path": "second.txt"})
        first.record_result(
            "one fact",
            {"ok": True, "evidence_valid": True, "output_text": "first"},
        )

        first_decision = first.decide("one fact")
        second_decision = second.decide("one fact")

        self.assertEqual(first_decision.action, "reuse_cached_result")
        self.assertEqual(second_decision.action, "reader_required")

    def test_network_shares_one_workspace_between_evidence_and_stage1(self):
        network = Network(
            question="Read the attachment.",
            agents=[AgentConfig(agent_id="a1", model_name="fake")],
            attachment={"file_path": "sample.txt", "extension": ".txt"},
            enable_stage2_score=False,
        )

        self.assertIs(network.evidence_runner.attachment_workspace, network.attachment_workspace)
        self.assertIs(network.stage1_runner.attachment_workspace, network.attachment_workspace)


if __name__ == "__main__":
    unittest.main()
