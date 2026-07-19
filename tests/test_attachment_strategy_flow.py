from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from context.context_builder import ContextPacket
from context.stage1_context import Stage1ContextBuilder
from core.evidence_runner import EvidenceRunner
from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.attachment_strategy import AttachmentStrategy, AttachmentStrategyExecutor
from tools.deterministic_handlers import DeterministicHandlerRouter


class _RecordingPlanner:
    def __init__(self, strategy: AttachmentStrategy) -> None:
        self.strategy = strategy
        self.received_profile: dict = {}

    def plan(self, *, question, attachment_profile, allowed_handlers):
        self.received_profile = dict(attachment_profile)
        return self.strategy, '{"planned": true}'


class _FailingPlanner:
    def plan(self, *, question, attachment_profile, allowed_handlers):
        raise RuntimeError("planner unavailable")


class _InvalidPlanner:
    def plan(self, *, question, attachment_profile, allowed_handlers):
        return AttachmentStrategy(missing_inputs=["invalid_strategy_json"]), "not-json"


class AttachmentStrategyFlowTests(unittest.TestCase):
    def test_stage1_prompt_receives_natural_language_answer_requirement(self):
        messages = Stage1ContextBuilder().build(
            question="How many records are present?",
            evidence_packets=[
                ContextPacket(
                    packet_type="answer_requirement",
                    content="Return the number of records.",
                    priority=95,
                )
            ],
        )

        self.assertIn("Answer_Requirement:\nReturn the number of records.", messages[1]["content"])

    def test_attachment_builder_returns_profile_and_parsed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("Alpha evidence\nBeta evidence\n", encoding="utf-8")

            result = AttachmentEvidenceBuilder().build(
                "Which value is present?",
                {"file_path": str(path), "file_name": path.name, "extension": ".txt"},
            )

        self.assertTrue(result["used"])
        self.assertEqual(result["profile"]["parse_status"], "success")
        self.assertIn("text", result["profile"]["content_types"])
        self.assertIn("Alpha evidence", result["profile"]["content_preview"])
        self.assertEqual(
            result["parsed_payload"]["provenance"]["source"],
            "attachment_reader",
        )

    def test_typed_csv_payload_exposes_rows_without_inferred_graph_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            path.write_text("name,value\nA,10\nB,20\n", encoding="utf-8")

            result = AttachmentEvidenceBuilder().build(
                "How many records are present?",
                {"file_path": str(path), "extension": ".csv"},
            )

        self.assertEqual(result["parsed_payload"]["tables"][0]["columns"], ["name", "value"])
        self.assertEqual(len(result["parsed_payload"]["tables"][0]["rows"]), 2)
        self.assertIn("rows", result["profile"]["available_inputs"])
        self.assertNotIn("edges", result["profile"]["available_inputs"])
        self.assertNotIn("pairs", result["profile"]["available_inputs"])

    def test_pdb_payload_exposes_atom_coordinates_without_fake_pairs(self):
        pdb_text = (
            "ATOM      1  N   ALA A   1      11.104  13.207   9.123  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1      12.560  13.100   9.456  1.00 20.00           C\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdb"
            path.write_text(pdb_text, encoding="utf-8")

            result = AttachmentEvidenceBuilder().build(
                "Inspect the structure.",
                {"file_path": str(path), "extension": ".pdb"},
            )

        self.assertEqual(len(result["parsed_payload"]["coordinates"]), 2)
        self.assertIn("coordinates", result["profile"]["available_inputs"])
        self.assertNotIn("pairs", result["profile"]["available_inputs"])

    def test_prose_with_commas_does_not_create_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("Alpha, beta, and gamma are ordinary prose.\nDelta, epsilon, and zeta.", encoding="utf-8")

            result = AttachmentEvidenceBuilder().build(
                "Summarize the note.",
                {"file_path": str(path), "extension": ".txt"},
            )

        self.assertNotIn("edges", result["profile"]["available_inputs"])
        self.assertNotIn("list_items", result["profile"]["available_inputs"])

    def test_executor_parses_attachment_before_planning(self):
        planner = _RecordingPlanner(
            AttachmentStrategy(
                information_need="Find the named value.",
                required_handler="",
                expected_answer="short text",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.txt"
            path.write_text("The named value is Aurora.", encoding="utf-8")
            executor = AttachmentStrategyExecutor(planner=planner)

            result = executor.run(
                question="What is the named value?",
                attachment={"file_path": str(path), "extension": ".txt"},
            )

        self.assertIn("Aurora", planner.received_profile["content_preview"])
        self.assertEqual(result.attachment_profile["parse_status"], "success")
        self.assertEqual(
            [item["tool_name"] for item in result.tool_usage[:2]],
            ["attachment_reader", "attachment_strategy_planner"],
        )

    def test_executor_preserves_attachment_when_planner_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonld"
            path.write_text('{"name":"Aurora","value":17}', encoding="utf-8")
            executor = AttachmentStrategyExecutor(planner=_FailingPlanner())

            result = executor.run(
                question="What is the named value?",
                attachment={"file_path": str(path), "extension": ".jsonld"},
            )

        self.assertEqual(result.reader_status, "success")
        self.assertEqual(result.strategy_status, "failed")
        self.assertIn("Aurora", result.attachment_context)
        self.assertEqual(result.handler_status, "not_required")

    def test_evidence_runner_preserves_json_attachment_when_strategy_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonld"
            path.write_text('{"name":"Aurora","value":17}', encoding="utf-8")
            executor = AttachmentStrategyExecutor(planner=_InvalidPlanner())
            evidence = EvidenceRunner(
                question="What is the named value?",
                attachment={"file_path": str(path), "extension": ".jsonld"},
                attachment_strategy_executor=executor,
            ).run()

        self.assertIn("Aurora", evidence["attachment_result"])
        self.assertEqual(
            evidence["attachment_strategy"]["strategy_status"],
            "invalid_output",
        )

    def test_attachment_provenance_blocks_table_handler_without_rows(self):
        router = DeterministicHandlerRouter()
        result = router.run(
            question="How many records are in the table?",
            attachment={"file_path": "notes.txt", "extension": ".txt"},
            attachment_result="A prose paragraph with 1, 2, and 3 in it.",
            handler_name="table_exact_operations",
            metadata={
                "require_attachment_provenance": True,
                "attachment_profile": {
                    "parse_status": "success",
                    "content_types": ["text"],
                    "available_inputs": ["question", "attachment_context", "source_text"],
                },
                "parsed_payload": {
                    "provenance": {
                        "source": "attachment_reader",
                        "parse_status": "success",
                    }
                },
            },
        )

        self.assertEqual(result.status, "missing_inputs")
        self.assertIn("rows", result.missing_inputs)

    def test_csv_handler_runs_with_profile_and_is_trusted(self):
        planner = _RecordingPlanner(
            AttachmentStrategy(
                information_need="Count the table records.",
                required_handler="table_exact_operations",
                required_inputs=["rows"],
                expected_answer="number of records",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            path.write_text("name,value\nA,10\nB,20\n", encoding="utf-8")
            executor = AttachmentStrategyExecutor(planner=planner)

            result = executor.run(
                question="How many records are in the table?",
                attachment={"file_path": str(path), "extension": ".csv"},
            )

        handler_usage = [
            item for item in result.tool_usage if item.get("tool_name") == "attachment_strategy_handler"
        ]
        self.assertEqual(len(handler_usage), 1)
        self.assertTrue(handler_usage[0]["evidence_valid"])
        self.assertEqual(handler_usage[0]["raw_result"]["answer"], "2")
        self.assertIn("Answer: 2", result.solver_context)

    def test_evidence_runner_exports_profile_requirement_and_verified_result(self):
        planner = _RecordingPlanner(
            AttachmentStrategy(
                information_need="Count the table records.",
                required_handler="table_exact_operations",
                required_inputs=["rows"],
                expected_answer="Return the number of records.",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            path.write_text("name,value\nA,10\nB,20\n", encoding="utf-8")
            executor = AttachmentStrategyExecutor(planner=planner)
            evidence = EvidenceRunner(
                question="How many records are in the table?",
                attachment={"file_path": str(path), "extension": ".csv"},
                attachment_strategy_executor=executor,
            ).run()

        self.assertEqual(evidence["answer_requirement"], "Return the number of records.")
        self.assertEqual(evidence["attachment_profile"]["parse_status"], "success")
        self.assertIn("Answer: 2", evidence["solver_result"])
        self.assertFalse(evidence["routing"]["search_decision"]["search_executed"])


if __name__ == "__main__":
    unittest.main()
