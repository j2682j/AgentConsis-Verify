from __future__ import annotations

import unittest

from parsers import Stage1OutputParser, Stage1ReplyParser, ToolRequestParser


class Stage1StructuredOutputTests(unittest.TestCase):
    def test_parses_valid_json_schema(self):
        parser = Stage1OutputParser()

        parsed = parser.parse(
            """
            {
              "reasoning_steps": ["step 1. E1 gives the value."],
              "final_answer": "42",
              "confidence": 0.9,
              "used_evidence_ids": ["E1"],
              "answer_type": "number",
              "tool_request": null
            }
            """
        )

        self.assertTrue(parsed["parse_completed"])
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["eligible_for_winner"])
        self.assertEqual(parsed["final_answer"], "42")
        self.assertEqual(parsed["structured_output"]["used_evidence_ids"], ["E1"])

    def test_repairs_legacy_text_output_without_scoring_penalty(self):
        parser = Stage1ReplyParser()

        parsed = parser.parse(
            """
            REASONING =
            step 1. E2 states the relevant result.
            step 2. Use that value as the answer.
            FINAL_ANSWER = Tokyo
            """
        )

        self.assertTrue(parsed["parse_completed"])
        self.assertTrue(parsed["repair_applied"])
        self.assertIn("extract_final_answer_label", parsed["repair_actions"])
        self.assertEqual(parsed["final_answer"], "Tokyo")
        self.assertEqual(parsed["structured_output"]["used_evidence_ids"], ["E2"])

    def test_tool_request_is_not_treated_as_final_answer(self):
        parser = ToolRequestParser()

        parsed = parser.parse(
            '{"type":"tool_request","reasoning_step":"step 1. Search first.","tool_name":"search","tool_args":{"input":"query"}}'
        )

        self.assertEqual(parsed["type"], "tool_request")
        self.assertEqual(parsed["final_answer"], "")
        self.assertFalse(parsed["eligible_for_winner"])
        self.assertEqual(parsed["validity_labels"], ["tool_request_pending"])

    def test_refusal_answer_is_repaired_but_not_eligible(self):
        parser = Stage1OutputParser()

        parsed = parser.parse(
            """
            step 1. The evidence is not enough.
            FINAL_ANSWER = unknown
            """
        )

        self.assertFalse(parsed["parse_completed"])
        self.assertTrue(parsed["repair_applied"])
        self.assertEqual(parsed["final_answer"], "unknown")
        self.assertIn("refusal_like_final_answer", parsed["validity_labels"])

    def test_repairs_markdown_answer_label_to_short_boolean(self):
        parser = ToolRequestParser()

        parsed = parser.parse(
            """
            The green cells are disconnected.

            **Answer**: No, Earl cannot complete the circuit without backtracking.
            """
        )

        self.assertEqual(parsed["type"], "final_answer")
        self.assertTrue(parsed["eligible_for_winner"])
        self.assertEqual(parsed["final_answer"], "no")

    def test_cleans_partial_markdown_boolean_answer(self):
        parser = ToolRequestParser()

        parsed = parser.parse('{"type": "final_answer", "final_answer": "** No.", "reasoning_steps": ["step 1. Check the grid."]}')

        self.assertEqual(parsed["type"], "final_answer")
        self.assertTrue(parsed["eligible_for_winner"])
        self.assertEqual(parsed["final_answer"], "no")

    def test_answer_type_union_is_normalized(self):
        parser = Stage1OutputParser()

        parsed = parser.parse(
            """
            {
              "type": "final_answer",
              "reasoning_steps": ["step 1. Use the evidence."],
              "final_answer": "42",
              "confidence": 0.8,
              "used_evidence_ids": [],
              "answer_type": "number | date | person | organization | location | title | list | short_text | boolean | unknown",
              "tool_request": null
            }
            """
        )

        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["structured_output"]["answer_type"], "unknown")
        self.assertTrue(parsed["eligible_for_winner"])

    def test_compact_long_list_remains_eligible(self):
        parser = Stage1OutputParser()
        answer = (
            "3/4,1/4,3/4,3/4,2/4,1/2,5/35,7/21,30/5,30/5,"
            "3/4,1/15,1/3,4/9,1/8,32/23,103/170"
        )

        parsed = parser.parse(
            "{"
            '"type":"final_answer",'
            '"reasoning_steps":["step 1. Read the requested sequence."],'
            f'"final_answer":"{answer}",'
            '"confidence":1.0,'
            '"used_evidence_ids":[], '
            '"answer_type":"list",'
            '"tool_request":null'
            "}"
        )

        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["eligible_for_winner"])
        self.assertEqual(parsed["final_answer"], answer)

    def test_repairs_broken_json_final_answer_field(self):
        parser = ToolRequestParser()

        parsed = parser.parse(
            """```json
            {"type": "final_answer", "reasoning_steps": ["The answer is in E1"], "final_answer": "Tokyo"
            ```"""
        )

        self.assertEqual(parsed["type"], "final_answer")
        self.assertEqual(parsed["final_answer"], "Tokyo")
        self.assertTrue(parsed["eligible_for_winner"])


if __name__ == "__main__":
    unittest.main()
