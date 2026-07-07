from __future__ import annotations

import unittest

from tools.validation import ToolResultValidator


class ToolResultValidatorTests(unittest.TestCase):
    def test_valid_tool_output_passes(self):
        result = ToolResultValidator().validate(
            tool_name="search",
            output_text="Evidence:\n[E1] useful source text",
            metadata={"ok": True},
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "ok")

    def test_empty_output_is_invalid(self):
        result = ToolResultValidator().validate(
            tool_name="search",
            output_text="",
            metadata={"ok": True},
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "empty_output")

    def test_missing_inputs_are_invalid_but_preserved(self):
        result = ToolResultValidator().validate(
            tool_name="deterministic_handler_router",
            output_text="",
            metadata={
                "ok": False,
                "missing_inputs": ["table_rows"],
                "next_action_hint": "Read attachment.",
            },
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "missing_input")
        self.assertEqual(result.missing_inputs, ["table_rows"])
        self.assertEqual(result.next_action_hint, "Read attachment.")

    def test_tool_call_like_output_is_invalid(self):
        result = ToolResultValidator().validate(
            tool_name="attachment_reader",
            output_text='{"tool_name": "search", "tool_args": {"query": "x"}}',
            metadata={"ok": True},
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid_format")


if __name__ == "__main__":
    unittest.main()
