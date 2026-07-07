from __future__ import annotations

import unittest

from tools.deterministic_handlers import HandlerResult, HandlerTrustGate


class HandlerTrustGateTests(unittest.TestCase):
    def test_trusted_handler_result_passes(self):
        result = HandlerResult(
            handler_name="simple_math",
            status="ok",
            answer="5",
            evidence_text="Deterministic handler evidence:\nAnswer: 5",
            confidence=0.9,
            input_summary={"expression": "9 - 4"},
            structured_result={"output_contract": {"required_outputs": ["answer"]}},
        )

        trust = HandlerTrustGate().validate(
            result,
            question="Compute 9 - 4.",
            handler_plan={"handler_name": "simple_math"},
        )

        self.assertTrue(trust.trusted)
        self.assertEqual(trust.status, "trusted")
        self.assertEqual(trust.answer, "5")

    def test_planned_handler_mismatch_is_untrusted(self):
        result = HandlerResult(
            handler_name="simple_math",
            status="ok",
            answer="5",
            evidence_text="Deterministic handler evidence:\nAnswer: 5",
        )

        trust = HandlerTrustGate().validate(
            result,
            question="Compute 9 - 4.",
            handler_plan={"handler_name": "table_exact_operations"},
        )

        self.assertFalse(trust.trusted)
        self.assertEqual(trust.status, "handler_mismatch")

    def test_missing_inputs_are_untrusted(self):
        result = HandlerResult.missing(
            handler_name="table_exact_operations",
            missing_inputs=["table_rows"],
            next_action_hint="Read attachment.",
        )

        trust = HandlerTrustGate().validate(
            result,
            question="How many rows are in the table?",
            handler_plan={"handler_name": "table_exact_operations"},
        )

        self.assertFalse(trust.trusted)
        self.assertEqual(trust.status, "missing_input")
        self.assertEqual(trust.missing_inputs, ["table_rows"])

    def test_refusal_answer_is_untrusted(self):
        result = HandlerResult(
            handler_name="text_extraction",
            status="ok",
            answer="unknown",
            evidence_text="Deterministic handler evidence:\nAnswer: unknown",
        )

        trust = HandlerTrustGate().validate(
            result,
            question="Who is named?",
            handler_plan={"handler_name": "text_extraction"},
        )

        self.assertFalse(trust.trusted)
        self.assertEqual(trust.status, "invalid_candidate")


if __name__ == "__main__":
    unittest.main()
