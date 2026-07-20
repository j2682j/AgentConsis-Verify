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
            output_type="final_answer",
            semantic_role="arithmetic_result",
            supporting_inputs=["9 - 4"],
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

    def test_generic_list_handler_cannot_finalize_truth_assignment(self):
        result = HandlerResult(
            handler_name="list_operations",
            status="ok",
            answer="2",
            evidence_text="Extracted the second item.",
            input_summary={"items": ["1", "2", "3"]},
            structured_result={
                "handler_role": "list_operation",
                "operation": "select_nth",
                "output_contract": {"required_outputs": ["answer"]},
            },
            output_type="final_answer",
            semantic_role="list_item",
            supporting_inputs=["1", "2", "3"],
            operation="select_nth",
            derivation_type="deterministic_computation",
            derivation_trace=[{"operation": "select_nth", "result": "2"}],
        )

        trust = HandlerTrustGate().validate(
            result,
            question=(
                "Exactly one person is lying. Determine who is telling the truth."
            ),
        )

        self.assertFalse(trust.trusted)
        self.assertIn("answer_role_binding_failed", trust.reasons)


if __name__ == "__main__":
    unittest.main()
