from __future__ import annotations

"""Regression tests for the level1_40 losses caused by the Debug-stage2 layer.

1. Router derivation-metadata fallback: a handler that computes a final
   answer without emitting a step payload (logic_equivalence style) must
   still receive a minimal derivation_trace, so the trust gate's
   missing_derivation_trace rule cannot reject a correctly computed answer.
2. Candidate verification search must be OFF by default: an A/B run showed
   it can promote wrong candidates whose text co-occurs with question terms.
"""

import unittest

from core.network import Network
from core.config import AgentConfig
from tools.deterministic_handlers import HandlerResult, HandlerTrustGate
from tools.deterministic_handlers.router import DeterministicHandlerRouter


class DerivationTraceFallbackTests(unittest.TestCase):
    def _traceless_final(self) -> HandlerResult:
        return HandlerResult(
            handler_name="logic_equivalence",
            status="ok",
            answer="(¬A → B) ↔ (A ∨ ¬B)",
            evidence_text=(
                "Deterministic handler evidence:\n"
                "Answer: (¬A → B) ↔ (A ∨ ¬B)"
            ),
            confidence=0.9,
            input_summary={"statements": ["..."]},
            structured_result={
                "handler_role": "logic_equivalence",
                "operation": "logic_equivalence_check",
                "output_contract": {"required_outputs": ["answer"]},
            },
            output_type="final_answer",
            semantic_role="logic_result",
            supporting_inputs=["statement list"],
            operation="logic_equivalence_check",
        )

    def test_router_fallback_builds_minimal_trace_for_final_answers(self) -> None:
        result = self._traceless_final()

        DeterministicHandlerRouter._apply_derivation_metadata(result)

        self.assertTrue(result.derivation_trace)
        self.assertEqual(result.derivation_trace[0]["trace"], "direct_computation")
        self.assertEqual(
            result.derivation_type,
            "deterministic_computation",
        )

    def test_normalized_traceless_final_passes_trust_gate(self) -> None:
        result = self._traceless_final()
        DeterministicHandlerRouter._apply_derivation_metadata(result)

        trust = HandlerTrustGate().validate(
            result,
            question=(
                "Which of the following statements is logically equivalent?"
            ),
            handler_plan={
                "handler_name": "logic_equivalence",
                "operation": "logic_equivalence_check",
            },
        )

        self.assertNotIn("missing_derivation_trace", trust.reasons)
        self.assertNotIn("missing_derivation_type", trust.reasons)
        self.assertTrue(trust.trusted)

    def test_intermediate_outputs_do_not_gain_fabricated_traces(self) -> None:
        result = HandlerResult(
            handler_name="list_operations",
            status="ok",
            answer="1, 2, 3",
            evidence_text="Extracted list items.",
            structured_result={},
            output_type="intermediate_value",
        )

        DeterministicHandlerRouter._apply_derivation_metadata(result)

        self.assertEqual(result.derivation_trace, [])
        self.assertEqual(result.derivation_type, "intermediate_extraction")


class CandidateVerificationDefaultTests(unittest.TestCase):
    def test_candidate_verification_search_is_off_by_default(self) -> None:
        network = Network(
            "Which city?",
            [AgentConfig(agent_id="a1", model_name="test-model")],
        )

        self.assertFalse(network.enable_candidate_verification_search)


if __name__ == "__main__":
    unittest.main()
