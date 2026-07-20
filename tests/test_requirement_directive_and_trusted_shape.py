from __future__ import annotations

"""Regression tests for two level1-batch failure patterns.

1. AnswerRequirementGate: explicit output-format directives in the task text
   must outrank a misclassified answer_role (boolean from "Could you
   please...", person from "Girls Who Code", number over an IOC-code ask).
2. EvidenceSupportChecker: a trusted deterministic handler final of a
   different answer shape (for example the numeric list "5, 7" extracted from
   a "5x7 block" phrase) must not contradict a text-shaped candidate answer.
"""

import unittest

from core.config import AgentReasoningSummary, EachAgentReply
from score.answer_requirement_gate import AnswerRequirementGate
from score.evidence_support_checker import EvidenceSupportChecker


def summary(agent_id: str, answer: str, reasoning: str) -> AgentReasoningSummary:
    run = EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=1,
        raw_reply="",
        reasoning=reasoning,
        final_answer=answer,
        tool_context="",
        parse_completed=True,
        schema_valid=True,
        eligible_for_winner=True,
    )
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=[run],
        compressed_answer=answer,
        compressed_reasoning=reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )


class ExplicitFormatDirectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = AnswerRequirementGate()

    def test_ingredient_list_directive_overrides_boolean_role(self) -> None:
        requirement = (
            "Could you please listen to the recipe and list all of the "
            "ingredients that my friend described? Please format your "
            "response as a comma separated list of ingredients."
        )
        result = self.gate.evaluate(
            answer="cornstarch, lemon juice, ripe strawberries, sugar",
            answer_type="list",
            answer_requirement=requirement,
            answer_role="boolean",
        )
        self.assertEqual(result.expected_type, "list")
        self.assertNotEqual(result.outcome, "incompatible")

    def test_ioc_code_directive_overrides_number_role(self) -> None:
        requirement = (
            "What country had the least number of athletes at the 1928 Summer "
            "Olympics? Give the IOC country code as your answer."
        )
        result = self.gate.evaluate(
            answer="CUB",
            answer_type="short_text",
            answer_requirement=requirement,
            answer_role="number",
        )
        self.assertEqual(result.expected_type, "text")
        self.assertNotEqual(result.outcome, "incompatible")

    def test_how_long_in_years_overrides_person_role(self) -> None:
        requirement = (
            "According to Girls Who Code, how long did it take in years for "
            "the percentage of computer scientists that were women to change "
            "by 13% from a starting point of 37%?"
        )
        result = self.gate.evaluate(
            answer="22",
            answer_type="number",
            answer_requirement=requirement,
            answer_role="person",
        )
        self.assertEqual(result.expected_type, "number")
        self.assertNotEqual(result.outcome, "incompatible")

    def test_role_still_used_without_explicit_directive(self) -> None:
        result = self.gate.evaluate(
            answer="Paris",
            answer_type="short_text",
            answer_requirement="Which city hosted the summit?",
            answer_role="place",
        )
        self.assertEqual(result.expected_type, "place")

    def test_measurement_rejects_conflicting_unit_family(self) -> None:
        result = self.gate.evaluate(
            answer="716 kg",
            answer_type="short_text",
            answer_requirement="What was the volume in m^3 of the fish bag?",
        )
        self.assertEqual(result.outcome, "incompatible")
        self.assertEqual(
            result.reason,
            "candidate_unit_conflicts_with_answer_requirement",
        )

    def test_measurement_allows_unit_inherited_from_question(self) -> None:
        result = self.gate.evaluate(
            answer="0.1777",
            answer_type="number",
            answer_requirement="What was the volume in m^3 of the fish bag?",
        )
        self.assertNotEqual(result.outcome, "incompatible")


class TrustedFinalShapeGuardTests(unittest.TestCase):
    def _evidence_with_trusted_final(self, value: str) -> dict:
        return {
            "tool_usage": [
                {
                    "tool_name": "deterministic_handler_router",
                    "ok": True,
                    "evidence_valid": True,
                    "trusted": True,
                    "output_type": "final_answer",
                    "output_text": value,
                    "handler_trust": {"trusted": True},
                    "raw_result": {
                        "output_type": "final_answer",
                        "final_answer": value,
                    },
                }
            ]
        }

    def test_numeric_list_final_does_not_contradict_text_answer(self) -> None:
        answer = "THESEAGULLGLIDEDPEACEFULLYTOMYCHAIR"
        target = summary("a1", answer, f"step 1. The sentence is {answer}.")

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, f"The sentence is {answer}.")],
            evidence=self._evidence_with_trusted_final("5, 7"),
        )

        self.assertNotEqual(support.status, "contradicted")

    def test_numeric_final_still_contradicts_conflicting_numeric_answer(self) -> None:
        target = summary("a1", "12", "step 1. The count is 12.")

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, "The count is 12.")],
            evidence=self._evidence_with_trusted_final("7"),
        )

        self.assertEqual(support.status, "contradicted")


if __name__ == "__main__":
    unittest.main()
