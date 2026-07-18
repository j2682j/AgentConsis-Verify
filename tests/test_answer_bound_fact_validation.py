from __future__ import annotations

import json
import unittest

from tools.evidence.fact_extraction import (
    AnswerBoundFactValidator,
    EvidenceFact,
    SearchContractFactAdapter,
    SemanticFactExtractor,
)
from tools.search_result_builder.evidence import EvidenceRoleContractBuilder


def grounded_answer_fact(value: str, *, context: str) -> EvidenceFact:
    return EvidenceFact(
        fact_id="F1",
        subject="La Voz de la Zafra",
        relation="released studio albums",
        object=value,
        role="ANSWER_SUPPORT",
        evidence_spans=[context],
        context=context,
        source_id="D1",
        grounding_status="grounded",
    )


class AnswerBoundFactValidationTests(unittest.TestCase):
    def test_count_rejects_year_but_accepts_aggregate_value(self) -> None:
        validator = AnswerBoundFactValidator()
        context = "La Voz de la Zafra released 3 studio albums by 1962."
        year = validator.bind(
            grounded_answer_fact("1962", context=context),
            question="How many studio albums were released?",
        )
        count = validator.bind(
            grounded_answer_fact("3", context=context),
            question="How many studio albums were released?",
        )
        self.assertNotEqual(year.role, "ANSWER_SUPPORT")
        self.assertEqual(year.qualifiers["binding_reason"], "count_rejects_standalone_year")
        self.assertTrue(validator.is_direct(count))

    def test_measurement_requires_number_and_unit(self) -> None:
        validator = AnswerBoundFactValidator()
        context = "The container volume is 0.1777 m3."
        missing_unit = validator.bind(
            grounded_answer_fact("0.1777", context=context),
            question="What is the volume in m3?",
        )
        with_unit = validator.bind(
            grounded_answer_fact("0.1777 m3", context=context),
            question="What is the volume in m3?",
        )
        self.assertNotEqual(missing_unit.role, "ANSWER_SUPPORT")
        self.assertTrue(validator.is_direct(with_unit))

    def test_contract_rejects_span_without_answer_bound_fact(self) -> None:
        contracts = EvidenceRoleContractBuilder().build(
            question="How many studio albums were released?",
            answer_requirement="how many studio albums",
            answer_target="studio album count",
            relation_plan=None,
            document_id="D1",
            source_title="Discography",
            url="https://example.test",
            text="Look at the discography.",
            span_assignments=[
                {
                    "accepted": True,
                    "role": "ANSWER_SUPPORT",
                    "goal_id": "",
                    "original_text": "Look",
                    "finalized_text": "Look",
                    "semantic_facts": [],
                }
            ],
        )
        self.assertEqual(contracts.direct, [])
        self.assertEqual(
            contracts.unsupported[0].reason,
            "missing_grounded_answer_fact",
        )

    def test_search_adapter_does_not_synthesize_missing_sro(self) -> None:
        item = {
            "tool_name": "search",
            "raw_result": {
                "evidence_items": [
                    {
                        "evidence_id": "E1",
                        "source_id": "D1",
                        "title": "Discography",
                        "text": "The artist released 3 studio albums.",
                        "direct_contracts": [
                            {
                                "answer_span": "3",
                                "answer_requirement": "how many studio albums",
                            }
                        ],
                    }
                ]
            },
        }
        self.assertEqual(
            SearchContractFactAdapter().convert(
                item,
                question="How many studio albums were released?",
                source_scope="evidence_prepare",
            ),
            [],
        )

    def test_malformed_outer_json_recovers_complete_units(self) -> None:
        unit = {
            "unit_id": "U1",
            "facts": [
                {
                    "subject": "A",
                    "relation": "is",
                    "object": "B",
                    "qualifiers": {},
                    "polarity": "positive",
                    "role": "BRIDGE",
                    "goal_id": "",
                    "evidence_spans": ["A is B"],
                }
            ],
        }
        malformed = '{"units":[' + json.dumps(unit) + ","
        parsed = SemanticFactExtractor._parse_response(malformed)
        self.assertEqual(parsed["units"], [unit])


if __name__ == "__main__":
    unittest.main()
