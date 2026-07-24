"""Plan-13 evidence trust contract: relaxed passages carry no support power."""

from __future__ import annotations

import unittest

from tools.search_result_builder.config import (
    EvidenceTier,
    is_support_eligible_payload,
)
from tools.search_result_builder.evidence.evidence_contract import (
    EvidenceSelectionContract,
)
from tools.search_result_builder.evidence.evidence_converter import EvidenceConverter
from score.evidence_support_checker import EvidenceSupportChecker


QUESTION = "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"


def relaxed_only_retrieval() -> dict:
    """A retrieval trace with no direct contracts — strict conversion yields nothing."""
    return {
        "retrieval": {
            "rounds": [
                {
                    "round_index": 1,
                    "query": QUESTION,
                    "documents": [
                        {
                            "document_id": "D1",
                            "title": "Word of the Day archive",
                            "url": "https://example.test/archive",
                            "text": (
                                "The Merriam-Webster Word of the Day for June 27 2022 "
                                "discusses jingoism, nationalism and chauvinism across "
                                "several editorial columns and archive citations."
                            ),
                            "label": "useful",
                            "retrieval_score": 0.8,
                        }
                    ],
                }
            ]
        }
    }


class EvidenceTrustContractTests(unittest.TestCase):
    def test_support_eligibility_defaults(self) -> None:
        self.assertTrue(is_support_eligible_payload({"text": "x"}))
        self.assertFalse(is_support_eligible_payload({"text": "x", "relaxed": True}))
        self.assertFalse(
            is_support_eligible_payload(
                {"text": "x", "evidence_tier": EvidenceTier.RELAXED_CONTEXT.value}
            )
        )
        self.assertTrue(
            is_support_eligible_payload(
                {"text": "x", "evidence_tier": EvidenceTier.ANSWER_GROUNDED.value}
            )
        )
        # An explicit flag always wins over the tier.
        self.assertFalse(
            is_support_eligible_payload(
                {
                    "text": "x",
                    "evidence_tier": EvidenceTier.ANSWER_GROUNDED.value,
                    "support_eligible": False,
                }
            )
        )
        self.assertFalse(is_support_eligible_payload("not a dict"))

    def test_relaxed_passages_never_become_evidence_items(self) -> None:
        converter = EvidenceConverter(max_items=8, max_chars=400)
        items = converter.convert_web_retrieval_output(
            relaxed_only_retrieval(),
            contract=EvidenceSelectionContract.from_parts(question=QUESTION),
        )

        self.assertEqual(items, [])
        self.assertTrue(converter.last_relaxed_references)
        for reference in converter.last_relaxed_references:
            self.assertFalse(reference["support_eligible"])
            self.assertFalse(reference["verification_ready"])
            self.assertEqual(
                reference["evidence_tier"], EvidenceTier.RELAXED_CONTEXT.value
            )
            self.assertTrue(reference["reference_id"].startswith("R"))
            self.assertEqual(reference["evidence_id"], "")

    def test_support_checker_ignores_relaxed_evidence_items(self) -> None:
        """A relaxed passage must not produce any support record.

        This is the regression that made wrong answers win: relaxed passages
        exposed matched_terms as intermediate values, so an agent answer that
        merely echoed question vocabulary was labelled bridge-supported.
        """
        checker = EvidenceSupportChecker()
        relaxed_item = {
            "tool_name": "search",
            "raw_result": {
                "evidence_items": [
                    {
                        "evidence_id": "E1",
                        "text": "Jingoism was the Word of the Day on June 27 2022.",
                        "title": "archive",
                        "matched_terms": ["jingoism", "merriam-webster"],
                        "compatible_spans": ["jingoism"],
                        "evidence_tier": EvidenceTier.RELAXED_CONTEXT.value,
                        "support_eligible": False,
                        "relaxed": True,
                    }
                ]
            },
        }
        self.assertEqual(
            checker._search_evidence_records(relaxed_item, source_scope="evidence_prepare"),
            [],
        )

        grounded_item = {
            "tool_name": "search",
            "raw_result": {
                "evidence_items": [
                    {
                        "evidence_id": "E1",
                        "text": "Jingoism was the Word of the Day on June 27 2022.",
                        "title": "archive",
                        "matched_terms": ["jingoism"],
                        "evidence_tier": EvidenceTier.ANSWER_GROUNDED.value,
                        "support_eligible": True,
                    }
                ]
            },
        }
        self.assertTrue(
            checker._search_evidence_records(
                grounded_item, source_scope="evidence_prepare"
            )
        )


if __name__ == "__main__":
    unittest.main()
