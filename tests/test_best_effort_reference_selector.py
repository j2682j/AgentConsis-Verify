from __future__ import annotations

import unittest

from tools.evidence.fact_extraction import TaskFactCollector, TaskFactStore
from tools.search_result_builder.evidence import BestEffortReferenceSelector


def document(
    document_id: str,
    *,
    score: float,
    url: str,
    text: str,
    duplicate: bool = False,
) -> dict:
    return {
        "document_id": document_id,
        "title": f"Title {document_id}",
        "text": text,
        "url": url,
        "retrieval_score": score,
        "record_type": "passage",
        "duplicate": duplicate,
    }


class BestEffortReferenceSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.long_text = (
            "This passage contains retrieved context that may help the agent reason "
            "about the question, but it has not passed the strict evidence contract."
        )

    def test_strict_evidence_disables_fallback(self) -> None:
        selector = BestEffortReferenceSelector()
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            document(
                                "D1",
                                score=0.9,
                                url="https://example.com/page",
                                text=self.long_text,
                            )
                        ],
                    }
                ]
            }
        }

        result = selector.select(
            output,
            strict_evidence_items=[{"evidence_id": "E1"}],
        )

        self.assertEqual(result, [])

    def test_selects_ranked_deduplicated_references(self) -> None:
        selector = BestEffortReferenceSelector(max_items=3, max_items_per_domain=1)
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            document(
                                "D1",
                                score=0.81,
                                url="https://one.example/page",
                                text=self.long_text + " First.",
                            ),
                            document(
                                "D2",
                                score=0.95,
                                url="https://two.example/top?tracking=1",
                                text=self.long_text + " Highest.",
                            ),
                            document(
                                "D3",
                                score=0.90,
                                url="https://two.example/other",
                                text=self.long_text + " Same domain.",
                            ),
                            document(
                                "D4",
                                score=0.99,
                                url="https://duplicate.example/page",
                                text=self.long_text + " Duplicate flag.",
                                duplicate=True,
                            ),
                        ],
                    }
                ]
            }
        }

        result = selector.select(output, strict_evidence_items=[])

        self.assertEqual([item.source_id for item in result], ["D2", "D1"])
        self.assertEqual([item.reference_id for item in result], ["R1", "R2"])
        self.assertTrue(all(not item.to_dict()["verified"] for item in result))

    def test_unverified_references_do_not_enter_fact_store(self) -> None:
        store = TaskFactStore()
        collector = TaskFactCollector()
        collector.collect_item(
            store,
            {
                "tool_name": "search",
                "raw_result": {
                    "evidence_items": [],
                    "unverified_references": [
                        {
                            "reference_id": "R1",
                            "title": "Reference",
                            "text": self.long_text,
                            "verified": False,
                        }
                    ],
                },
            },
            question="What is the answer?",
            source_scope="evidence_prepare",
        )

        self.assertEqual(store.all(), [])


if __name__ == "__main__":
    unittest.main()
