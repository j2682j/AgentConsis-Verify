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


def collection_row(
    document_id: str,
    *,
    score: float,
    text: str,
    url: str = "https://example.com/collection",
    record_type: str = "database_row",
) -> dict:
    return {
        "document_id": document_id,
        "title": f"Collection {document_id}",
        "text": text,
        "url": url,
        "retrieval_score": score,
        "record_type": record_type,
        "record_fields": {"parent_url": url},
    }


class BestEffortReferenceSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.long_text = (
            "This passage contains retrieved context that may help the agent reason "
            "about the question, but it has not passed the strict evidence contract."
        )

    def test_strict_evidence_no_longer_disables_the_fallback(self) -> None:
        """Withholding references whenever any grounded item existed cost tasks.

        The rule assumed a grounded item states the answer, which held while
        level1_final_13 produced one across 53 tasks. level1_final_14 produced
        five, three of which did not, and each removed all eight references
        from its task -- 046 fell from 9 of 9 runs correct to 1 of 9. The two
        now share the prompt, with the context budget giving evidence its
        allowance first, so this selector no longer decides.
        """
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

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_id, "D1")

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

    def test_collection_merge_filters_navigation_placeholders(self) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "question": "How many studio albums were released?",
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "junk-1",
                                score=0.99,
                                text="Wiktionary (0 entries) Content: edit",
                            ),
                            collection_row(
                                "junk-2",
                                score=0.98,
                                text="Special:WhatLinksHere /wiki/example Content link: edit",
                            ),
                            collection_row(
                                "valid-1",
                                score=0.80,
                                text="Date: 2001 Album: First studio album",
                            ),
                            collection_row(
                                "valid-2",
                                score=0.79,
                                text="Date: 2005 Album: Second studio album",
                            ),
                        ],
                    }
                ]
            },
        }

        references = selector.select(output)

        self.assertEqual(len(references), 1)
        self.assertNotIn("0 entries", references[0].text)
        self.assertNotIn("Special:", references[0].text)
        self.assertIn("First studio album", references[0].text)
        self.assertIn("Second studio album", references[0].text)

    def test_collection_with_fewer_than_two_effective_rows_is_dropped(self) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "junk",
                                score=0.99,
                                text="Wikipedia (0 entries) Content: edit",
                            ),
                            collection_row(
                                "valid",
                                score=0.80,
                                text="Date: 2001 Album: Only useful row",
                            ),
                        ],
                    }
                ]
            }
        }

        self.assertEqual(selector.select(output), [])

    def test_wikidata_schema_metadata_group_is_dropped(self) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "schema-1",
                                score=0.99,
                                text=(
                                    "Source: Example - Wikidata Content Link: "
                                    "https://www.wikidata.org/wiki/Property:P360"
                                ),
                            ),
                            collection_row(
                                "schema-2",
                                score=0.98,
                                text=(
                                    "Source: Example - Wikidata Language: English "
                                    "Label: Example Also known as: Example list"
                                ),
                            ),
                        ],
                    }
                ]
            }
        }

        self.assertEqual(selector.select(output), [])

    def test_genuine_singleton_structured_record_remains_available(self) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "single",
                                score=0.80,
                                text="Perigee minimum distance: 356400 km",
                            )
                        ],
                    }
                ]
            }
        }

        references = selector.select(output)

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].source_id, "single")
        self.assertIn("356400 km", references[0].text)

    def test_year_range_rows_are_prioritized_before_retrieval_score(self) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "question": "How many studio albums were released between 2000 and 2009?",
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "1966",
                                score=0.99,
                                text="Date: 1966 Album: Outside High Score",
                            ),
                            collection_row(
                                "2001",
                                score=0.70,
                                text="Date: 2001 Album: In Range Lower Score",
                            ),
                            collection_row(
                                "2009",
                                score=0.80,
                                text="Date: 2009 Album: In Range Higher Score",
                            ),
                            collection_row(
                                "2011",
                                score=0.90,
                                text="Date: 2011 Album: Outside Medium Score",
                            ),
                        ],
                    }
                ]
            },
        }

        references = selector.select(output)

        self.assertEqual(len(references), 1)
        text = references[0].text
        self.assertLess(text.index("2009"), text.index("2001"))
        self.assertLess(text.index("2001"), text.index("1966"))
        self.assertLess(text.index("1966"), text.index("2011"))

    def test_sibling_rows_with_different_record_types_share_parent_collection(
        self,
    ) -> None:
        selector = BestEffortReferenceSelector(min_chars=20)
        output = {
            "question": "How many albums were released between 2000 and 2009?",
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            collection_row(
                                "2002",
                                score=0.80,
                                text="Date: 2002 Album: First",
                                record_type="article",
                            ),
                            collection_row(
                                "2005",
                                score=0.79,
                                text="Date: 2005 Album: Second",
                                record_type="database_row",
                            ),
                            collection_row(
                                "1966",
                                score=0.99,
                                text="Date: 1966 Album: Outside",
                                record_type="database_row",
                            ),
                        ],
                    }
                ]
            },
        }

        references = selector.select(output)

        self.assertEqual(len(references), 1)
        self.assertIn("2002", references[0].text)
        self.assertIn("2005", references[0].text)
        self.assertLess(references[0].text.index("2005"), references[0].text.index("1966"))

    def test_default_reference_budget_is_expanded(self) -> None:
        selector = BestEffortReferenceSelector()

        self.assertEqual(selector.max_items, 5)
        self.assertEqual(selector.max_total_chars, 4000)


if __name__ == "__main__":
    unittest.main()
