from __future__ import annotations

import unittest

from core.evidence_runner import EvidenceRunner


class EvidenceRunnerSelectionTests(unittest.TestCase):
    def test_web_retrieval_evidence_exports_only_direct_contracts(self):
        runner = EvidenceRunner(question="test question")
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            {
                                "document_id": "D1",
                                "title": "Continue evidence",
                                "text": "Primary continue evidence.",
                                "url": "https://example.com/continue",
                                "retrieval_score": 0.91,
                                "sequence_tag": "<CONTINUE>",
                                "useful_tokens": ["bridge"],
                                "bridge_contracts": [
                                    {
                                        "goal_id": "G1",
                                        "bridge_span": "bridge",
                                        "context": "Primary continue evidence.",
                                        "document_id": "D1",
                                        "next_goal_id": "G2",
                                    }
                                ],
                            },
                            {
                                "document_id": "D2",
                                "title": "High terminate evidence",
                                "text": "Secondary terminate evidence with candidate answer.",
                                "url": "https://example.com/terminate-high",
                                "retrieval_score": 0.88,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": ["candidate"],
                                "direct_contracts": [
                                    {
                                        "goal_id": "G2",
                                        "answer_span": "candidate",
                                        "context": "Secondary terminate evidence with candidate answer.",
                                        "document_id": "D2",
                                        "source_title": "High terminate evidence",
                                        "url": "https://example.com/terminate-high",
                                        "answer_requirement": "test question",
                                    }
                                ],
                            },
                            {
                                "document_id": "D3",
                                "title": "Low terminate evidence",
                                "text": "Low score terminate evidence with a generic token.",
                                "url": "https://example.com/terminate-low",
                                "retrieval_score": 0.70,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": ["generic"],
                            },
                            {
                                "document_id": "D5",
                                "title": "Weak terminate evidence",
                                "text": "High score terminate evidence with weak one-word token.",
                                "url": "https://example.com/terminate-weak",
                                "retrieval_score": 0.89,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": ["snow"],
                            },
                            {
                                "document_id": "D4",
                                "title": "Fallback evidence",
                                "text": "Plain fallback evidence.",
                                "url": "https://example.com/fallback",
                                "retrieval_score": 0.87,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": [],
                            },
                        ],
                    }
                ]
            }
        }

        items = runner._web_retrieval_evidence_items(output, max_items=8)

        self.assertEqual([item["source_id"] for item in items], ["D2"])
        self.assertEqual(
            [item["selection_reason"] for item in items],
            ["direct_evidence_contract"],
        )
        self.assertEqual([item["evidence_id"] for item in items], ["E1"])
        self.assertNotIn("D1", {item["source_id"] for item in items})
        self.assertNotIn("D3", {item["source_id"] for item in items})
        self.assertNotIn("D4", {item["source_id"] for item in items})

    def test_web_retrieval_evidence_does_not_fallback_without_direct_contract(self):
        runner = EvidenceRunner(question="test question")
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            {
                                "document_id": "D1",
                                "title": "Low terminate evidence",
                                "text": "Low score terminate evidence with token.",
                                "url": "https://example.com/low",
                                "retrieval_score": 0.73,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": ["token"],
                            },
                            {
                                "document_id": "D2",
                                "title": "Plain fallback evidence",
                                "text": "Plain fallback evidence.",
                                "url": "https://example.com/plain",
                                "retrieval_score": 0.72,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": [],
                            },
                        ],
                    }
                ]
            }
        }

        items = runner._web_retrieval_evidence_items(output, max_items=2)

        self.assertEqual(items, [])

    def test_web_retrieval_adds_unverified_references_when_strict_layer_is_empty(self):
        runner = EvidenceRunner(question="test question")
        output = {
            "question": "test question",
            "generated_queries": ["test query"],
            "salient_spans": ["test"],
            "web_searches": [
                {
                    "query": "test query",
                    "backend": "searxng",
                    "result_count": 1,
                }
            ],
            "corpus_path": "corpus.jsonl",
            "embedding_path": "embeddings",
            "corpus_record_count": 1,
            "retrieval": {
                "stop_reason": "goal_incomplete_no_viable_recovery",
                "rounds": [
                    {
                        "round_index": 1,
                        "documents": [
                            {
                                "document_id": "D1",
                                "title": "Retrieved page",
                                "text": (
                                    "This retrieved passage contains potentially useful context "
                                    "but did not produce a grounded direct evidence contract."
                                ),
                                "url": "https://example.com/reference",
                                "retrieval_score": 0.87,
                            }
                        ],
                    }
                ],
                "searched_queries": ["test query"],
            },
            "diagnostics": {"filtered_source_count": 1},
        }

        references = runner._web_retrieval_unverified_references(
            output,
            evidence_items=[],
        )
        raw_result = runner._web_retrieval_raw_result(
            output_dict=output,
            evidence_items=[],
            unverified_references=references,
        )

        self.assertEqual(raw_result["evidence_items"], [])
        self.assertEqual(len(raw_result["unverified_references"]), 1)
        self.assertFalse(raw_result["unverified_references"][0]["verified"])
        self.assertIn("Unverified References:", raw_result["summary"])
        self.assertIn("Source Title: Retrieved page", raw_result["summary"])
        self.assertNotIn("https://example.com/reference", raw_result["summary"])
        self.assertTrue(
            raw_result["diagnostics"]["best_effort_evidence"]["triggered"]
        )
        self.assertIn(
            "evidence_conversion_empty",
            raw_result["diagnostics"]["pipeline_failure_stage"],
        )

    def test_web_retrieval_raw_result_exports_blocked_source_details(self):
        runner = EvidenceRunner(question="test question")
        output = {
            "question": "test question",
            "generated_queries": ["test query"],
            "salient_spans": ["test"],
            "web_searches": [],
            "corpus_path": "corpus.jsonl",
            "embedding_path": "embeddings",
            "corpus_record_count": 0,
            "retrieval": None,
            "diagnostics": {
                "source_count": 1,
                "filtered_source_count": 0,
                "blocked_source_count": 1,
            },
            "blocked_sources": [
                {
                    "source_id": "S1",
                    "query_id": "Q1",
                    "rank": 1,
                    "title": "Question echo",
                    "url": "https://example.com/echo",
                    "domain": "example.com",
                    "snippet": "test question",
                    "raw_content": "test question raw content",
                    "block_reason": "question_semantic_echo",
                    "filter_reasons": [
                        "semantic_echo=0.970",
                        "lexical_overlap=1.000",
                        "new_information_ratio=0.000",
                    ],
                }
            ],
        }

        raw_result = runner._web_retrieval_raw_result(
            output_dict=output,
            evidence_items=[],
        )

        self.assertEqual(len(raw_result["blocked_sources"]), 1)
        blocked = raw_result["blocked_sources"][0]
        self.assertEqual(blocked["source_id"], "S1")
        self.assertEqual(blocked["block_reason"], "question_semantic_echo")
        self.assertIn("semantic_echo=0.970", blocked["filter_reasons"])
        self.assertIn("raw_content_preview", blocked)

    def test_web_retrieval_raw_result_exports_answer_candidates(self):
        runner = EvidenceRunner(question="test question")
        output = {
            "question": "test question",
            "generated_queries": [],
            "salient_spans": [],
            "web_searches": [],
            "corpus_path": "",
            "embedding_path": "",
            "corpus_record_count": 0,
            "retrieval": None,
            "diagnostics": {},
        }
        evidence_items = [
            {
                "evidence_id": "E1",
                "source_id": "D1",
                "title": "Source",
                "text": "The answer is 42.",
            }
        ]
        answer_candidates = [
            {
                "text": "42",
                "evidence_id": "E1",
                "answer_type": "number",
                "score": 0.91,
            }
        ]

        raw_result = runner._web_retrieval_raw_result(
            output_dict=output,
            evidence_items=evidence_items,
            answer_candidates=answer_candidates,
        )

        self.assertEqual(raw_result["answer_candidates"], answer_candidates)
        self.assertIn("Candidate Answers:", raw_result["summary"])
        self.assertIn("1. 42", raw_result["summary"])
        self.assertEqual(
            raw_result["diagnostics"]["final_counts"]["answer_candidate_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
