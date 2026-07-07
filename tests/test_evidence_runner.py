from __future__ import annotations

import unittest

from core.evidence_runner import EvidenceRunner


class EvidenceRunnerSelectionTests(unittest.TestCase):
    def test_web_retrieval_evidence_prefers_continue_finish_then_high_score_terminate(self):
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
                            },
                            {
                                "document_id": "D2",
                                "title": "High terminate evidence",
                                "text": "Secondary terminate evidence with useful token.",
                                "url": "https://example.com/terminate-high",
                                "retrieval_score": 0.88,
                                "sequence_tag": "<TERMINATE>",
                                "useful_tokens": ["candidate"],
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

        self.assertEqual([item["source_id"] for item in items], ["D1", "D2"])
        self.assertEqual(
            [item["selection_reason"] for item in items],
            ["primary_labeler_sequence", "secondary_terminate_with_terms"],
        )
        self.assertEqual([item["evidence_id"] for item in items], ["E1", "E2"])
        self.assertNotIn("D3", {item["source_id"] for item in items})
        self.assertNotIn("D4", {item["source_id"] for item in items})

    def test_web_retrieval_evidence_uses_fallback_only_when_no_selected_items(self):
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

        self.assertEqual([item["source_id"] for item in items], ["D1", "D2"])
        self.assertEqual(
            [item["selection_reason"] for item in items],
            ["fallback_retrieval_order", "fallback_retrieval_order"],
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
