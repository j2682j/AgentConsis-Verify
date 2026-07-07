from __future__ import annotations

import unittest

from tools.search_result_builder.evidence import EvidenceAnswerExtractor


class AnswerCandidateExtractorTests(unittest.TestCase):
    def test_extracts_and_ranks_candidates_from_evidence(self):
        def fake_qa(payload, **kwargs):
            context = payload["context"]
            if "0.1777 m3" in context:
                return [
                    {"answer": "0.1777 m3", "score": 0.91},
                    {"answer": "water", "score": 0.24},
                ]
            return {"answer": "unknown", "score": 0.99}

        def fake_similarity(left, right):
            combined = f"{left} {right}".lower()
            if "tank" in combined and "0.1777" in combined:
                return 0.84
            if "number" in combined or "measurement" in combined:
                return 0.88
            return 0.10

        extractor = EvidenceAnswerExtractor(
            qa_pipeline=fake_qa,
            similarity_fn=fake_similarity,
            max_candidates=3,
        )
        candidates = extractor.extract(
            question="What is the water volume in the tank?",
            evidence_items=[
                {
                    "evidence_id": "E1",
                    "source_id": "D1",
                    "title": "Tank volume",
                    "text": "The calculated tank water volume is 0.1777 m3.",
                }
            ],
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["text"], "0.1777 m3")
        self.assertEqual(candidates[0]["evidence_id"], "E1")
        self.assertIn("score", candidates[0])
        self.assertIn("qa_span_score", candidates[0])
        self.assertIn("question_context_relevance", candidates[0])
        self.assertIn("answer_type_compatibility", candidates[0])

    def test_deduplicates_candidates_by_normalized_answer(self):
        def fake_qa(payload, **kwargs):
            return [
                {"answer": "Paris", "score": 0.40},
                {"answer": "paris", "score": 0.90},
            ]

        extractor = EvidenceAnswerExtractor(
            qa_pipeline=fake_qa,
            similarity_fn=lambda left, right: 0.50,
        )
        candidates = extractor.extract(
            question="What city is the answer?",
            evidence_items=[
                {
                    "evidence_id": "E1",
                    "source_id": "D1",
                    "title": "City",
                    "text": "The answer is Paris.",
                }
            ],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["text"], "paris")
        self.assertEqual(candidates[0]["qa_span_score"], 0.90)


if __name__ == "__main__":
    unittest.main()
