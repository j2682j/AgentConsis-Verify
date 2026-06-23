from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.search_result_builder.config import EvidenceItem
from tools.search_result_builder.next_hop_query import RetrievalController


class FakeSemanticScorer:
    def __init__(self, similarities):
        self.similarities = similarities

    def semantic_similarities(self, reference, texts):
        del reference
        return self.similarities[: len(texts)]


class FakeNLP:
    def __init__(self, entities_by_text):
        self.entities_by_text = entities_by_text

    def __call__(self, text):
        entities = [
            SimpleNamespace(text=value, label_=label)
            for value, label in self.entities_by_text.get(text, [])
        ]
        return SimpleNamespace(ents=entities)


def evidence(text: str, title: str = "") -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E1",
        source_id="S1",
        query_id="Q1",
        text=text,
        title=title,
    )


class RetrievalControllerTests(unittest.TestCase):
    def test_stops_when_entities_semantics_and_constraints_are_covered(self):
        question = "Who led Example Org in Taiwan in 2024 according to the official report?"
        evidence_text = (
            "The official report states that Alice Chen led Example Org in Taiwan during 2024."
        )
        nlp = FakeNLP(
            {
                question: [
                    ("Example Org", "ORG"),
                    ("Taiwan", "GPE"),
                    ("2024", "DATE"),
                ],
                evidence_text: [
                    ("Alice Chen", "PERSON"),
                    ("Example Org", "ORG"),
                    ("Taiwan", "GPE"),
                    ("2024", "DATE"),
                ],
            }
        )
        controller = RetrievalController(
            semantic_scorer=FakeSemanticScorer([0.82]),
            nlp=nlp,
        )

        decision = controller.assess(
            question=question,
            evidence_items=[evidence(evidence_text)],
        )

        self.assertFalse(decision.need_next_hop)
        self.assertEqual(decision.reason, "sufficient_evidence")
        self.assertEqual(decision.scores["entity_coverage"], 1.0)
        self.assertEqual(decision.scores["constraint_coverage"], 1.0)
        self.assertEqual(decision.scores["semantic_relevance"], 0.82)

    def test_requests_next_hop_when_year_and_answer_role_are_missing(self):
        question = "Who led Example Org in Taiwan in 2024?"
        evidence_text = "Example Org has offices in Taiwan."
        nlp = FakeNLP(
            {
                question: [
                    ("Example Org", "ORG"),
                    ("Taiwan", "GPE"),
                    ("2024", "DATE"),
                ],
                evidence_text: [
                    ("Example Org", "ORG"),
                    ("Taiwan", "GPE"),
                ],
            }
        )
        controller = RetrievalController(
            semantic_scorer=FakeSemanticScorer([0.8]),
            nlp=nlp,
        )

        decision = controller.assess(
            question=question,
            evidence_items=[evidence(evidence_text)],
        )

        self.assertTrue(decision.need_next_hop)
        self.assertIn("entity_coverage", decision.missing_info)
        self.assertIn("constraint_coverage", decision.missing_info)
        self.assertIn("year:2024", decision.scores["constraint_details"]["missing"])
        self.assertIn("answer_role:person", decision.scores["constraint_details"]["missing"])

    def test_requests_next_hop_when_semantic_relevance_is_low(self):
        question = "What is the capital of France?"
        evidence_text = "A recipe for baking sourdough bread."
        controller = RetrievalController(
            semantic_scorer=FakeSemanticScorer([0.12]),
            nlp=FakeNLP({}),
        )

        decision = controller.assess(
            question=question,
            evidence_items=[evidence(evidence_text)],
        )

        self.assertTrue(decision.need_next_hop)
        self.assertIn("semantic_relevance", decision.missing_info)

    def test_reranks_evidence_with_the_same_three_signal_formula(self):
        controller = RetrievalController(
            semantic_scorer=FakeSemanticScorer([0.15, 0.9]),
            nlp=FakeNLP({}),
        )
        low = evidence("Weakly related background.", title="Low")
        high = EvidenceItem(
            evidence_id="E2",
            source_id="S2",
            query_id="H2",
            text="Strongly related evidence.",
            title="High",
        )

        ranked, diagnostics = controller.rank_evidence(
            question="Explain the relevant evidence.",
            evidence_items=[low, high],
        )

        self.assertEqual([item.source_id for item in ranked], ["S2", "S1"])
        self.assertGreater(ranked[0].evidence_quality, ranked[1].evidence_quality)
        self.assertEqual(diagnostics[0]["semantic_relevance"], 0.9)


if __name__ == "__main__":
    unittest.main()
