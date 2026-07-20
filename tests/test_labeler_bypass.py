from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tools.search_result_builder.evidence import (
    ANSWER_SUPPORT,
    NOISE,
    PassageEvidenceUnitBuilder,
    SpanRoleBatchResult,
    SpanRoleResult,
)
from tools.search_result_builder.retrieval_control import IterativeRetrievalControl
from tools.search_result_builder.query import SearchIntentPlan
from tools.evidence.fact_extraction import SemanticExtractionResult


class _SemanticEmbedder:
    def prepare_query_text(self, text: str) -> str:
        return f"query: {text}"

    def prepare_passage_text(self, text: str) -> str:
        return f"passage: {text}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if lowered.startswith("query:") or "three studio albums" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class _Index:
    def search(self, query_vectors, top_k):
        del query_vectors, top_k
        return [(["D1"], [0.9])]


class _Retriever:
    model_type = "multilingual-e5-base"

    def __init__(self) -> None:
        self.embedder = _SemanticEmbedder()
        self.index = _Index()
        self.passage_map = {
            "D1": {
                "id": "D1",
                "title": "Mercedes Sosa discography",
                "text": (
                    "The page contains general biographical information. "
                    "Mercedes Sosa released three studio albums between 2000 and 2009."
                ),
                "url": "https://example.com/discography",
            }
        }


class _ForbiddenLabeler:
    def __init__(self) -> None:
        self.called = False

    def label_texts(self, **kwargs):
        del kwargs
        self.called = True
        raise AssertionError("Labeler must not run in bypass mode")


class _UnboundAnswerSupportClassifier:
    def classify_batch(self, *, spans, **kwargs):
        del kwargs
        return SpanRoleBatchResult(
            results=[
                SpanRoleResult(
                    id=span.id,
                    text=span.text,
                    role=NOISE,
                    model_role=ANSWER_SUPPORT,
                    goal_id="G1",
                )
                for span in spans
            ],
            diagnostics={"success": True},
        )


class _EmptySemanticFactExtractor:
    model_name = "fake-extractor"
    max_units_per_call = 8

    def extract_batch(self, **kwargs):
        del kwargs
        return SemanticExtractionResult(diagnostics={"success": True})


class LabelerBypassTests(unittest.TestCase):
    def test_evidence_unit_builder_retains_semantically_relevant_sentence(self):
        builder = PassageEvidenceUnitBuilder(max_units=1)

        result = builder.build(
            question="How many studio albums were released?",
            documents=[
                {
                    "id": "D1",
                    "title": "Discography",
                    "text": (
                        "An unrelated biographical sentence. "
                        "Mercedes Sosa released three studio albums between 2000 and 2009."
                    ),
                }
            ],
            embedder=_SemanticEmbedder(),
        )

        self.assertEqual(len(result.units), 1)
        self.assertIn("three studio albums", result.units[0].text)
        self.assertEqual(
            result.diagnostics["ranking_method"],
            "e5_semantic_similarity",
        )

    def test_iterative_control_does_not_call_labeler_in_bypass_mode(self):
        labeler = _ForbiddenLabeler()
        controller = IterativeRetrievalControl(
            retriever=_Retriever(),
            labeler=labeler,
            bypass_labeler=True,
            max_iter=1,
            top_k=1,
            min_retrieval_score=0.0,
            relative_score_margin=1.0,
        )

        with patch.object(controller, "_apply_span_role_classification"), patch.object(
            controller,
            "_apply_cross_context_fact_extraction",
        ):
            result = controller.run(
                "How many studio albums were released?",
                intent_plan=SearchIntentPlan.from_dict(
                    {
                        "answer_role": "number of studio albums",
                        "relation_plan": {
                            "active_goal_id": "G1",
                            "goals": [
                                {
                                    "goal_id": "G1",
                                    "subject": "Mercedes Sosa",
                                    "relation": "released",
                                    "target": "number of studio albums",
                                    "state": "active",
                                },
                                {
                                    "goal_id": "G2",
                                    "subject": "studio albums",
                                    "relation": "count",
                                    "target": "final number",
                                    "state": "pending",
                                }
                            ],
                        },
                    }
                ),
            )

        self.assertFalse(labeler.called)
        document = result.rounds[0].documents[0]
        self.assertEqual(document.sequence_tag, "<BYPASS>")
        self.assertTrue(document.useful_spans)
        self.assertFalse(document.labeler_diagnostics["labeler_called"])
        self.assertIn("labeler_bypass", result.rounds[0].filter_metadata)

    def test_unbound_model_answer_support_is_preserved_as_bridge(self):
        controller = IterativeRetrievalControl(
            retriever=_Retriever(),
            bypass_labeler=True,
            span_role_classifier=_UnboundAnswerSupportClassifier(),
            semantic_fact_extractor=_EmptySemanticFactExtractor(),
            max_iter=1,
            top_k=1,
            min_retrieval_score=0.0,
            relative_score_margin=1.0,
        )

        with patch.object(controller, "_apply_cross_context_fact_extraction"):
            result = controller.run(
                "How many studio albums were released?",
                intent_plan=SearchIntentPlan.from_dict(
                    {
                        "answer_role": "number of studio albums",
                        "relation_plan": {
                            "active_goal_id": "G1",
                            "goals": [
                                {
                                    "goal_id": "G1",
                                    "subject": "Mercedes Sosa",
                                    "relation": "released",
                                    "target": "number of studio albums",
                                    "state": "active",
                                },
                                {
                                    "goal_id": "G2",
                                    "subject": "studio albums",
                                    "relation": "count",
                                    "target": "final number",
                                    "state": "pending",
                                }
                            ],
                        },
                    }
                ),
            )

        document = result.rounds[0].documents[0]
        self.assertTrue(document.bridge_spans)
        self.assertTrue(document.valid_for_next_hop)
        self.assertEqual(
            document.span_roles[0]["role_repair"],
            "unbound_answer_support_demoted_to_bridge",
        )


if __name__ == "__main__":
    unittest.main()
