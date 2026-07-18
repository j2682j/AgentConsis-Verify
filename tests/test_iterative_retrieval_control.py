from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.search_result_builder.next_hop_query.rag_filter import RAGFilterResult
from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.retrieval_control import (
    IterativeRetrievalControl,
    WebRetrievalControl,
)
from tools.search_result_builder.source_analyze.rag_labeler import (
    CONTINUE_TAG,
    RAGLabelResult,
)
from tools.search_result_builder.source_analyze.seer import SourceFilter


class FakeEmbedder:
    def prepare_query_text(self, text):
        return f"query: {text}"

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeIndex:
    def __init__(self, score=0.9):
        self.calls = 0
        self.score = score

    def search(self, query_vectors, top_k):
        del query_vectors, top_k
        self.calls += 1
        return [(["D1"], [self.score])]


class FakeRetriever:
    model_type = "multilingual-e5-base"

    def __init__(self, score=0.9):
        self.embedder = FakeEmbedder()
        self.index = FakeIndex(score)
        self.passage_map = {
            "D1": {
                "id": "D1",
                "title": "Relevant document",
                "text": "A useful bridge token appears here.",
                "url": "https://example.com/doc",
            }
        }


class FakeLabeler:
    def __init__(self, kept_tokens=None):
        self.calls = []
        self.kept_tokens = (
            ["bridge"] if kept_tokens is None else kept_tokens
        )

    def label_texts(self, *, question, texts):
        self.calls.append((question, texts))
        return [
            RAGLabelResult(
                label="useful",
                kept_tokens=list(self.kept_tokens),
                metadata={
                    "sequence_tag": CONTINUE_TAG,
                    "continue_probability": 0.9,
                    "terminate_probability": 0.1,
                },
            )
            for _ in texts
        ]


class FakeFilter:
    def __init__(self, queries):
        self.queries = iter(queries)
        self.calls = []

    def build_query(self, *, question, evidence_items):
        self.calls.append((question, evidence_items))
        return RAGFilterResult(
            query=next(self.queries),
            kept_evidence_tokens=["bridge"],
            metadata={"method": "fake_filter"},
        )


class IterativeRetrievalControlTests(unittest.TestCase):
    def test_incomplete_goal_does_not_accept_surface_sufficiency(self):
        labeler = FakeLabeler()
        rag_filter = FakeFilter(["original query"])
        controller = IterativeRetrievalControl(
            retriever=FakeRetriever(),
            labeler=labeler,
            rag_filter=rag_filter,
            max_iter=4,
            top_k=1,
        )

        result = controller.run("original query")

        self.assertEqual(result.stop_reason, "goal_incomplete_no_viable_recovery")
        self.assertEqual(result.searched_queries, ["original query"])
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.rounds[0].documents[0].retrieval_score, 0.9)
        self.assertEqual(
            labeler.calls[0][1],
            ["Relevant document A useful bridge token appears here."],
        )
        self.assertEqual(rag_filter.calls, [])

    def test_exhausted_corpus_does_not_repeat_duplicate_documents(self):
        controller = IterativeRetrievalControl(
            retriever=FakeRetriever(),
            labeler=FakeLabeler(),
            rag_filter=FakeFilter(["next query"]),
            max_iter=4,
            top_k=1,
        )

        result = controller.run("original query")

        self.assertEqual(result.stop_reason, "goal_incomplete_no_viable_recovery")
        self.assertEqual(result.searched_queries, ["original query"])
        self.assertEqual(len(result.rounds), 1)

    def test_empty_initial_query_does_not_search(self):
        retriever = FakeRetriever()
        controller = IterativeRetrievalControl(
            retriever=retriever,
            labeler=FakeLabeler(),
            rag_filter=FakeFilter([]),
        )

        result = controller.run("   ")

        self.assertEqual(result.stop_reason, "empty_initial_query")
        self.assertEqual(retriever.index.calls, 0)

    def test_continue_without_useful_tokens_is_not_sent_to_filter(self):
        rag_filter = FakeFilter(["next query"])
        controller = IterativeRetrievalControl(
            retriever=FakeRetriever(),
            labeler=FakeLabeler(kept_tokens=[]),
            rag_filter=rag_filter,
            max_iter=4,
            top_k=1,
        )

        result = controller.run("original query")

        self.assertEqual(
            result.stop_reason,
            "goal_incomplete_no_viable_recovery",
        )
        self.assertEqual(rag_filter.calls, [])
        self.assertFalse(
            result.rounds[0].filter_metadata["goal_completion"]["sufficient"]
        )

    def test_continue_below_absolute_e5_threshold_is_not_sent_to_filter(self):
        rag_filter = FakeFilter(["next query"])
        controller = IterativeRetrievalControl(
            retriever=FakeRetriever(score=0.7),
            labeler=FakeLabeler(),
            rag_filter=rag_filter,
            max_iter=4,
            top_k=1,
            min_retrieval_score=0.75,
        )

        result = controller.run("original query")

        self.assertEqual(
            result.stop_reason,
            "goal_incomplete_no_viable_recovery",
        )
        self.assertEqual(rag_filter.calls, [])
        self.assertFalse(
            result.rounds[0].filter_metadata["goal_completion"]["sufficient"]
        )


class FakeQueryGenerator:
    def plan(self, question, max_queries):
        del question, max_queries
        return {
            "queries": ["first web query", "second web query"],
            "salient_spans": ["bridge term"],
            "precision_needed": True,
        }


class FakeSearchTool:
    def __init__(self):
        self.calls = []

    def run(self, parameters):
        self.calls.append(parameters)
        query = parameters["input"]
        return {
            "backend": "fake-search",
            "results": [
                {
                    "title": f"Title for {query}",
                    "url": f"https://example.com/{len(self.calls)}",
                    "content": (
                        f"{query} provides a sufficiently long passage for "
                        "corpus cleaning and chunking during the retrieval test."
                    ),
                }
            ],
            "notices": [],
        }


class FakeCorpusRecord:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class FakeExporter:
    def export(self, records, output_path):
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = list(records)
        path.write_text(
            "\n".join(json.dumps(record.to_dict()) for record in values) + "\n",
            encoding="utf-8",
        )
        return len(values)


class FakeCorpusBuilder:
    def __init__(self):
        self.calls = []
        self.exporter = FakeExporter()

    def build_records(self, sources, **kwargs):
        self.calls.append((sources, kwargs))
        return [
            FakeCorpusRecord(
                {
                    "id": f"page-{index:03d}-000",
                    "title": source.title,
                    "text": source.snippet,
                    "url": source.url,
                    "retrieved_at": "2026-06-25",
                }
            )
            for index, source in enumerate(sources, start=1)
        ]


class WebRetrievalControlTests(unittest.TestCase):
    def test_searches_generated_queries_before_building_corpus(self):
        search_tool = FakeSearchTool()
        corpus_builder = FakeCorpusBuilder()
        control = WebRetrievalControl(
            query_generator=FakeQueryGenerator(),
            search_tool=search_tool,
            corpus_builder=corpus_builder,
            max_queries=2,
        )

        sources, traces = control._search_queries(
            ["first web query", "second web query"]
        )

        self.assertEqual(
            [call["input"] for call in search_tool.calls],
            ["first web query", "second web query"],
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].backend, "fake-search")

        records = corpus_builder.build_records(
            sources,
            fetch_missing=True,
            max_pages_to_fetch=12,
        )
        self.assertEqual(len(records), 2)
        self.assertIn("first web query", records[0].to_dict()["text"])

    def test_seer_filter_can_rescue_soft_domain_limit_for_minimum_coverage(self):
        sources = [
            SearchSourceCandidate(
                source_id="S1",
                query_id="Q1",
                title="Low quality answer",
                url="https://www.quora.com/example",
                snippet="A low trust result.",
            ),
            SearchSourceCandidate(
                source_id="S2",
                query_id="Q1",
                title="First",
                url="https://example.com/one",
                snippet="A sufficiently distinct first source snippet.",
            ),
            SearchSourceCandidate(
                source_id="S3",
                query_id="Q1",
                title="Second",
                url="https://example.com/two",
                snippet="A sufficiently distinct second source snippet.",
            ),
        ]

        filtered = SourceFilter(max_urls_per_domain=1).filter_sources(
            sources,
            question="Unrelated question",
            fetch_limit=3,
        )

        self.assertEqual([source.source_id for source in filtered], ["S2", "S3"])
        self.assertEqual(sources[0].block_reason, "blocked_domain")
        self.assertEqual(sources[2].block_reason, "")
        self.assertIn(
            "rescued_soft_block:domain_result_limit",
            sources[2].filter_reasons,
        )

    def test_seer_filter_marks_question_semantic_echo_without_blocking(self):
        question = "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"
        sources = [
            SearchSourceCandidate(
                source_id="S1",
                query_id="Q1",
                title="Question mirror",
                url="https://example.com/mirror",
                snippet=question,
            )
        ]

        with patch(
            "tools.search_result_builder.source_analyze.seer.source_filter.semantic_similarity_score",
            return_value=0.97,
        ):
            filtered = SourceFilter().filter_sources(
                sources,
                question=question,
                fetch_limit=3,
            )

        self.assertEqual([source.source_id for source in filtered], ["S1"])
        self.assertFalse(sources[0].blocked)
        self.assertIn("question_semantic_echo", sources[0].filter_reasons)
        self.assertTrue(any(reason.startswith("semantic_echo=") for reason in sources[0].filter_reasons))
        self.assertTrue(any(reason.startswith("lexical_overlap=") for reason in sources[0].filter_reasons))
        self.assertTrue(any(reason.startswith("new_information_ratio=") for reason in sources[0].filter_reasons))

    def test_seer_filter_keeps_similar_source_with_new_information(self):
        question = "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"
        sources = [
            SearchSourceCandidate(
                source_id="S1",
                query_id="Q1",
                title="Merriam-Webster Word of the Day: Jingoism",
                url="https://example.com/jingoism",
                snippet=(
                    "The June 27 2022 Word of the Day quotes Annie Levin from "
                    "The New York Observer in a discussion of jingoism, nationalism, "
                    "chauvinism, editorial commentary, cultural institutions, "
                    "political rhetoric, publication history, archive citation, "
                    "observer columnist, magazine excerpt, and usage examples."
                ),
            )
        ]

        with patch(
            "tools.search_result_builder.source_analyze.seer.source_filter.semantic_similarity_score",
            return_value=0.93,
        ):
            filtered = SourceFilter().filter_sources(
                sources,
                question=question,
                fetch_limit=3,
            )

        self.assertEqual([source.source_id for source in filtered], ["S1"])
        self.assertFalse(sources[0].blocked)


if __name__ == "__main__":
    unittest.main()
