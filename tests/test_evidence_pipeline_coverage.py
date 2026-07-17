from __future__ import annotations

import unittest

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.next_hop_query.rag_filter import RAGFilterResult
from tools.search_result_builder.retrieval_control import IterativeRetrievalControl
from tools.search_result_builder.source_analyze.rag_labeler import RAGLabelResult
from tools.search_result_builder.source_analyze.seer.source_filter import SourceFilter


class SourceFilterCoverageTests(unittest.TestCase):
    def test_records_question_echo_signal_without_blocking_source(self):
        question = "Who led Example Org in Taiwan in 2024?"
        sources = [
            SearchSourceCandidate(
                source_id="S1",
                query_id="Q1",
                title="Example Org Taiwan 2024",
                url="https://example.com/one",
                snippet="Who led Example Org in Taiwan in 2024?",
                rank=1,
            ),
            SearchSourceCandidate(
                source_id="S2",
                query_id="Q1",
                title="Example Org leadership",
                url="https://example.com/two",
                snippet="Leadership note: Alice Chen was named director after a board vote.",
                rank=2,
            ),
        ]
        source_filter = SourceFilter(
            max_urls_per_domain=3,
            min_sources=2,
            semantic_echo_threshold=2.0,
            lexical_echo_threshold=0.25,
            max_new_information_ratio=0.65,
        )

        filtered = source_filter.filter_sources(sources, question=question, fetch_limit=2)

        self.assertEqual(len(filtered), 2)
        self.assertFalse(sources[0].blocked)
        self.assertIn("question_echo_only", sources[0].filter_reasons)


class FakeEmbedder:
    def embed(self, texts):
        del texts
        return [[1.0]]


class FakeIndex:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query_vector, top_k):
        del query_vector, top_k
        self.calls += 1
        if self.calls == 1:
            return [(["doc-1"], [0.82])]
        return [([], [])]


class FakeRetriever:
    model_type = "other"

    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.index = FakeIndex()
        self.passage_map = {
            "doc-1": {
                "id": "doc-1",
                "title": "Example Org background note",
                "text": "Example Org has offices in Taiwan and published a general background note.",
                "url": "https://example.com/leadership",
            }
        }


class NoContinueLabeler:
    def label_texts(self, *, question, texts):
        del question
        return [
            RAGLabelResult(
                label="useless",
                kept_tokens=[],
                metadata={"sequence_tag": "<TERMINATE>"},
            )
            for _ in texts
        ]


class FixedFilter:
    def build_query(self, *, question, evidence_items):
        del question
        text = " ".join(item.text for item in evidence_items)
        return RAGFilterResult(
            query="Example Org Taiwan 2024 Alice Chen",
            kept_evidence_tokens=["Example", "Org", "Taiwan", "2024", "Alice", "Chen"],
            fallback_used=True,
            metadata={"method": "test_filter", "source_text": text},
        )


class IterativeRetrievalCoverageTests(unittest.TestCase):
    def test_no_bridge_span_stops_without_fallback_next_hop_query(self):
        controller = IterativeRetrievalControl(
            retriever=FakeRetriever(),
            labeler=NoContinueLabeler(),
            rag_filter=FixedFilter(),
            max_iter=2,
            top_k=1,
            min_retrieval_score=0.0,
            relative_score_margin=1.0,
        )

        result = controller.run("Who led Example Org in Taiwan in 2024?")

        self.assertEqual(result.searched_queries[0], "Who led Example Org in Taiwan in 2024?")
        self.assertEqual(len(result.searched_queries), 1)
        self.assertEqual(result.rounds[0].stop_reason, "no_continue_chunks")
        self.assertFalse(
            any(document.valid_for_next_hop for document in result.rounds[0].documents)
        )


if __name__ == "__main__":
    unittest.main()
