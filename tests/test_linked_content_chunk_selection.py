"""Pin which chunks of a linked page survive `max_chunks`.

`build_enriched_records` fetches the page a structured record points at, chunks
it, and keeps `max_chunks` of them. It used to keep the leading chunks, which
discards by position rather than by relevance: on level1_final_06 a scikit-learn
changelog of 25,618 characters reached retrieval as 674, and the entry the task
asked about sat past the cut. Every linked page large enough to be chunked has
this shape, so the answer's position in the page decided the outcome.

These tests hold the selection to relevance when a question and embedder are
available, keep the document order of whatever is selected so `passage_index`
and downstream adjacency stay meaningful, and leave the old leading-chunk
behaviour in place when there is nothing to rank with.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.corpus.chunker import DocumentChunker
from tools.search_result_builder.corpus.web_corpus_builder import (
    CorpusRecord,
    WebCorpusBuilder,
)
from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
    PageFetchResult,
)

ANSWER = "BaseLabelPropagation"
QUESTION = "which predictor base command received a bug fix?"

# Ten chunk-sized sections; the answer sits in the last one, past any leading cut.
SECTIONS = [
    "Release notes describe installation steps and packaging changes only.",
    "Documentation typos were corrected across the tutorial pages this cycle.",
    "The build system now pins its compiler toolchain for reproducible wheels.",
    "Continuous integration moved to a newer runner image without other changes.",
    "Deprecation warnings were reworded to name their replacement directly.",
    "Example galleries were regenerated so their figures match current output.",
    "Benchmark scripts gained a flag controlling their random seed.",
    "Translation files were resynchronised with the English source strings.",
    "Contributor guidelines now explain the review checklist in more detail.",
    f"A bug fix landed for {ANSWER} affecting its predictor base command.",
]
PAGE = "\n\n".join(f"{index}. {body}" for index, body in enumerate(SECTIONS, start=1))


class _KeywordEmbedder:
    """Score by answer-term overlap, so relevance ordering is deterministic.

    Returns two-dimensional vectors: dimension 0 marks the query, dimension 1
    carries the term hit, which makes cosine similarity rank a chunk mentioning
    the term above every chunk that does not.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for index, text in enumerate(texts):
            if index == 0:
                vectors.append([0.0, 1.0])
                continue
            hit = 1.0 if ANSWER.casefold() in text.casefold() else 0.0
            vectors.append([1.0, hit * 4.0])
        return vectors


def _record() -> CorpusRecord:
    return CorpusRecord(
        id="record-001-001-001",
        title="Version 0.19",
        text="Record Type: article\nTitle: Version 0.19",
        url="https://example.org/whats_new",
        retrieved_at="2026-07-29",
        record_type="article",
        record_id="record-001-001",
        content_url="https://example.org/whats_new/v0.19",
        parent_url="https://example.org/whats_new",
        extraction_method="html_list",
    )


def _builder() -> WebCorpusBuilder:
    def fetcher(url: str, *, max_tokens: int, **kwargs: object) -> PageFetchResult:
        return PageFetchResult(content=PAGE, method="test")

    return WebCorpusBuilder(
        chunker=DocumentChunker(max_chars=120, overlap_chars=0, min_chars=10),
        page_fetcher=fetcher,
    )


class LinkedContentChunkSelectionTest(unittest.TestCase):
    def test_relevant_chunk_survives_even_when_it_is_last(self) -> None:
        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=3,
            question=QUESTION,
            embedder=_KeywordEmbedder(),
        )

        joined = " ".join(item.text for item in enriched).casefold()
        self.assertIn(ANSWER.casefold(), joined)
        self.assertEqual(len(enriched), 3)

    def test_leading_chunks_would_drop_the_answer(self) -> None:
        """Guard the regression direction: no ranker means position decides."""

        enriched = _builder().build_enriched_records(_record(), max_chunks=3)

        joined = " ".join(item.text for item in enriched).casefold()
        self.assertNotIn(ANSWER.casefold(), joined)

    def test_selected_chunks_keep_document_order(self) -> None:
        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=4,
            question=QUESTION,
            embedder=_KeywordEmbedder(),
        )

        indexes = [item.passage_index for item in enriched]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(indexes, list(range(len(enriched))))

    def test_a_page_within_the_budget_is_kept_whole(self) -> None:
        expected = len(DocumentChunker(max_chars=120, overlap_chars=0, min_chars=10).chunk(PAGE))

        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=50,
            question=QUESTION,
            embedder=_KeywordEmbedder(),
        )

        joined = " ".join(item.text for item in enriched).casefold()
        self.assertIn(ANSWER.casefold(), joined)
        self.assertEqual(len(enriched), expected)
        self.assertFalse(any(item.content_truncated for item in enriched))

    def test_a_failing_embedder_falls_back_to_leading_chunks(self) -> None:
        class _Broken:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embedder unavailable")

        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=3,
            question=QUESTION,
            embedder=_Broken(),
        )

        self.assertEqual(len(enriched), 3)
        joined = " ".join(item.text for item in enriched).casefold()
        self.assertNotIn(ANSWER.casefold(), joined)

    def test_an_array_returning_embedder_ranks_instead_of_raising(self) -> None:
        """The production embedder returns an array, and the stub above does not.

        That gap hid a real failure for a full benchmark run. The guard on the
        returned vectors read `if not vectors`, which raises on a numpy array
        holding more than one row, and it sits outside the `try` that catches a
        failing embedder -- so the error escaped `_select_linked_chunks` and
        took the entire search evidence build with it. Measured on
        level1_final_18, `prepared_status` was `prepared_usable` on 6 of 53
        tasks against 28 in the run before it, and the run still reported a
        normal score.
        """

        numpy = __import__("numpy")

        class _ArrayEmbedder(_KeywordEmbedder):
            def embed(self, texts: list[str]):
                return numpy.asarray(super().embed(texts), dtype="float32")

        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=3,
            question=QUESTION,
            embedder=_ArrayEmbedder(),
        )

        joined = " ".join(item.text for item in enriched).casefold()
        self.assertIn(ANSWER.casefold(), joined)
        self.assertEqual(len(enriched), 3)

    def test_an_empty_array_falls_back_instead_of_ranking(self) -> None:
        numpy = __import__("numpy")

        class _EmptyArray:
            def embed(self, texts: list[str]):
                return numpy.empty((0, 2), dtype="float32")

        enriched = _builder().build_enriched_records(
            _record(),
            max_chunks=3,
            question=QUESTION,
            embedder=_EmptyArray(),
        )

        self.assertEqual(len(enriched), 3)
        joined = " ".join(item.text for item in enriched).casefold()
        self.assertNotIn(ANSWER.casefold(), joined)


if __name__ == "__main__":
    unittest.main()
