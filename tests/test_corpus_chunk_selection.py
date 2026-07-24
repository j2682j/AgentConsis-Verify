"""Corpus construction keeps the chunks that answer the question.

A long reference page yields far more chunks than the per-page budget. Taking
the first N kept only the page's introduction, so a fact stated further down
never entered the corpus and no amount of retrieval quality could recover it.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.corpus.web_corpus_builder import WebCorpusBuilder


FILLER = (
    "This introductory section describes the project history and its general "
    "release policy in broad terms without naming any specific component. "
)
ANSWER = (
    "Other predictors fix semi_supervised.BaseLabelPropagation to correctly "
    "implement LabelPropagation and LabelSpreading as documented. "
)


def page(answer_position: str = "late") -> str:
    body = [FILLER * 6 for _ in range(8)]
    if answer_position == "late":
        body.append(ANSWER * 3)
    else:
        body.insert(0, ANSWER * 3)
    return "\n\n".join(body)


def build(builder: WebCorpusBuilder, text: str, *, question: str, limit: int):
    return builder.build_records(
        [{
            "source_id": "S1",
            "query_id": "Q1",
            "title": "changelog",
            "url": "https://example.test/whats_new",
            "raw_content": text,
            "snippet": "",
        }],
        fetch_missing=False,
        max_chunks_per_url=limit,
        max_records=limit,
        question=question,
    )


class ChunkSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = WebCorpusBuilder()
        self.question = (
            "In the Scikit-Learn changelog, what other predictor base command "
            "received a bug fix?"
        )

    def test_answer_late_in_page_still_enters_the_corpus(self) -> None:
        records = build(self.builder, page("late"), question=self.question, limit=3)
        self.assertTrue(records)
        self.assertLessEqual(len(records), 3)
        self.assertTrue(
            any("BaseLabelPropagation" in r.text for r in records),
            "a chunk answering the question must survive the per-page budget",
        )

    def test_selection_never_exceeds_the_budget(self) -> None:
        for limit in (1, 2, 5):
            with self.subTest(limit=limit):
                records = build(
                    self.builder, page("late"), question=self.question, limit=limit
                )
                self.assertLessEqual(len(records), limit)

    def test_selection_picks_by_relevance_and_keeps_document_order(self) -> None:
        """The two invariants of the selector, asserted directly.

        Going through the whole builder cannot express this: the chunker
        merges adjacent paragraphs, so a chunk boundary is not a paragraph
        boundary and per-paragraph markers do not survive.
        """
        chunks = [
            FILLER,
            "predictor base command bug fix BaseLabelPropagation semi_supervised",
            FILLER,
            FILLER,
            "scikit-learn changelog predictor bug fix entry for this release",
            FILLER,
        ]
        self.builder._question_terms = self.builder._informative_terms(self.question)

        selected = self.builder._select_relevant_chunks(chunks, limit=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, [chunks[1], chunks[4]])
        indexes = [chunks.index(text) for text in selected]
        self.assertEqual(indexes, sorted(indexes))

    def test_without_a_question_behaviour_is_document_order(self) -> None:
        """Callers that pass no question must be unaffected by this change."""
        text = page("late")
        records = build(self.builder, text, question="", limit=3)
        plain = self.builder.build_records(
            [{
                "source_id": "S1", "query_id": "Q1", "title": "changelog",
                "url": "https://example.test/whats_new", "raw_content": text,
                "snippet": "",
            }],
            fetch_missing=False, max_chunks_per_url=3, max_records=3,
        )
        self.assertEqual([r.text for r in records], [r.text for r in plain])

    def test_short_page_is_untouched(self) -> None:
        text = ANSWER + FILLER
        records = build(self.builder, text, question=self.question, limit=20)
        self.assertTrue(any("BaseLabelPropagation" in r.text for r in records))


if __name__ == "__main__":
    unittest.main()
