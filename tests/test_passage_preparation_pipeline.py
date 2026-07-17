from __future__ import annotations

import unittest

from tools.search_result_builder.corpus import StructuredDocumentExtractor
from tools.search_result_builder.passage_candidate_selector import (
    PassageCandidateSelector,
)
from tools.search_result_builder.retrieval_control import (
    IterativeRetrievalControl,
    RetrievedDocumentTrace,
)
from tools.search_result_builder.source_analyze.labeler_input_builder import (
    LabelerInputBuilder,
)


class StructuredDocumentExtractorTests(unittest.TestCase):
    def test_html_table_rows_remain_atomic_and_keep_headers(self):
        extractor = StructuredDocumentExtractor()
        html = """
        <html><body>
          <h2>Discography</h2>
          <table>
            <tr><th>Artist</th><th>Studio albums</th></tr>
            <tr><td>Example Band</td><td>7</td></tr>
            <tr><td>Second Band</td><td>4</td></tr>
          </table>
          <p>The article also describes the band's history.</p>
        </body></html>
        """

        units = extractor.extract(html)

        table_rows = [unit for unit in units if unit.unit_type == "table_row"]
        self.assertEqual(len(table_rows), 2)
        self.assertIn("Section: Discography", table_rows[0].text)
        self.assertIn("Columns: Artist | Studio albums", table_rows[0].text)
        self.assertIn("Row: Example Band | 7", table_rows[0].text)
        self.assertTrue(any(unit.unit_type == "paragraph" for unit in units))


class PassageCandidateSelectorTests(unittest.TestCase):
    def setUp(self):
        self.passages = {
            "page-a-000": {
                "id": "page-a-000",
                "title": "Unrelated result",
                "text": "A general article about books and publishing.",
                "url": "https://one.example/a",
            },
            "page-b-000": {
                "id": "page-b-000",
                "title": "Annie Levin interview",
                "text": "On June 27 2022 writer Annie Levin discussed her work.",
                "url": "https://two.example/b",
            },
            "page-c-000": {
                "id": "page-c-000",
                "title": "Research policy",
                "text": "Wikipedia requires no original research in published articles.",
                "url": "https://three.example/c",
            },
        }

    def test_bm25_recovers_lexical_passage_missing_from_dense_rank(self):
        selector = PassageCandidateSelector(max_neighbor_items=0)

        selections = selector.select(
            passage_map=self.passages,
            ranked_dense_lists={"dense_current": [("page-a-000", 0.91)]},
            lexical_query="writer Annie Levin June 27 2022",
            max_items=3,
        )

        by_id = {item.document["id"]: item for item in selections}
        self.assertIn("page-b-000", by_id)
        self.assertIn("bm25", by_id["page-b-000"].selection_sources)

    def test_adjacent_passage_is_added_without_displacing_primary(self):
        passages = {
            "castle-000": {
                "id": "castle-000",
                "title": "Castle",
                "text": "The castle was constructed in 1880.",
                "url": "https://castle.example/page",
            },
            "castle-001": {
                "id": "castle-001",
                "title": "Castle",
                "text": "It was later acquired by the city council.",
                "url": "https://castle.example/page",
            },
        }
        selector = PassageCandidateSelector(max_neighbor_items=1, max_per_domain=3)

        selections = selector.select(
            passage_map=passages,
            ranked_dense_lists={"dense_current": [("castle-000", 0.9)]},
            lexical_query="when was the castle constructed",
            max_items=2,
        )

        self.assertEqual(selections[0].document["id"], "castle-000")
        self.assertEqual(selections[1].document["id"], "castle-001")
        self.assertEqual(selections[1].expanded_from, "castle-000")


class LabelerInputBuilderTests(unittest.TestCase):
    def test_uses_current_query_once_and_plain_title_plus_passage(self):
        batch = LabelerInputBuilder().build_batch(
            question="Original question that should not be duplicated",
            current_query="current retrieval query",
            documents=[
                {
                    "title": "Source title",
                    "text": "The exact corpus passage.",
                }
            ],
        )

        self.assertEqual(batch.question_context, "current retrieval query")
        self.assertEqual(batch.texts, ["Source title The exact corpus passage."])
        self.assertNotIn("Source title:", batch.texts[0])
        self.assertNotIn("Passage:", batch.texts[0])


class GroundedSpanFallbackTests(unittest.TestCase):
    def test_classifier_failure_keeps_only_spans_found_in_source(self):
        controller = object.__new__(IterativeRetrievalControl)
        trace = RetrievedDocumentTrace(
            document_id="doc-1",
            title="Relevant source",
            text="The Dimond Center contains the KGOT radio studios.",
            url="https://example.com",
            retrieval_score=0.9,
            useful_spans=["Dimond Center", "invented place"],
            raw_labeler_spans=["KGOT", "invented place"],
        )

        controller._retain_grounded_spans(
            trace,
            reason="span_role_classifier_failed",
        )

        self.assertEqual(trace.useful_spans, ["Dimond Center", "KGOT"])
        self.assertEqual(trace.support_level, "unclassified")
        self.assertEqual(trace.label_status, "grounded_unclassified")
        self.assertFalse(trace.valid_for_next_hop)


if __name__ == "__main__":
    unittest.main()
