from __future__ import annotations

import unittest

from tools.search_result_builder.corpus import (
    CorpusRecord,
    CollectionRecordExtractor,
    DocumentChunker,
    WebCorpusBuilder,
)
from tools.search_result_builder.source_analyze.seer.page_content_fetcher import PageFetchResult
from tools.search_result_builder.embeddings.embedder import Embedder
from tools.search_result_builder.passage_candidate_selector import PassageCandidateSelector
from tools.search_result_builder.source_analyze.labeler_input_builder import LabelerInputBuilder


class CollectionRecordExtractorTests(unittest.TestCase):
    def test_publication_table_keeps_author_title_year_and_link_together(self):
        html = """
        <html><body><table>
          <tr><th>Paper Title</th><th>Authors</th><th>Year</th><th>Journal</th></tr>
          <tr><td><a href="/paper-a">Paper A</a></td><td>Alice Chen; Bob Lin</td><td>2024</td><td>Journal One</td></tr>
          <tr><td><a href="/paper-b">Paper B</a></td><td>Carol Wu</td><td>2023</td><td>Journal Two</td></tr>
        </table></body></html>
        """

        result = CollectionRecordExtractor().extract(
            html,
            parent_url="https://example.org/publications",
            source_title="Publication list",
            source_kind="collection",
        )

        self.assertEqual(len(result.records), 2)
        first, second = result.records
        self.assertEqual(first.record_type, "publication")
        self.assertEqual(first.title, "Paper A")
        self.assertEqual(first.authors, ("Alice Chen", "Bob Lin"))
        self.assertEqual(first.date, "2024")
        self.assertEqual(first.source, "Journal One")
        self.assertEqual(first.content_url, "https://example.org/paper-a")
        self.assertEqual(second.authors, ("Carol Wu",))
        self.assertNotIn("Carol Wu", first.authors)

    def test_same_title_different_years_remain_separate_records(self):
        html = """
        <table>
          <tr><th>Title</th><th>Year</th><th>Author</th></tr>
          <tr><td>Annual Review</td><td>2023</td><td>Alice</td></tr>
          <tr><td>Annual Review</td><td>2024</td><td>Bob</td></tr>
        </table>
        """

        result = CollectionRecordExtractor().extract(
            html,
            parent_url="https://example.org/archive",
        )

        self.assertEqual(len(result.records), 2)
        self.assertEqual([record.date for record in result.records], ["2023", "2024"])

    def test_repeated_article_cards_keep_language_country_and_content_link(self):
        html = """
        <section class="archive">
          <article class="entry">
            <h2><a href="/articles/one">Article One</a></h2>
            <time datetime="2025-01-02">January 2, 2025</time>
            <span class="language">English</span>
            <span class="country">Canada</span>
            <p>The first article summary.</p>
          </article>
          <article class="entry">
            <h2><a href="/articles/two">Article Two</a></h2>
            <time datetime="2025-02-03">February 3, 2025</time>
            <span class="language">French</span>
            <span class="country">France</span>
            <p>The second article summary.</p>
          </article>
        </section>
        """

        result = CollectionRecordExtractor().extract(
            html,
            parent_url="https://news.example/archive",
            source_title="Annual archive",
            source_kind="collection",
        )

        self.assertEqual(len(result.records), 2)
        first = result.records[0]
        self.assertEqual(first.title, "Article One")
        self.assertEqual(first.language, "English")
        self.assertEqual(first.country, "Canada")
        self.assertEqual(first.content_url, "https://news.example/articles/one")
        self.assertNotIn("Article Two", first.content)

    def test_json_ld_scholarly_article_is_extracted(self):
        html = """
        <script type="application/ld+json">
        {
          "@type": "ScholarlyArticle",
          "headline": "Structured Research",
          "author": [{"name": "Dana Lee"}, {"name": "Evan Yu"}],
          "datePublished": "2024-06-01",
          "publisher": {"name": "Research Press"},
          "url": "/papers/structured",
          "abstract": "A structured abstract."
        }
        </script>
        """

        result = CollectionRecordExtractor().extract(
            html,
            parent_url="https://research.example/list",
        )

        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.record_type, "publication")
        self.assertEqual(record.authors, ("Dana Lee", "Evan Yu"))
        self.assertEqual(record.source, "Research Press")

    # A regression pair around the repeated-block content cap.
    # level1_final_06 task d0633230 hit a section with 20+ paragraphs where the
    # answer sat past the 3rd, and _block_record kept only the first three,
    # silently dropping it. The extractor's raw-text sibling path has a mirror
    # test in test_corpus_chunk_selection.py; these two keep the HTML path in
    # sync with it.

    def _paged_section_html(self, answer_paragraph: str) -> str:
        """Two repeated <article>s (needed to trip the block detector).

        The version-0.19 article carries an itemised bug-fix section: many
        earlier fixes, the answer somewhere past the third item, more fixes
        after. Mirrors the shape of Sphinx-generated changelogs.
        """

        earlier = "\n".join(
            f"<p>Fix number {i} in module x.</p>" for i in range(1, 8)
        )
        later = "\n".join(
            f"<p>Fix number {i} in module y.</p>" for i in range(11, 15)
        )
        return f"""
        <div>
          <article>
            <h2><a href="/v019">Version 0.19</a></h2>
            <h3>Bug fixes</h3>
            {earlier}
            <p>{answer_paragraph}</p>
            {later}
          </article>
          <article>
            <h2><a href="/v018">Version 0.18</a></h2>
            <p>An earlier release notes summary.</p>
          </article>
        </div>
        """

    def test_repeated_block_keeps_content_past_the_third_paragraph(self):
        answer = (
            "Other predictors fix semi_supervised.BaseLabelPropagation to correctly "
            "implement LabelPropagation and LabelSpreading as documented."
        )

        result = CollectionRecordExtractor().extract(
            self._paged_section_html(answer),
            parent_url="https://example.test/whats_new",
            source_title="changelog",
            source_kind="web",
        )

        v019 = next(record for record in result.records if record.title == "Version 0.19")
        self.assertIn("BaseLabelPropagation", v019.content)

    def test_a_cap_of_three_would_drop_the_answer(self):
        """Explicit guard on the regression direction."""

        answer = (
            "Other predictors fix semi_supervised.BaseLabelPropagation to correctly "
            "implement LabelPropagation and LabelSpreading as documented."
        )

        result = CollectionRecordExtractor(max_block_content_paragraphs=3).extract(
            self._paged_section_html(answer),
            parent_url="https://example.test/whats_new",
            source_title="changelog",
            source_kind="web",
        )

        v019 = next(record for record in result.records if record.title == "Version 0.19")
        self.assertNotIn("BaseLabelPropagation", v019.content)

    def test_default_content_paragraph_cap_is_thirty(self):
        self.assertEqual(
            CollectionRecordExtractor().max_block_content_paragraphs, 30
        )

    # A regression pair around the repeated-block text-size cap.
    # The original 4000-char upper bound rejected sections whose full text was
    # larger, which silently excluded the v0.19.0 Bug fixes section (~8000
    # chars, contains BaseLabelPropagation) from ever becoming a block.

    def _oversized_section_html(self, answer_paragraph: str) -> str:
        # <section> matches the Sphinx-generated shape of the scikit-learn
        # changelog. <article> was tried first, but that tag is added to the
        # block pool unconditionally at the top of _extract_repeated_blocks and
        # bypasses max_block_text_chars entirely — the cap only ever guards
        # ul/ol/div/section repetition detection.
        long_fix = (
            "<p>A long detailed fix explanation with reasoning and "
            "reproduction steps that spans several sentences to push the "
            "total block text well past four thousand characters. "
            "It repeats context for the reader across each entry. </p>"
        )
        earlier = "\n".join(long_fix for _ in range(20))
        later = "\n".join(long_fix for _ in range(5))
        return f"""
        <div>
          <section>
            <h2><a href="/v019">Version 0.19</a></h2>
            <h3>Bug fixes</h3>
            {earlier}
            <p>{answer_paragraph}</p>
            {later}
          </section>
          <section>
            <h2><a href="/v018">Version 0.18</a></h2>
            {long_fix}
          </section>
        </div>
        """

    def test_oversized_section_still_becomes_a_record(self):
        answer = (
            "Other predictors fix semi_supervised.BaseLabelPropagation to correctly "
            "implement LabelPropagation and LabelSpreading as documented."
        )

        result = CollectionRecordExtractor(
            max_block_content_paragraphs=200,
        ).extract(
            self._oversized_section_html(answer),
            parent_url="https://example.test/whats_new",
            source_title="changelog",
            source_kind="web",
        )

        v019 = next(record for record in result.records if record.title == "Version 0.19")
        self.assertIn("BaseLabelPropagation", v019.content)

    def test_a_four_thousand_char_cap_would_drop_the_section(self):
        """Explicit guard on the regression direction."""

        answer = (
            "Other predictors fix semi_supervised.BaseLabelPropagation to correctly "
            "implement LabelPropagation and LabelSpreading as documented."
        )

        result = CollectionRecordExtractor(
            max_block_content_paragraphs=200,
            max_block_text_chars=4000,
        ).extract(
            self._oversized_section_html(answer),
            parent_url="https://example.test/whats_new",
            source_title="changelog",
            source_kind="web",
        )

        v019 = [record for record in result.records if record.title == "Version 0.19"]
        # With the old cap the oversize section is rejected outright, so no
        # record is produced for the version at all.
        self.assertFalse(v019)

    def test_default_text_char_cap_is_twenty_thousand(self):
        self.assertEqual(
            CollectionRecordExtractor().max_block_text_chars, 20000
        )


class StructuredCorpusIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.html = """
        <table>
          <tr><th>Title</th><th>Authors</th><th>Year</th><th>Language</th><th>Country</th></tr>
          <tr><td><a href="/a">Paper A</a></td><td>Alice</td><td>2024</td><td>English</td><td>Taiwan</td></tr>
          <tr><td><a href="/b">Paper B</a></td><td>Bob</td><td>2025</td><td>French</td><td>France</td></tr>
        </table>
        """

    def test_builder_bypasses_generic_chunking_and_exports_structured_fields(self):
        builder = WebCorpusBuilder(
            chunker=DocumentChunker(max_chars=300, overlap_chars=0, min_chars=10)
        )
        records = builder.build_records(
            [{
                "title": "Research archive",
                "url": "https://example.org/list",
                "raw_content": "Fallback text that should not replace records.",
                "raw_html": self.html,
                "source_kind": "collection",
            }],
            fetch_missing=False,
            retrieved_at="2026-07-17",
        )

        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first.record_type, "publication")
        self.assertEqual(first.authors, ("Alice",))
        self.assertEqual(first.record_id, "record-001-001")
        self.assertIn("Title: Paper A", first.text)
        self.assertIn("Authors: Alice", first.text)
        self.assertIn("Language: English", first.text)
        self.assertNotIn("Paper B", first.text)
        self.assertEqual(first.to_dict()["authors"], ["Alice"])

    def test_embedder_and_labeler_do_not_duplicate_structured_title(self):
        document = {
            "id": "record-1",
            "title": "Paper A",
            "text": "Record Type: publication\nTitle: Paper A\nAuthors: Alice",
            "record_type": "publication",
            "record_id": "record-1",
            "authors": ["Alice"],
        }
        embedder = object.__new__(Embedder)
        embedder.no_title = False
        embedder.text_lower_case = False
        embedder.text_normalize = False

        embedded_text = embedder.process_text(document)
        batch = LabelerInputBuilder().build_batch(
            question="Who wrote Paper A?",
            current_query="Paper A author",
            documents=[document],
        )

        self.assertEqual(embedded_text.count("Paper A"), 1)
        self.assertEqual(batch.texts[0].count("Paper A"), 1)
        self.assertEqual(batch.documents[0].diagnostics["record_type"], "publication")
        self.assertEqual(
            batch.documents[0].diagnostics["record_fields"]["authors"],
            ["Alice"],
        )

    def test_linked_content_enrichment_repeats_record_identity(self):
        def fetcher(url: str, *, max_tokens: int):
            self.assertEqual(url, "https://example.org/paper-a")
            self.assertEqual(max_tokens, 1200)
            return PageFetchResult(
                content=(
                    "The detailed paper explains the target relationship and gives "
                    "the factual answer needed by the downstream agent."
                ),
                method="test",
            )

        builder = WebCorpusBuilder(
            chunker=DocumentChunker(max_chars=300, overlap_chars=0, min_chars=10),
            page_fetcher=fetcher,
        )
        record = CorpusRecord(
            id="record-001-001-001",
            title="Paper A",
            text="Record Type: publication\nTitle: Paper A\nAuthors: Alice",
            url="https://example.org/paper-a",
            retrieved_at="2026-07-17",
            record_type="publication",
            record_id="record-001-001",
            authors=("Alice",),
            date="2024",
            content_url="https://example.org/paper-a",
            parent_url="https://example.org/list",
            extraction_method="html_table",
        )

        enriched = builder.build_enriched_records(
            record,
            max_tokens=1200,
        )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].record_id, "record-001-001")
        self.assertEqual(enriched[0].authors, ("Alice",))
        self.assertIn("Title: Paper A", enriched[0].text)
        self.assertIn("Authors: Alice", enriched[0].text)
        self.assertIn("detailed paper", enriched[0].text)
        self.assertIn("linked_content", enriched[0].extraction_method)

    def test_structured_records_are_not_limited_as_same_domain_or_neighbor_expanded(self):
        passages = {
            f"record-{index:03d}-001": {
                "id": f"record-{index:03d}-001",
                "title": f"Paper {index}",
                "text": f"Title: Paper {index} Authors: Author {index}",
                "url": f"https://example.org/paper/{index}",
                "record_type": "publication",
            }
            for index in range(1, 6)
        }
        selector = PassageCandidateSelector(max_per_domain=1, max_neighbor_items=4)
        dense = [(document_id, 0.95 - index * 0.01) for index, document_id in enumerate(passages)]

        selected = selector.select(
            passage_map=passages,
            ranked_dense_lists={"dense": dense},
            lexical_query="Paper Author",
            max_items=5,
        )

        self.assertEqual(len(selected), 5)
        self.assertTrue(all(not item.expanded_from for item in selected))


if __name__ == "__main__":
    unittest.main()
