from __future__ import annotations

import unittest

from tools.search_result_builder.evidence.span_builder import SpanBuilder


class SpanBuilderTests(unittest.TestCase):
    def test_build_context_maps_terms_back_to_original_sentence(self):
        builder = SpanBuilder(max_context_chars=360)
        text = (
            "Mercedes Sosa recorded many albums. "
            "Sosa won the Latin Grammy Award for Best Folk Album in 2000 "
            "(Misa Criolla), 2003 (Acústico), and 2006 (Corazón Libre). "
            "An unrelated sentence follows."
        )

        context, spans = builder.build_context(text, ["misa criolla", "2000"])

        self.assertIn("Misa Criolla", context)
        self.assertIn("2000", context)
        self.assertIn("Best Folk Album", context)
        self.assertEqual(len(spans), 1)
        self.assertIn("misa criolla", spans[0].term.lower())

    def test_build_context_matches_accent_insensitive_terms(self):
        builder = SpanBuilder(max_context_chars=240)
        text = "The album Acústico was listed in 2003 by the source."

        context, spans = builder.build_context(text, ["acustico"])

        self.assertIn("Acústico", context)
        self.assertEqual(len(spans), 1)

    def test_build_context_falls_back_when_terms_are_not_found(self):
        builder = SpanBuilder(max_context_chars=120)
        text = "This chunk has useful information but no matching token."

        context, spans = builder.build_context(
            text,
            ["missing term"],
            fallback_chars=25,
        )

        self.assertEqual(spans, [])
        self.assertEqual(context, "This chunk has useful inf...")

    def test_build_context_merges_nearby_sentence_spans(self):
        builder = SpanBuilder(max_context_chars=500, merge_distance_chars=160)
        text = (
            "First clue mentions Pietro Murano. "
            "Second clue mentions Iram Khan. "
            "The remaining sentence is unrelated."
        )

        context, spans = builder.build_context(text, ["Pietro Murano", "Iram Khan"])

        self.assertEqual(len(spans), 1)
        self.assertIn("Pietro Murano", context)
        self.assertIn("Iram Khan", context)

    def test_build_context_ignores_weak_anchor_terms(self):
        builder = SpanBuilder(max_context_chars=120)
        text = (
            "Formula reference 1 is not enough by itself. "
            "The answer appears later as 0.1777 m3."
        )

        context, spans = builder.build_context(text, ["1"], fallback_chars=45)

        self.assertEqual(spans, [])
        self.assertEqual(context, "Formula reference 1 is not enough by itself....")


if __name__ == "__main__":
    unittest.main()
