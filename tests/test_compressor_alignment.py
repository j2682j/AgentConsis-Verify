"""Aligning compressed evidence to its original, in the compressor's coordinates.

A probe that compared whole lines reported a 68,000-character block as
`transformed_or_unalignable` and produced a confident conclusion: long evidence
is rewritten before the budget sees it. The compressor does no such thing. It
strips lines, drops blank ones, keeps the first `max_lines`, and cuts to
`max_chars` with a trailing `" ..."`. The single line the character cut landed
inside stopped equalling its original, and whole-line equality called that
unalignable.

So these tests pin the shapes the compressor can actually produce, and the four
edge cases where an offset map is easy to get quietly wrong: indentation that
shifts everything after it, blank lines that vanish, CRLF, and text that ended
in `...` before the compressor ever touched it.
"""

from __future__ import annotations

import unittest

from context.compressor_alignment import align, child_survival, normalise


def compress(text: str, max_lines: int = 80, max_chars: int = 12000) -> str:
    """The production rule, restated so the tests do not depend on a builder."""

    raw = str(text or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    compressed = "\n".join(lines[:max_lines]).strip()
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars].rstrip() + " ..."
    return compressed


class NormaliseTest(unittest.TestCase):
    def test_indentation_is_removed_and_the_trail_still_points_home(self) -> None:
        raw = "alpha\n    beta\n"
        result = normalise(raw)

        self.assertEqual(result.text, "alpha\nbeta")
        self.assertEqual(raw[result.origin[result.text.index("beta")]], "b")

    def test_blank_lines_vanish_without_breaking_the_mapping(self) -> None:
        raw = "one\n\n\ntwo"
        result = normalise(raw)

        self.assertEqual(result.text, "one\ntwo")
        self.assertEqual(raw[result.origin[result.text.index("two")]], "t")

    def test_crlf_is_handled_like_lf(self) -> None:
        self.assertEqual(normalise("a\r\nb").text, normalise("a\nb").text)

    def test_every_character_maps_inside_the_original(self) -> None:
        raw = "  head  \n\n    body line\n\ttabbed\n"
        result = normalise(raw)

        for index, character in enumerate(result.text):
            if character == "\n":
                continue
            with self.subTest(index=index):
                self.assertEqual(raw[result.origin[index]], character)


class AlignTest(unittest.TestCase):
    def test_an_untouched_block_aligns_exactly(self) -> None:
        raw = "line one\nline two"
        result = align(raw, compress(raw), max_lines=80, max_chars=12000)

        self.assertEqual(result["shape"], "exact_after_line_normalization")
        self.assertFalse(result["truncation_marker_added"])

    def test_stripped_indentation_alone_is_still_exact(self) -> None:
        """The compressor removing leading spaces is not a content change."""

        raw = "    indented\n\n  also indented"
        result = align(raw, compress(raw), max_lines=80, max_chars=12000)

        self.assertEqual(result["shape"], "exact_after_line_normalization")

    def test_a_character_cut_inside_a_line_is_a_prefix_not_a_rewrite(self) -> None:
        """The case the old probe called `transformed_or_unalignable`."""

        raw = "header\n" + "x" * 40000
        compressed = compress(raw, max_chars=8000)
        result = align(raw, compressed, max_lines=80, max_chars=8000)

        self.assertEqual(result["shape"], "prefix_after_line_normalization")
        self.assertTrue(result["truncation_marker_added"])

    def test_a_line_cut_is_a_prefix_too(self) -> None:
        raw = "\n".join(f"line {i}" for i in range(200))
        result = align(raw, compress(raw, max_lines=10), max_lines=10, max_chars=12000)

        self.assertEqual(result["shape"], "prefix_after_line_normalization")

    def test_text_that_already_ended_in_an_ellipsis_is_not_mistaken_for_a_cut(self) -> None:
        raw = "a short line that ends in ..."
        result = align(raw, compress(raw), max_lines=80, max_chars=12000)

        self.assertEqual(result["shape"], "exact_after_line_normalization")

    def test_genuinely_foreign_text_is_reported_unaligned(self) -> None:
        result = align("original text", "something else entirely",
                       max_lines=80, max_chars=12000)

        self.assertEqual(result["shape"], "unaligned")


class ChildSurvivalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = "\n".join(f"paragraph {i} " + "y" * 200 for i in range(60))
        self.alignment = align(
            self.raw, compress(self.raw, max_chars=4000), max_lines=80, max_chars=4000
        )

    def test_a_span_before_the_cut_is_kept(self) -> None:
        result = child_survival(self.raw, self.alignment, [(0, 100)])

        self.assertEqual(result[0]["survival"], "kept")
        self.assertEqual(result[0]["retained_ratio"], 1.0)

    def test_a_span_after_the_cut_is_dropped(self) -> None:
        end = len(self.raw)
        result = child_survival(self.raw, self.alignment, [(end - 200, end)])

        self.assertEqual(result[0]["survival"], "dropped")
        self.assertEqual(result[0]["retained_chars"], 0)

    def test_a_span_across_the_cut_is_partial_not_rounded_to_either_side(self) -> None:
        boundary = self.alignment["kept_original_offset_end"]
        result = child_survival(self.raw, self.alignment, [(boundary - 50, boundary + 50)])

        self.assertEqual(result[0]["survival"], "partial")
        self.assertGreater(result[0]["retained_ratio"], 0.0)
        self.assertLess(result[0]["retained_ratio"], 1.0)

    def test_an_unaligned_block_yields_no_survival_claim(self) -> None:
        """Better to say nothing than to report offsets that mean nothing."""

        unaligned = align("abc", "totally different", max_lines=80, max_chars=100)
        result = child_survival("abc", unaligned, [(0, 3)])

        self.assertEqual(result[0]["survival"], "unsupported")


if __name__ == "__main__":
    unittest.main()
