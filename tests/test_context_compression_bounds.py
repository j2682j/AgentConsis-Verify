"""Pin which bound decides how much prepared evidence an Agent sees.

`_compress_multiline_text` cuts lines first and characters second, so the line
bound decides the outcome and the character bound only trims what survives it.
A retrieval record runs 6 lines and 443 characters at the median and retrieval
hands over 16 of them: 96 lines against a bound of 80, but only 7,088 characters
against a bound of 12,000. Anyone retuning these has to move the line bound --
moving the character bound alone changes nothing.

The bounds were raised to 240/24000 for level1_final_09 and reverted. Delivering
more evidence made the 4B Agents worse: grouping that run's 477 Agent runs by
how much their context grew against level1_final_08 gives 30.6%->29.8% correct
where it barely moved, 16.4%->11.9% where it grew up to 2x, and 29.5%->15.9%
where it more than doubled. So these tests hold the ordering, which is a
property of the code, and record the tuning result rather than a target.
"""

from __future__ import annotations

import unittest

from context.context_builder import ContextBuilder, ContextConfig

MEDIAN_LINES_PER_RECORD = 6
RETRIEVED_RECORDS = 16


def _record(index: int) -> str:
    """A retrieval record at its median shape: 6 lines, about 443 characters."""

    return "\n".join([
        "Record Type: article",
        f"Title: Result {index}",
        f"Source: Example Source {index}",
        f"Content Link: https://example.org/{index}",
        "Visual Labels: img",
        f"Content: {'x' * 300}",
    ])


def _evidence(count: int = RETRIEVED_RECORDS) -> str:
    return "\n".join(_record(index) for index in range(count))


class CompressionOrderingTest(unittest.TestCase):
    """The ordering is a property of the code, independent of the tuning."""

    def setUp(self) -> None:
        self.builder = ContextBuilder()

    def _compress(self, text: str, **kwargs) -> str:
        return self.builder._compress_multiline_text(text, **kwargs)

    def test_line_bound_is_applied_before_the_character_bound(self) -> None:
        text = "\n".join(f"line {index}" for index in range(200))

        compressed = self._compress(text, max_lines=10, max_chars=10_000)

        self.assertEqual(len(compressed.splitlines()), 10)

    def test_a_generous_character_bound_cannot_rescue_a_tight_line_bound(self) -> None:
        """Why raising max_context_chars on its own is a no-op."""

        compressed = self._compress(_evidence(), max_lines=80, max_chars=1_000_000)

        self.assertEqual(len(compressed.splitlines()), 80)
        self.assertLess(compressed.count("Record Type:"), RETRIEVED_RECORDS)

    def test_character_bound_still_trims_what_survives_the_line_bound(self) -> None:
        text = "\n".join("y" * 500 for _ in range(100))

        compressed = self._compress(text, max_lines=100, max_chars=1_000)

        self.assertLessEqual(len(compressed), 1_010)
        self.assertTrue(compressed.endswith("..."))


class DefaultBoundsTest(unittest.TestCase):
    """Record the tuning outcome so the raise is not repeated by accident."""

    def test_defaults_are_the_reverted_pair(self) -> None:
        config = ContextConfig()

        self.assertEqual(config.max_context_lines, 80)
        self.assertEqual(config.max_context_chars, 12_000)

    def test_defaults_deliberately_cut_a_full_retrieval_batch(self) -> None:
        """Not an oversight: 240/24000 was measured and performed worse."""

        compressed = ContextBuilder()._compress_multiline_text(_evidence())

        self.assertLess(compressed.count("Record Type:"), RETRIEVED_RECORDS)
        self.assertEqual(len(compressed.splitlines()), 80)

    def test_the_line_bound_is_what_binds_at_these_defaults(self) -> None:
        config = ContextConfig()
        evidence = _evidence()

        line_limited = "\n".join(
            line for line in evidence.splitlines() if line.strip()
        ).splitlines()[: config.max_context_lines]

        self.assertLess(
            len("\n".join(line_limited)),
            config.max_context_chars,
            "characters are under the bound, so only the line bound can bite",
        )


if __name__ == "__main__":
    unittest.main()
