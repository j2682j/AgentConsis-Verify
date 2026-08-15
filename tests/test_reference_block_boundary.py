"""Pin that the search block ends on a whole reference, not a character offset.

`_compact_search_evidence` sliced the reference section at `budget[:n]`. On
level1_final_16 that ended every one of the 28 retrieval tasks on a fragment --
89% mid-word, averaging 294 characters, 13% of the allowance -- and 14% ended on
a reference header with no content, so the Agents' last evidence read
`Hiccup would have had to carry 8 ...`.

Two details matter and are pinned here:

* strict evidence forms on 3 of 28 retrieval tasks, so the `[E#]` branch is not
  the one that mattered. The early return for "no evidence blocks" is the path
  almost every task takes, and it was slicing blindly.
* this is not the block-aware truncation reverted after level1_final_12. That
  gave every reference its head, turning 4 references of ~430 characters into 8
  of ~150, and cost four tasks. Per-reference depth is unchanged; only the end
  of the section moves.

Measured after the change: fragment endings 100% -> 11%, gold delivery unchanged
at 4/21, complete references 3.6 per task against 4.2 including a broken one.
"""

from __future__ import annotations

import unittest

from context.context_budget import ContextBudget, ContextBudgetManager

HEADER = "Unverified References:\nThese retrieved passages are NOT verified answer support."


def _reference(index: int, chars: int) -> str:
    body = ("word " * (chars // 5)).strip()
    return f"[R{index}] Source Title: Page {index} Content: {body}."


def _section(count: int, chars: int) -> str:
    return HEADER + "\n" + "\n".join(_reference(i, chars) for i in range(1, count + 1))


def _manager(items: int = 5, chars: int = 450) -> ContextBudgetManager:
    return ContextBudgetManager(
        ContextBudget(max_search_evidence_items=items, max_search_evidence_chars=chars)
    )


class ReferenceBlockBoundaryTest(unittest.TestCase):
    def test_the_block_does_not_end_mid_reference(self) -> None:
        kept, _dropped = _manager()._compact_search_evidence(_section(8, 400))

        self.assertFalse(kept.rstrip().endswith("..."))
        self.assertTrue(kept.rstrip().endswith("."))

    def test_references_that_do_not_fit_are_dropped_whole(self) -> None:
        kept, _dropped = _manager()._compact_search_evidence(_section(8, 400))

        self.assertLess(kept.count("[R"), 8)
        # Every marker present must carry its content, not just its header.
        for index in range(1, 9):
            marker = f"[R{index}]"
            if marker in kept:
                body = kept.split(marker, 1)[1]
                self.assertIn("Content:", body)
                self.assertGreater(len(body.split("Content:", 1)[1].strip()), 20)

    def test_per_reference_depth_is_unchanged(self) -> None:
        """The final_12 revert was about shallower references, not fewer."""

        kept, _dropped = _manager()._compact_search_evidence(_section(8, 400))
        bodies = [
            part.split("Content:", 1)[1]
            for part in kept.split("[R")[1:]
            if "Content:" in part
        ]

        self.assertTrue(bodies)
        self.assertTrue(all(len(body.strip()) > 300 for body in bodies))

    def test_a_single_oversized_reference_is_still_shown(self) -> None:
        """Dropping it whole would leave the reader nothing at all."""

        kept, _dropped = _manager()._compact_search_evidence(_section(1, 4000))

        self.assertIn("[R1]", kept)
        self.assertGreater(len(kept), 1000)

    def test_text_without_reference_markers_is_untouched_in_shape(self) -> None:
        plain = "Grounded Evidence:\n" + ("word " * 2000)

        kept, dropped = _manager()._compact_search_evidence(plain)

        self.assertEqual(dropped, 0)
        self.assertLessEqual(len(kept), 2250 + len(" ..."))

    def test_dropped_references_stay_out_of_the_evidence_count(self) -> None:
        """`dropped_evidence_count` means `[E#]` items; references are not that.

        Reporting reference drops there was tried and reverted: it changes what
        every existing report and analysis means, and the distinction is
        deliberate. See `test_dropping_the_tail_is_not_counted_as_dropped_evidence`.
        """

        _kept, dropped = _manager()._compact_search_evidence(_section(8, 400))

        self.assertEqual(dropped, 0)


if __name__ == "__main__":
    unittest.main()
