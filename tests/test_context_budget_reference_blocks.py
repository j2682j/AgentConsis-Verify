"""Record why the `[R#]` reference block takes a plain cut, not a block-aware one.

Stage1 gets a reference-shaped search context whenever no grounded evidence was
found, which on recent runs is every task with retrieval. `_evidence_blocks`
only recognises `[E#]`, so that shape falls through to `_truncate` and the
allowance is spent front-to-back: about 4 references survive at roughly 430
characters each, and whatever follows is dropped outright.

That looks like a defect, and it was fixed for level1_final_12 by splitting the
same allowance across every entry so each kept its id, title and a share of its
body. It cost four tasks:

    with references      21% (5/24) -> 4% (1/27)
    without references   48% (14/29) -> 50% (13/26)
    run-level accuracy   27.5% -> 28.8%

The Agents got *better* while the score fell, so what broke was selection, and
the damage was confined to exactly the tasks the change touched. The mechanism
is the shape it produces: the same allowance becomes 8 references of about 150
characters instead of 4 of about 430, and the 4B Agents do worse with the
shallow spread. An offline sweep run before the change had already measured
this -- 4 deep references put the answer in the prompt on 8 of 12 tasks against
5-6 for 8 shallow ones -- and it was not connected to what the change would
produce.

So these tests hold the plain cut and the shape it yields. The depth assertion
is the one that matters: it is what a future block-aware rewrite would break.
"""

from __future__ import annotations

import re
import statistics
import unittest

from context.context_budget import ContextBudget, ContextBudgetManager

_BODY = "retrieved passage sentence " * 30


def _references(count: int) -> str:
    lines = [
        "Grounded Evidence:",
        "None",
        "",
        "Unverified References:",
        "These retrieved passages are NOT verified answer support.",
    ]
    for index in range(1, count + 1):
        lines.extend([f"[R{index}]", f"Source Title: Source {index}", f"Content: {_BODY}"])
    return "\n".join(lines)


def _kept(text: str) -> tuple[int, int]:
    """How many references survived, and the median body length."""

    bodies = [len(body) for body in re.findall(r"(?m)^Content: (.*)$", text)]
    return len(re.findall(r"(?m)^\[R\d+\]", text)), (
        round(statistics.median(bodies)) if bodies else 0
    )


class ReferenceShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ContextBudgetManager(
            ContextBudget(max_total_chars=100_000, max_search_evidence_items=5, max_search_evidence_chars=450)
        )

    def _compact(self, text: str) -> str:
        compacted, _dropped = self.manager._compact_search_evidence(text)
        return compacted

    def test_references_are_kept_deep_rather_than_spread_thin(self) -> None:
        """The tuning result: few complete references beat many partial ones."""

        count, median_body = _kept(self._compact(_references(8)))

        self.assertLessEqual(count, 5, "spreading the allowance over all 8 cost four tasks")
        self.assertGreater(median_body, 300, "a kept reference has to carry usable content")

    def test_the_allowance_is_what_binds(self) -> None:
        source = _references(8)
        compacted = self._compact(source)

        self.assertLessEqual(len(compacted), self.manager._search_evidence_budget() + len(" ..."))
        # Content is dropped to reach the allowance. This used to assert the
        # trailing "..." as well, which pinned the mechanism rather than the
        # property: the section now ends on the last complete `[R#]` block
        # instead of a character offset, so the marker is usually absent. See
        # tests/test_reference_block_boundary.py.
        self.assertLess(len(compacted), len(source.strip()))

    def test_a_block_under_the_allowance_is_untouched(self) -> None:
        text = "\n".join(_references(1).splitlines()[:5])

        self.assertEqual(self._compact(text), text.strip())

    def test_dropping_the_tail_is_not_counted_as_dropped_evidence(self) -> None:
        """`dropped_evidence_count` tracks `[E#]` items; references are not that."""

        _compacted, dropped = self.manager._compact_search_evidence(_references(8))

        self.assertEqual(dropped, 0)


class EvidencePathTest(unittest.TestCase):
    """`[E#]` blocks keep their own item cap and per-item trim."""

    def setUp(self) -> None:
        self.manager = ContextBudgetManager(
            ContextBudget(max_total_chars=100_000, max_search_evidence_items=5, max_search_evidence_chars=120)
        )

    def test_evidence_blocks_drop_past_the_item_cap(self) -> None:
        lines = ["Evidence:"]
        for index in range(1, 8):
            lines.extend([f"[E{index}]", f"Source Title: S{index}", "Evidence: " + ("word " * 80)])

        compacted, dropped = self.manager._compact_search_evidence("\n".join(lines))

        self.assertIn("[E5]", compacted)
        self.assertNotIn("[E6]", compacted)
        self.assertEqual(dropped, 2)

    def test_unmarked_text_takes_the_plain_cut(self) -> None:
        text = "plain retrieved prose without any markers. " * 100

        compacted, dropped = self.manager._compact_search_evidence(text)

        self.assertEqual(dropped, 0)
        self.assertLessEqual(len(compacted), self.manager._search_evidence_budget() + len(" ..."))


if __name__ == "__main__":
    unittest.main()
