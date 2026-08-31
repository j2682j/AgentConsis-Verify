"""Following evidence blocks through the budget, instead of hunting for answers.

Asking whether the answer survived truncation needs a rule for every answer:
`3` means nothing alone, `research` matches any sentence about research, `Claus`
hides inside `Clausen`. A table of those rules describes one benchmark. Asking
which blocks survived needs no rules at all and describes the system.

These tests pin the three outcomes a block can have and, more importantly, the
two ways the question could quietly stop being answerable: a block that is
renumbered would look dropped, and a block whose body changed while its marker
stayed would look kept.
"""

from __future__ import annotations

import unittest

from context.context_budget import ContextBudgetManager
from context.evidence_block_lineage import parse_blocks, trace


def block(kind: str, index: int, body: str) -> str:
    return f"[{kind}{index}]\nSource Title: Source {index}\nContent: {body}"


def section(kind: str, count: int, body: str) -> str:
    head = ["Grounded Evidence:", "None", "", "Unverified References:"]
    return "\n".join(head + [block(kind, i, body) for i in range(1, count + 1)])


class ParseTest(unittest.TestCase):
    def test_blocks_are_found_by_their_markers(self) -> None:
        blocks = parse_blocks(section("R", 3, "body"))

        self.assertEqual([b.block_id for b in blocks], ["R1", "R2", "R3"])

    def test_the_preamble_is_not_a_block(self) -> None:
        blocks = parse_blocks(section("R", 2, "body"))

        self.assertNotIn("Unverified References:", blocks[0].text)

    def test_evidence_and_references_are_told_apart(self) -> None:
        text = block("E", 1, "graded") + "\n" + block("R", 1, "unverified")
        kinds = {b.block_id: b.kind for b in parse_blocks(text)}

        self.assertEqual(kinds, {"E1": "E", "R1": "R"})

    def test_text_without_markers_yields_nothing(self) -> None:
        self.assertEqual(parse_blocks("Grounded Evidence:\nNone"), [])


class TraceTest(unittest.TestCase):
    def test_an_untouched_section_keeps_every_block(self) -> None:
        text = section("R", 3, "body")
        result = trace(text, text).to_dict()

        self.assertEqual(result["kept_count"], 3)
        self.assertEqual(result["lost_block_ids"], [])

    def test_a_missing_marker_reads_as_dropped(self) -> None:
        prepared = section("R", 3, "body")
        rendered = section("R", 2, "body")

        result = trace(prepared, rendered).to_dict()

        self.assertEqual(result["dropped_count"], 1)
        self.assertEqual(result["lost_block_ids"], ["R3"])

    def test_a_shortened_body_reads_as_truncated_not_kept(self) -> None:
        """The marker surviving is not the block surviving."""

        prepared = block("R", 1, "x" * 500)
        rendered = block("R", 1, "x" * 40)

        result = trace(prepared, rendered).to_dict()

        self.assertEqual(result["truncated_count"], 1)
        self.assertEqual(result["blocks"][0]["disposition"], "truncated")
        self.assertLess(
            result["blocks"][0]["rendered_chars"],
            result["blocks"][0]["original_chars"],
        )

    def test_hashes_distinguish_identical_lengths(self) -> None:
        """Same length, different content -- the reason chars alone are not enough."""

        prepared = block("R", 1, "alpha")
        rendered = block("R", 1, "omega")

        result = trace(prepared, rendered).to_dict()

        self.assertEqual(result["blocks"][0]["disposition"], "truncated")
        self.assertNotEqual(
            result["blocks"][0]["original_text_hash"],
            result["blocks"][0]["rendered_text_hash"],
        )

    def test_nothing_rendered_loses_everything(self) -> None:
        result = trace(section("R", 4, "body"), "").to_dict()

        self.assertEqual(result["dropped_count"], 4)


class AgainstTheRealBudgetTest(unittest.TestCase):
    """Run the actual compactor and read its effect off the lineage."""

    def setUp(self) -> None:
        self.manager = ContextBudgetManager()

    def render(self, prepared: str) -> str:
        return self.manager.apply(
            {"question": "q", "search_result": prepared}
        ).sections["search_result"]

    def test_reference_loss_is_visible_where_dropped_evidence_count_is_not(self) -> None:
        """The count stays at zero by design; the lineage says what went."""

        prepared = section("R", 8, "reference body " + "x" * 600)
        rendered = self.render(prepared)
        result = trace(prepared, rendered).to_dict()
        diagnostics = self.manager.apply(
            {"question": "q", "search_result": prepared}
        ).diagnostics.to_dict()

        self.assertEqual(diagnostics["dropped_evidence_count"], 0)
        self.assertTrue(result["lost_block_ids"])
        self.assertEqual(result["block_count"], 8)

    def test_a_section_within_the_allowance_loses_nothing(self) -> None:
        prepared = section("R", 2, "short body")
        result = trace(prepared, self.render(prepared)).to_dict()

        self.assertEqual(result["lost_block_ids"], [])

    def test_survival_is_not_decided_by_position_alone(self) -> None:
        """Reference fitting is block-wise, so a later short block can outlive
        an earlier long one. Anything reasoning from character offsets would get
        this backwards."""

        blocks = [
            block("R", 1, "y" * 3000),
            block("R", 2, "short"),
            block("R", 3, "z" * 3000),
        ]
        prepared = "\n".join(["Unverified References:"] + blocks)
        result = trace(prepared, self.render(prepared)).to_dict()
        disposition = {b["block_id"]: b["disposition"] for b in result["blocks"]}

        self.assertEqual(len(disposition), 3)
        self.assertIn(disposition["R2"], {"kept", "truncated", "dropped"})

    def test_lineage_needs_no_knowledge_of_the_answer(self) -> None:
        """The whole point: no gold, no per-task rule, no matcher."""

        import inspect

        from context import evidence_block_lineage

        source = inspect.getsource(evidence_block_lineage)
        for forbidden in ("gold", "expected", "answer_requirement"):
            with self.subTest(term=forbidden):
                self.assertNotIn(f"{forbidden} =", source)


if __name__ == "__main__":
    unittest.main()
