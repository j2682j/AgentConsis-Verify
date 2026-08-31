"""Whether the agent saw the prepared evidence has to be readable from the record.

It was not. `tool_context` holds a runtime tool trace on the tool-use path and
formatted evidence on the other, so the same field means two things, and a run
whose trace read `"None"` was taken for a run that answered with no context at
all. Its prompt was six thousand characters and had dropped nothing.

The fix is not a better guess. It is measuring the search evidence on both sides
of the budget, saying outright what `tool_context` contains, and keeping the
per-turn numbers instead of summing them into a total that answers nothing.

`dropped_evidence_count` stays in the record and stays unusable as proof: a
reference block trimmed at the tail does not increment it, by design and by an
existing test. Loss is visible in the rendered chars and the truncation flag, so
those are what these tests pin down.
"""

from __future__ import annotations

import unittest

from context.context_budget import ContextBudget, ContextBudgetManager


class BudgetTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ContextBudgetManager()

    def apply(self, search_result: str) -> dict:
        return self.manager.apply(
            {"question": "q", "search_result": search_result, "solver_result": "s"}
        ).diagnostics.to_dict()

    def test_untouched_evidence_hashes_the_same_on_both_sides(self) -> None:
        diagnostics = self.apply("a short piece of evidence")

        self.assertEqual(
            diagnostics["prepared_search_context_hash"],
            diagnostics["rendered_search_context_hash"],
        )
        self.assertEqual(
            diagnostics["prepared_search_context_chars"],
            diagnostics["rendered_search_context_chars"],
        )
        self.assertFalse(diagnostics["search_result_truncated"])

    def test_truncated_evidence_shrinks_and_changes_hash(self) -> None:
        diagnostics = self.apply("E" * 9000)

        self.assertLess(
            diagnostics["rendered_search_context_chars"],
            diagnostics["prepared_search_context_chars"],
        )
        self.assertNotEqual(
            diagnostics["prepared_search_context_hash"],
            diagnostics["rendered_search_context_hash"],
        )
        self.assertTrue(diagnostics["search_result_truncated"])

    def test_absent_evidence_reports_zero_rather_than_a_hash_of_nothing(self) -> None:
        diagnostics = self.apply("")

        self.assertEqual(diagnostics["prepared_search_context_chars"], 0)
        self.assertEqual(diagnostics["prepared_search_context_hash"], "")
        self.assertEqual(diagnostics["rendered_search_context_hash"], "")
        self.assertFalse(diagnostics["search_result_truncated"])

    def test_a_section_missing_entirely_does_not_fabricate_telemetry(self) -> None:
        diagnostics = self.manager.apply({"question": "q"}).diagnostics.to_dict()

        self.assertEqual(diagnostics["prepared_search_context_chars"], 0)
        self.assertFalse(diagnostics["search_result_truncated"])

    def test_dropped_evidence_count_is_not_a_loss_signal(self) -> None:
        """References trimmed at the tail leave it at zero, by existing design.

        Pinned here because a zero was read as proof that nothing was lost, and
        the numbers that actually carry that fact are the rendered ones.
        """

        # Shaped like the real thing: the compactor recognises `[R#]` blocks and
        # trims their tails, which is the loss `dropped_evidence_count` is
        # documented not to count.
        body = "reference body " + "x" * 600
        lines = ["Grounded Evidence:", "None", "", "Unverified References:",
                 "These retrieved passages are NOT verified answer support."]
        for index in range(1, 9):
            lines += [f"[R{index}]", f"Source Title: Source {index}", f"Content: {body}"]
        diagnostics = self.apply("\n".join(lines))

        self.assertEqual(diagnostics["dropped_evidence_count"], 0)
        self.assertTrue(diagnostics["search_result_truncated"])
        self.assertLess(
            diagnostics["rendered_search_context_chars"],
            diagnostics["prepared_search_context_chars"],
        )


class ToolUseSummaryTest(unittest.TestCase):
    """The per-turn numbers must survive the fold into the run record."""

    def summarise(self, turns: list[dict]) -> dict:
        from core.stage1_tool_use_runner import Stage1ToolUseRunner

        runner = Stage1ToolUseRunner.__new__(Stage1ToolUseRunner)
        return runner._context_budget_summary(
            [{"context_budget": turn} for turn in turns]
        )

    def turn(self, prepared: int, rendered: int, tag: str) -> dict:
        return {
            "original_chars": prepared + 10,
            "final_chars": rendered + 10,
            "truncation_applied": prepared != rendered,
            "dropped_evidence_count": 0,
            "truncated_sections": ["search_result"] if prepared != rendered else [],
            "section_chars": {"search_result": rendered},
            "prepared_search_context_chars": prepared,
            "prepared_search_context_hash": f"prep{tag}",
            "rendered_search_context_chars": rendered,
            "rendered_search_context_hash": f"rend{tag}",
            "search_result_truncated": prepared != rendered,
        }

    def test_every_turn_is_kept(self) -> None:
        summary = self.summarise([self.turn(900, 900, "a"), self.turn(900, 400, "b")])

        self.assertEqual(len(summary["turns"]), 2)
        self.assertEqual(summary["turns"][0]["prepared_search_context_hash"], "prepa")
        self.assertEqual(summary["turns"][1]["rendered_search_context_hash"], "rendb")

    def test_the_first_turn_is_reachable_without_indexing(self) -> None:
        summary = self.summarise([self.turn(900, 900, "a"), self.turn(700, 300, "b")])

        self.assertEqual(summary["first_turn_prepared_search_context_chars"], 900)
        self.assertEqual(summary["first_turn_rendered_search_context_hash"], "renda")

    def test_truncated_turns_are_counted_not_merged(self) -> None:
        summary = self.summarise(
            [self.turn(900, 900, "a"), self.turn(900, 400, "b"), self.turn(900, 100, "c")]
        )

        self.assertEqual(summary["search_result_truncated_turn_count"], 2)

    def test_hashes_are_never_combined_across_turns(self) -> None:
        """A hash of two turns identifies nothing a replay could check."""

        summary = self.summarise([self.turn(900, 900, "a"), self.turn(900, 400, "b")])

        self.assertNotIn("prepared_search_context_hash", summary)
        self.assertNotIn("rendered_search_context_hash", summary)

    def test_a_trajectory_without_budgets_summarises_to_nothing(self) -> None:
        self.assertEqual(self.summarise([]), {})


class ContextSourceTest(unittest.TestCase):
    """`tool_context` alone cannot say which of its two meanings applies."""

    def test_the_field_defaults_to_declaring_itself_unknown(self) -> None:
        from core.config import EachAgentReply

        reply = EachAgentReply(
            agent_id="a", model_name="m", run_index=1, raw_reply="", reasoning="",
            final_answer="", parse_completed=True, tool_context="None",
        )

        self.assertEqual(reply.context_source, "unspecified")
        self.assertEqual(reply.runtime_tool_trace_chars, 0)

    def test_a_trace_of_none_is_not_evidence_of_an_empty_prompt(self) -> None:
        """The reading that started this: four characters taken for no context."""

        from core.config import EachAgentReply

        reply = EachAgentReply(
            agent_id="a", model_name="m", run_index=1, raw_reply="", reasoning="",
            final_answer="", parse_completed=True, tool_context="None",
            context_source="runtime_tool_trace", runtime_tool_trace_chars=4,
            context_budget={"prepared_search_context_chars": 3072,
                            "rendered_search_context_chars": 3072,
                            "final_chars": 6004},
        )

        self.assertEqual(reply.context_source, "runtime_tool_trace")
        self.assertEqual(reply.context_budget["prepared_search_context_chars"], 3072)

    def test_the_non_tool_path_declares_formatted_evidence(self) -> None:
        from core.config import EachAgentReply

        reply = EachAgentReply(
            agent_id="a", model_name="m", run_index=1, raw_reply="", reasoning="",
            final_answer="", parse_completed=True, tool_context="Evidence: ...",
            context_source="formatted_evidence",
        )

        self.assertEqual(reply.context_source, "formatted_evidence")


if __name__ == "__main__":
    unittest.main()
