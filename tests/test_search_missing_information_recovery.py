"""Pin that a misplaced `missing_information` costs no tool turn.

The Stage1 search gate blocks a refinement request whose tool_args omit
`missing_information`, and the block consumes one of the run's few tool turns.
The agent has usually already said what it needs, just in `reasoning_step`:
across level1_final_06, _07 and _08 every one of the 142 requests blocked this
way carried one, reading like "step 1. I need to find the minimum perigee
distance between Earth and the Moon".

So the field is misplaced rather than missing, and normalisation reads it across.
The gate's requirement is untouched -- a request that says nothing about what it
needs still recovers nothing and still blocks.
"""

from __future__ import annotations

import unittest

from core.stage1_tool_use_runner import Stage1ToolUseRunner

derive = Stage1ToolUseRunner._missing_information_from_reasoning


class MissingInformationRecoveryTest(unittest.TestCase):
    def test_step_prefix_and_intent_phrase_are_stripped(self) -> None:
        self.assertEqual(
            derive("step 1. I need to find the minimum perigee distance"),
            "find the minimum perigee distance",
        )

    def test_capitalised_and_alternate_phrasings_are_handled(self) -> None:
        for text in (
            "Step 1. I need to find the album count",
            "step 1: I must find the album count",
            "step 1) Need to find the album count",
        ):
            self.assertEqual(derive(text), "find the album count", text)

    def test_a_step_with_no_content_recovers_nothing(self) -> None:
        """The gate must still block a request that states no need."""

        for text in ("", "   ", "step 3.", "step 12:"):
            self.assertEqual(derive(text), "", repr(text))

    def test_a_long_step_is_capped(self) -> None:
        derived = derive(
            "step 1. I need to find the perigee distance " + "and the apogee too " * 40
        )

        self.assertLessEqual(len(derived), 200)
        self.assertTrue(derived)

    def test_a_step_that_only_restates_the_need_recovers_nothing(self) -> None:
        """Otherwise the gate's requirement becomes vacuous.

        The prompt asks for a reasoning_step on every tool request, so reading
        any step across would let every request satisfy the gate.
        """

        for text in (
            "step 1. Obtain the one missing fact.",
            "step 1. why this tool is needed",
            "step 2. I need to find the missing information",
            "step 1. get the required value",
        ):
            self.assertEqual(derive(text), "", repr(text))

    def test_text_without_a_step_prefix_is_kept(self) -> None:
        self.assertEqual(
            derive("the release year of the second album"),
            "the release year of the second album",
        )


class NormalizeToolArgsTest(unittest.TestCase):
    def _normalize(self, args: dict, reasoning_step: str = "") -> dict:
        runner = Stage1ToolUseRunner.__new__(Stage1ToolUseRunner)
        return runner._normalize_tool_args(
            "search", args, reasoning_step=reasoning_step
        )

    def test_missing_field_is_filled_from_the_reasoning_step(self) -> None:
        args = self._normalize(
            {"input": "perigee distance moon"},
            reasoning_step="step 1. I need to find the minimum perigee distance",
        )

        self.assertEqual(
            args["missing_information"], "find the minimum perigee distance"
        )

    def test_an_explicit_value_is_never_overwritten(self) -> None:
        args = self._normalize(
            {"input": "q", "missing_information": "the stated need"},
            reasoning_step="step 1. I need to find something else",
        )

        self.assertEqual(args["missing_information"], "the stated need")

    def test_a_blank_value_is_treated_as_absent(self) -> None:
        args = self._normalize(
            {"input": "q", "missing_information": "   "},
            reasoning_step="step 1. I need to find the album count",
        )

        self.assertEqual(args["missing_information"], "find the album count")

    def test_nothing_is_invented_without_a_reasoning_step(self) -> None:
        args = self._normalize({"input": "q"})

        self.assertNotIn("missing_information", args)

    def test_other_search_defaults_still_apply(self) -> None:
        args = self._normalize({"query": "q"}, reasoning_step="step 1. I need to find x")

        self.assertEqual(args["input"], "q")
        self.assertEqual(args["mode"], "text")


if __name__ == "__main__":
    unittest.main()
