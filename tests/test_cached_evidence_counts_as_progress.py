"""Pin that content new to a run counts as progress however it was obtained.

`already_available` marks a tool result the system had cached or shared, not one
this run has already read, and the payload comes attached: search results of
1000-2200 characters, attachment text of 1400-2700. Rejecting it on the status
alone counted 124 such turns across level1_final_06 to _08 as no progress while
the Agent was reading them for the first time, across 111 runs.

That matters twice over. Two no-progress turns in a row end tool use and force
the repair prompt, and a progress turn is also what extends the turn budget --
so runs that should have earned more turns instead spent five on one tool and
hit the hard limit. gemma took that path on 71 of the 72 runs that hit it.

The fingerprint check still rejects content the run has already seen, so genuine
repetition is unaffected, and the statuses that carry no payload at all stay out.
"""

from __future__ import annotations

import unittest

from core.tool_turn_policy import AdaptiveToolTurnPolicy


def _result(text: str, *, status: str = "success", valid: bool = True) -> dict:
    return {
        "ok": True,
        "status": status,
        "output_text": text,
        "raw_result": {"results": [{"title": text}]},
        "evidence_valid": valid,
    }


class CachedEvidenceCountsAsProgressTest(unittest.TestCase):
    def test_cached_content_new_to_this_run_is_progress(self) -> None:
        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)

        self.assertTrue(
            policy.observe(_result("lunar perigee 356500 km", status="already_available"))
        )
        self.assertEqual(policy.no_progress_streak, 0)

    def test_cached_content_extends_the_turn_budget(self) -> None:
        """The budget extension is why this mattered beyond the streak."""

        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)

        policy.observe(_result("first page", status="already_available"))

        self.assertEqual(policy.allowed_budget, 3)

    def test_repeating_cached_content_is_still_no_progress(self) -> None:
        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)
        repeated = _result("same page", status="already_available")

        self.assertTrue(policy.observe(repeated))
        self.assertFalse(policy.observe(repeated))

    def test_two_repeats_still_end_tool_use(self) -> None:
        policy = AdaptiveToolTurnPolicy(base_budget=4, hard_limit=8, no_progress_limit=2)
        repeated = _result("same page", status="already_available")

        policy.observe(repeated)
        policy.observe(repeated)
        policy.observe(repeated)

        self.assertTrue(policy.force_final)
        self.assertEqual(policy.stop_reason, "consecutive_no_progress")

    def test_statuses_with_no_payload_are_still_rejected(self) -> None:
        for status in ("duplicate_blocked", "unsupported", "fatal"):
            policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)

            self.assertFalse(
                policy.observe(_result("anything", status=status)), status
            )

    def test_invalid_evidence_is_still_rejected(self) -> None:
        policy = AdaptiveToolTurnPolicy(base_budget=2, hard_limit=4)

        self.assertFalse(
            policy.observe(
                _result("text", status="already_available", valid=False)
            )
        )


if __name__ == "__main__":
    unittest.main()
