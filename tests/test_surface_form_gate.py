"""One answer spelled three ways must not lose the vote to itself.

Task 048 asks where specimens were deposited and adds `Just give me the city
name without abbreviations`. Its runs produced `St Petersburg` three times,
`St. Petersburg` three times and `Saint Petersburg` twice. Consensus counts
candidate keys, and the three spellings are three keys, so the gold answer
placed third with eight of its own votes split across the two spellings the
question rules out. It has failed this way in three of the four recorded runs --
final_13, final_15 and final_20.

The existing `_corpus_surface_form_is_authoritative` already reads this
directive, but only to stop corpus attestation promoting an abbreviated form.
Nothing merged the votes or preferred the full spelling.

These tests hold the merge to the narrow shape it was built with: it acts only
when the question states the directive, only on abbreviations whose expansion is
unambiguous, and only by promoting a spelling some run actually produced.
"""

from __future__ import annotations

import unittest

from core.config import CandidateEvaluation
from score.final_winner_selector import FinalWinnerSelector


def _candidate(answer: str, *, runs: int, agents: list[str]) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_key=answer.casefold(),
        answer=answer,
        eligible=True,
        supporting_run_count=runs,
        supporting_agent_ids=list(agents),
    )


def _selector(question: str) -> FinalWinnerSelector:
    return FinalWinnerSelector(question=question)


PETERSBURG = "Where were the specimens deposited? Just give me the city name without abbreviations."


class SurfaceFormGateTest(unittest.TestCase):
    def _task_048(self) -> list[CandidateEvaluation]:
        return [
            _candidate("St Petersburg", runs=3, agents=["a1"]),
            _candidate("St. Petersburg", runs=3, agents=["a1", "a2"]),
            _candidate("Saint Petersburg", runs=2, agents=["a3"]),
            _candidate("Zoological Institute, St. Petersburg", runs=1, agents=["a2"]),
        ]

    def test_the_split_spellings_merge_into_the_full_form(self) -> None:
        candidates = self._task_048()
        result = _selector(PETERSBURG)._apply_surface_form_gate(candidates, evidence={})

        survivors = [item.answer for item in result.survivors]
        self.assertIn("Saint Petersburg", survivors)
        self.assertNotIn("St Petersburg", survivors)
        self.assertNotIn("St. Petersburg", survivors)

        canonical = next(i for i in candidates if i.answer == "Saint Petersburg")
        self.assertEqual(canonical.supporting_run_count, 8)
        self.assertEqual(sorted(canonical.supporting_agent_ids), ["a1", "a2", "a3"])

    def test_a_longer_answer_is_not_absorbed(self) -> None:
        """`Zoological Institute, St. Petersburg` is a different answer."""

        candidates = self._task_048()
        result = _selector(PETERSBURG)._apply_surface_form_gate(candidates, evidence={})

        self.assertIn(
            "Zoological Institute, St. Petersburg",
            [item.answer for item in result.survivors],
        )

    def test_no_directive_means_no_merge(self) -> None:
        """Guard the blast radius: 52 of 53 tasks must be untouched."""

        candidates = self._task_048()
        result = _selector("Where were the specimens deposited?")._apply_surface_form_gate(
            candidates, evidence={}
        )

        self.assertEqual(len(result.survivors), len(candidates))
        self.assertFalse(result.metadata["applied"])
        self.assertEqual(
            [item.supporting_run_count for item in candidates], [3, 3, 2, 1]
        )

    def test_a_group_with_no_full_form_is_left_alone(self) -> None:
        """Never invent a spelling no run produced."""

        candidates = [
            _candidate("St Petersburg", runs=3, agents=["a1"]),
            _candidate("St. Petersburg", runs=3, agents=["a2"]),
        ]
        result = _selector(PETERSBURG)._apply_surface_form_gate(candidates, evidence={})

        self.assertEqual(len(result.survivors), 2)
        self.assertEqual([item.supporting_run_count for item in candidates], [3, 3])

    def test_the_directive_is_read_from_the_requirement_too(self) -> None:
        selector = _selector("Where were the specimens deposited?")
        evidence = {"answer_requirement": "Give the full name, do not abbreviate."}

        result = selector._apply_surface_form_gate(self._task_048(), evidence=evidence)

        self.assertTrue(result.metadata["applied"])
        self.assertIn("Saint Petersburg", [item.answer for item in result.survivors])

    def test_unrelated_answers_never_merge(self) -> None:
        candidates = [
            _candidate("Mount Everest", runs=2, agents=["a1"]),
            _candidate("Mount Fuji", runs=3, agents=["a2"]),
        ]
        result = _selector(PETERSBURG)._apply_surface_form_gate(candidates, evidence={})

        self.assertEqual(len(result.survivors), 2)
        self.assertEqual([item.supporting_run_count for item in candidates], [2, 3])

    def test_mount_expands_like_saint(self) -> None:
        candidates = [
            _candidate("Mt Everest", runs=4, agents=["a1"]),
            _candidate("Mount Everest", runs=1, agents=["a2"]),
        ]
        result = _selector(PETERSBURG)._apply_surface_form_gate(candidates, evidence={})

        survivors = [item.answer for item in result.survivors]
        self.assertEqual(survivors, ["Mount Everest"])
        self.assertEqual(candidates[1].supporting_run_count, 5)


if __name__ == "__main__":
    unittest.main()
