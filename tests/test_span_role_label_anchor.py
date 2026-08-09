"""Pin what ANSWER_SUPPORT is anchored on, and why it is not the active goal.

The span role classifier decides whether a retrieved passage can support an
answer. On level1_final_13 it returned ANSWER_SUPPORT for 37 of 2,209 spans
(1.7%) and not once for a span carrying the gold answer -- spans as plain as
"the bemoaned fluffy dragons remain prevalent alongside Tolkien's menacing
breed." came back BRIDGE. Those became bridge contracts, were rejected
downstream as goal mismatches 97 times, and the run finished with
evidence_count = 1 across 28 retrieval tasks. Every Agent then worked from
unverified references instead of evidence.

The cause was the wording, not the model. The prompt defined ANSWER_SUPPORT
against the active relation goal and added "a clue, entity, row, date, or
intermediate value is BRIDGE, even when it is relevant", which covers nearly
every span a web page yields.

Replayed offline on 47 spans known to contain the gold answer and 108 known not
to, with the real questions and goals:

    shipped wording      recall 13%   false positives 1%   tasks 3 of 9
    minus the clue rule  recall 30%   false positives 2%
    anchored on value    recall 43%   false positives 2%   tasks 8 of 9

These tests hold the anchor and the absence of the rules that suppressed it.
They are about prompt content because that is where the behaviour lives.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.evidence.span_role_classifier import (
    CandidateSpan,
    SpanRoleClassifier,
)

SPAN = CandidateSpan(
    id="1",
    text="the bemoaned fluffy dragons remain prevalent alongside Tolkien's menacing breed",
    source_title="Dragons are Tricksy",
    local_context="Ruth Stein in 1968 and Margaret Blount in 1974 both comment with distaste",
    source_id="doc-1",
    source_type="web",
)
GOALS = [{"goal_id": "G1", "subject": "Emily Midkiff", "relation": "quotes", "target": "word"}]


def _prompt(**overrides) -> str:
    kwargs = {
        "question": "What word was quoted from two different authors in distaste?",
        "answer_requirement": "one word",
        "answer_target": "word",
        "active_goal": "quotes",
        "next_goal": "",
        "relation_goals": GOALS,
        "spans": [SPAN],
    }
    kwargs.update(overrides)
    return SpanRoleClassifier()._prompt(**kwargs)


class LabelAnchorTest(unittest.TestCase):
    def test_answer_support_is_anchored_on_stating_a_candidate_answer(self) -> None:
        prompt = _prompt()

        self.assertIn(
            "ANSWER_SUPPORT = the span states a value that could be the question's final answer.",
            prompt,
        )

    def test_answer_support_is_not_anchored_on_the_active_goal(self) -> None:
        """Anchoring on the goal is what held the label to 1.7% of spans."""

        prompt = _prompt()

        self.assertNotIn("BRIDGE = fills the active goal", prompt)
        self.assertIn("BRIDGE = the span helps reach the answer but does not state it.", prompt)

    def test_the_rules_that_suppressed_the_label_are_gone(self) -> None:
        prompt = _prompt()

        self.assertNotIn("intermediate value is BRIDGE", prompt)
        self.assertNotIn("the fact object itself is a final answer value", prompt)
        self.assertNotIn("an individual item is not ANSWER_SUPPORT", prompt)

    def test_an_unconfirmed_candidate_still_counts(self) -> None:
        """Most real spans need another step; requiring certainty empties the label."""

        self.assertIn(
            "Prefer ANSWER_SUPPORT whenever the span contains a candidate answer value",
            _prompt(),
        )


class NoiseDefinitionTest(unittest.TestCase):
    """NOISE has to be defined by absence of content, not by a list of chrome.

    Listing kinds of chrome let the classifier file answer-bearing spans as
    "generic text". On level1_final_15, 10 of the 21 classified spans that
    contained the gold answer came back NOISE -- "Jack O'Neill: Isn't that hot?
    Teal'c: Extremely" among them. Replayed on those spans, stating the
    exclusion as a rule takes that rate from 47% to 13%, and the rescued spans
    land in BRIDGE rather than inflating ANSWER_SUPPORT (6 before, 6 after).
    """

    def test_a_span_stating_a_fact_is_excluded_from_noise(self) -> None:
        self.assertIn(
            "A span that states any fact about the entities in the question is never NOISE.",
            _prompt(),
        )

    def test_noise_is_no_longer_defined_as_generic_text(self) -> None:
        prompt = _prompt()

        self.assertNotIn("or generic text", prompt)
        self.assertIn("boilerplate carrying no factual content", prompt)

    def test_the_exclusion_shares_a_line_with_the_definition(self) -> None:
        """Not cosmetic: the split cost two thirds of the effect.

        The same fixed spans measured 13% NOISE with both sentences on one
        line and 33% with them on two, three runs each and identical every
        time. The classifier is deterministic here, so that gap is real.
        """

        line = next(
            candidate
            for candidate in _prompt().splitlines()
            if candidate.startswith("NOISE =")
        )

        self.assertIn("is never NOISE.", line)

    def test_the_chrome_examples_survive(self) -> None:
        """The rule replaces the catch-all, not the concrete cases."""

        prompt = _prompt()

        for example in ("navigation", "login", "captcha"):
            with self.subTest(example=example):
                self.assertIn(example, prompt)


class PromptStillCarriesItsContextTest(unittest.TestCase):
    """The rewrite must not drop what the classifier needs to decide."""

    def test_goal_binding_rule_survives_for_goal_driven_tasks(self) -> None:
        prompt = _prompt()

        self.assertIn("goal_id must name the one goal supported by the span", prompt)
        self.assertIn("For NOISE, goal_id must be an empty string.", prompt)

    def test_goal_free_tasks_are_told_to_leave_goal_id_empty(self) -> None:
        prompt = _prompt(relation_goals=[])

        self.assertIn("No relation goals are defined", prompt)

    def test_question_and_requirement_reach_the_model(self) -> None:
        prompt = _prompt()

        self.assertIn("Question: What word was quoted", prompt)
        self.assertIn("Answer Requirement: one word", prompt)
        self.assertIn("Answer Target: word", prompt)

    def test_the_span_and_its_context_reach_the_model(self) -> None:
        prompt = _prompt()

        self.assertIn("the bemoaned fluffy dragons", prompt)
        self.assertIn("Ruth Stein in 1968", prompt)
        self.assertIn("Source Title: Dragons are Tricksy", prompt)

    def test_every_candidate_is_still_required_back(self) -> None:
        prompt = _prompt()

        self.assertIn("exactly 1 JSON objects", prompt)
        self.assertIn("Do not skip any candidate id.", prompt)


if __name__ == "__main__":
    unittest.main()
