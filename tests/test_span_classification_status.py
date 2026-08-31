"""Separate "this span does nothing" from "the classifier gave up".

`_role_from_scores` returns `"other"` for both, so a single value covered two
different outcomes. Annotating 133 spans measured what that cost: of the 39 the
system labelled `other`, **all 39** were abstentions, while a human judged only
5 spans genuinely inert -- four of which were among the abstained. Reported
together, `other` scored a precision of 0.103, which reads as a classifier that
判斷 badly. Reported apart, it is right 75.5% of the time it commits and
declines on 29.3% of spans. Those call for different repairs.

`role` keeps its old values because query generation and prompt assembly read
it: `source_clue` and `constraint` become `must_include`, `format_instruction`
becomes `avoid_terms`, and `other` is rendered into the prompt's other-spans
section. This change records the distinction and alters no decision, so the
tests below pin the behaviour rather than the intent -- the prompt sections have
to come out identical, in the same order, including duplicates.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.query.span_classifier import (
    ClassifiedSpan,
    SpanRoleClassifier,
)


def span(text: str, role: str, *, unresolved: bool = False, score: float = 0.5,
         confidence: float = 0.05, top: str = "") -> ClassifiedSpan:
    return ClassifiedSpan(
        text=text,
        role=role,
        confidence=confidence,
        score=score,
        classification_status="unresolved" if unresolved else "resolved",
        semantic_role=None if unresolved else role,
        predicted_top_role=top or role,
    )


class StatusSeparationTest(unittest.TestCase):
    def test_an_abstention_carries_no_semantic_role(self) -> None:
        item = span("First M", "other", unresolved=True, top="source_clue")

        self.assertEqual(item.classification_status, "unresolved")
        self.assertIsNone(item.semantic_role)
        self.assertEqual(item.predicted_top_role, "source_clue")
        self.assertTrue(item.unresolved)

    def test_a_genuine_other_is_resolved(self) -> None:
        """The state the system has never produced, and must be able to."""

        item = span("authors", "other")

        self.assertEqual(item.classification_status, "resolved")
        self.assertEqual(item.semantic_role, "other")
        self.assertFalse(item.unresolved)

    def test_the_legacy_role_is_unchanged_either_way(self) -> None:
        for unresolved in (True, False):
            with self.subTest(unresolved=unresolved):
                self.assertEqual(span("x", "other", unresolved=unresolved).role, "other")


class PromptCompatibilityTest(unittest.TestCase):
    def _mixed(self) -> list[ClassifiedSpan]:
        return [
            span("Pie Menus", "source_clue", score=0.9),
            span("First M", "other", unresolved=True, score=0.8, top="source_clue"),
            span("2015", "constraint", score=0.7),
            span("authors", "other", score=0.6),
            span("Last", "other", unresolved=True, score=0.5, top="constraint"),
            span("authors", "other", score=0.4),
        ]

    def test_grouped_still_puts_every_other_together(self) -> None:
        grouped = SpanRoleClassifier.grouped(SpanRoleClassifier, self._mixed())

        self.assertEqual(
            [item.text for item in grouped["other"]],
            ["First M", "authors", "Last", "authors"],
        )

    def test_legacy_other_matches_grouped_exactly(self) -> None:
        """What the prompt renders must not move."""

        spans = self._mixed()
        legacy = SpanRoleClassifier.grouped_by_status(SpanRoleClassifier, spans)[
            "legacy_other_spans"
        ]

        self.assertEqual(
            [item.text for item in legacy],
            [item.text for item in spans if item.role == "other"],
        )

    def test_the_split_keeps_input_order_and_duplicates(self) -> None:
        """Concatenating two filtered lists would reorder; one pass does not."""

        result = SpanRoleClassifier.grouped_by_status(SpanRoleClassifier, self._mixed())

        self.assertEqual(
            [i.text for i in result["unresolved_spans"]], ["First M", "Last"]
        )
        self.assertEqual(
            [i.text for i in result["semantic_other_spans"]], ["authors", "authors"]
        )
        self.assertEqual(
            len(result["legacy_other_spans"]),
            len(result["unresolved_spans"]) + len(result["semantic_other_spans"]),
        )

    def test_spans_outside_other_are_not_collected(self) -> None:
        result = SpanRoleClassifier.grouped_by_status(SpanRoleClassifier, self._mixed())

        for key in result:
            with self.subTest(key=key):
                self.assertTrue(all(i.role == "other" for i in result[key]))


class SerialisationTest(unittest.TestCase):
    def test_a_v2_record_round_trips(self) -> None:
        original = span("First M", "other", unresolved=True, top="source_clue")

        restored = ClassifiedSpan.from_dict(original.to_dict())

        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_a_v1_record_loads_without_inventing_a_status(self) -> None:
        """`role == "other"` in an old record cannot say which kind it was."""

        legacy = {"text": "authors", "role": "other", "confidence": 0.2}

        restored = ClassifiedSpan.from_dict(legacy)

        self.assertEqual(restored.classification_status, "legacy_unknown")
        self.assertIsNone(restored.semantic_role)
        self.assertEqual(restored.schema_version, 1)

    def test_a_v1_record_keeps_its_legacy_role(self) -> None:
        restored = ClassifiedSpan.from_dict(
            {"text": "Pie Menus", "role": "source_clue", "confidence": 0.2}
        )

        self.assertEqual(restored.role, "source_clue")
        self.assertEqual(restored.classification_status, "legacy_unknown")


class RoleDecisionTest(unittest.TestCase):
    """The threshold decides abstention; the collapsed role cannot.

    `legacy_role` is `"other"` both when `other` wins outright and when nothing
    clears the margin, so reconstructing the outcome from it needs a comparison
    against the argmax -- and that comparison is wrong in exactly one case: an
    `other` argmax with a short margin abstained, and would be read as a
    decision. The five cases below cover both axes.
    """

    def _classifier(self) -> SpanRoleClassifier:
        classifier = SpanRoleClassifier.__new__(SpanRoleClassifier)
        classifier.min_confidence = 0.015
        return classifier

    def test_a_clear_non_other_winner_is_resolved(self) -> None:
        decision = self._classifier()._role_from_scores(
            {"source_clue": 0.9, "other": 0.5}
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(decision.top_role, "source_clue")
        self.assertEqual(decision.legacy_role, "source_clue")

    def test_a_narrow_non_other_winner_abstains(self) -> None:
        decision = self._classifier()._role_from_scores(
            {"source_clue": 0.9, "constraint": 0.895}
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.top_role, "source_clue")
        self.assertEqual(decision.legacy_role, "other")

    def test_a_clear_other_winner_is_a_decision(self) -> None:
        decision = self._classifier()._role_from_scores(
            {"other": 0.9, "source_clue": 0.5}
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(decision.top_role, "other")
        self.assertEqual(decision.legacy_role, "other")

    def test_a_narrow_other_winner_abstains(self) -> None:
        """The case a comparison against the argmax gets wrong."""

        decision = self._classifier()._role_from_scores(
            {"other": 0.9, "source_clue": 0.895}
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.top_role, "other")
        self.assertEqual(decision.legacy_role, "other")

    def test_an_exact_tie_abstains(self) -> None:
        decision = self._classifier()._role_from_scores(
            {"other": 0.5, "source_clue": 0.5}
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
