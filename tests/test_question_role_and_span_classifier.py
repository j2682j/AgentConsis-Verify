from __future__ import annotations

import unittest

from tools.search_result_builder.query import QuestionRole, QuestionRoleExtractor, SpanRoleClassifier
from tools.search_result_builder.query.span_repair import SalientSpan


class FakeScorer:
    def semantic_similarities(self, reference: str, texts: list[str]) -> list[float]:
        lowered = reference.lower()
        scores = []
        for text in texts:
            prototype = text.lower()
            score = 0.2
            if "how many studio albums" in lowered and "count" in prototype:
                score = 0.9
            elif "mercedes sosa" in lowered and "lookup anchor" in prototype:
                score = 0.86
            elif "between 2000 and 2009" in lowered and "time range" in prototype:
                score = 0.88
            elif "studio albums" in lowered and "requested answer type" in prototype:
                score = 0.84
            return_score = score
            scores.append(return_score)
        return scores


class QuestionRoleAndSpanClassifierTests(unittest.TestCase):
    def test_question_role_extracts_count_target(self):
        extractor = QuestionRoleExtractor(scorer=FakeScorer())

        role = extractor.extract(
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009?"
        )

        self.assertEqual(role.answer_role, "count")
        self.assertIn("studio albums", role.head_span.lower())
        self.assertIn("studio albums", role.answer_target.lower())

    def test_main_context_ignores_title_internal_which(self):
        extractor = QuestionRoleExtractor(scorer=FakeScorer())

        context = extractor.locate_main_question_context(
            'Of the authors that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" '
            "in 2015, what was the title of the first paper authored by the one that had authored prior papers?"
        )

        self.assertIn("what was the title", context.text.lower())
        self.assertNotEqual(context.text.strip().lower(), 'which is better?"')

    def test_main_context_ignores_doctor_who_entity(self):
        extractor = QuestionRoleExtractor(scorer=FakeScorer())

        context = extractor.locate_main_question_context(
            "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. "
            "What is this location called in the official script for the episode? "
            "Give the setting exactly as it appears in the first scene heading."
        )

        self.assertIn("what is this location called", context.text.lower())
        self.assertIn("give the setting", context.text.lower())

    def test_main_context_keeps_boolean_can_question(self):
        extractor = QuestionRoleExtractor(scorer=FakeScorer())

        context = extractor.locate_main_question_context(
            "Each cell in the attached spreadsheet represents a plot of land. "
            "The color of the cell indicates who owns that plot. "
            "Can Earl walk through every plot he owns and return to his starting plot without backtracking? "
            "For this question, consider backtracking to be any repeated plot."
        )

        self.assertTrue(context.text.lower().startswith("can earl walk"))
        self.assertIn("for this question", context.text.lower())

    def test_span_classifier_uses_relation_to_question_role(self):
        classifier = SpanRoleClassifier(scorer=FakeScorer(), min_confidence=0.0)
        question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009?"
        role = QuestionRole(
            head_span="How many studio albums",
            answer_role="count",
            answer_target="studio albums",
        )
        spans = [
            SalientSpan(
                text="Mercedes Sosa",
                start=42,
                end=55,
                score=0.1,
                tokens=[],
                token_indices=[],
            ),
            SalientSpan(
                text="between 2000 and 2009",
                start=56,
                end=77,
                score=0.08,
                tokens=[],
                token_indices=[],
            ),
            SalientSpan(
                text="studio albums",
                start=9,
                end=22,
                score=0.07,
                tokens=[],
                token_indices=[],
            ),
        ]

        classified = classifier.classify(question, spans, question_role=role)
        roles = {span.text: span.role for span in classified}

        self.assertEqual(roles["Mercedes Sosa"], "source_clue")
        self.assertEqual(roles["between 2000 and 2009"], "constraint")
        self.assertEqual(roles["studio albums"], "answer_target")


if __name__ == "__main__":
    unittest.main()
