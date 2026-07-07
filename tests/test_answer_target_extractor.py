from __future__ import annotations

import unittest

from tools.search_result_builder.config import EvidenceItem
from tools.search_result_builder.next_hop_query import (
    AnswerTargetExtractor,
    RetrievalController,
)


class FakeSemanticScorer:
    def semantic_similarities(self, reference, texts):
        del reference
        return [0.8 for _ in texts]


def evidence(text: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E1",
        source_id="S1",
        query_id="Q1",
        text=text,
    )


class AnswerTargetExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import spacy

            cls.nlp = spacy.load("en_core_web_md")
        except Exception as exc:
            raise unittest.SkipTest(f"spaCy dependency model unavailable: {exc}")

    def setUp(self):
        self.extractor = AnswerTargetExtractor(
            nlp=self.nlp,
            semantic_scorer=FakeSemanticScorer(),
        )

    def test_duration_focus_wins_over_subordinate_when(self):
        question = (
            "How many thousand hours would it take to run the distance "
            "when carrying out the calculation?"
        )
        target = self.extractor.extract(question)

        self.assertEqual(target.role, "duration")
        self.assertEqual(target.lemma, "hour")
        self.assertEqual(target.unit, "hour")
        self.assertIn("hours", target.phrase.lower())

    def test_calendar_year_is_date(self):
        target = self.extractor.extract("Which year was the paper published?")

        self.assertEqual(target.role, "date")
        self.assertEqual(target.lemma, "year")

    def test_count_question_is_number(self):
        target = self.extractor.extract(
            "How many studio albums were published between 2000 and 2009?"
        )

        self.assertEqual(target.role, "number")
        self.assertEqual(target.lemma, "album")

    def test_volume_question_extracts_unit(self):
        target = self.extractor.extract(
            "What was the volume in m^3 of the fish bag?"
        )

        self.assertEqual(target.role, "volume")
        self.assertEqual(target.lemma, "volume")
        self.assertEqual(target.unit, "m^3")

    def test_duration_constraint_requires_number_and_time_unit(self):
        question = (
            "How many thousand hours would it take to run the distance "
            "when carrying out the calculation?"
        )
        controller = RetrievalController(
            semantic_scorer=FakeSemanticScorer(),
            nlp=self.nlp,
            answer_target_extractor=self.extractor,
        )

        missing = controller.assess(
            question=question,
            evidence_items=[evidence("The Moon has a minimum perigee distance.")],
        )
        covered = controller.assess(
            question=question,
            evidence_items=[evidence("The calculated duration is 17130 hours.")],
        )

        self.assertIn(
            "answer_role:duration",
            missing.scores["constraint_details"]["missing"],
        )
        self.assertIn(
            "answer_role:duration",
            covered.scores["constraint_details"]["matched"],
        )
        self.assertEqual(
            covered.scores["constraint_details"]["answer_target"]["unit"],
            "hour",
        )


if __name__ == "__main__":
    unittest.main()
