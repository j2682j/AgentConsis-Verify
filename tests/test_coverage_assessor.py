from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from tools.search_result_builder.next_hop_query.coverage_assessor import CoverageAssessor


@dataclass
class Doc:
    title: str
    text: str
    retrieval_score: float = 0.8
    useful_tokens: list[str] = field(default_factory=list)


class CoverageAssessorTests(unittest.TestCase):
    def test_zip_code_answer_type_requires_five_digit_evidence(self):
        assessor = CoverageAssessor()

        result = assessor.assess(
            question="According to the USGS, where was the fish found before 2020? Give the five-digit zip code.",
            documents=[
                Doc(
                    title="USGS nonnative fish",
                    text="The fish was observed in Florida before 2020, but the page does not list a postal code.",
                )
            ],
        )

        self.assertFalse(result.sufficient)
        self.assertEqual(result.answer_type, "zip_code")
        self.assertFalse(result.answer_type_covered)
        self.assertIn("answer_type_not_covered", result.trigger_reason)

    def test_sufficient_when_constraints_and_answer_type_are_covered(self):
        assessor = CoverageAssessor()

        result = assessor.assess(
            question="According to the USGS, where was the fish found before 2020? Give the five-digit zip code.",
            documents=[
                Doc(
                    title="USGS nonnative fish location",
                    text="USGS records show the fish was found before 2020 near ZIP code 34689.",
                    useful_tokens=["USGS", "34689"],
                )
            ],
        )

        self.assertTrue(result.sufficient)
        self.assertEqual(result.answer_type, "zip_code")
        self.assertTrue(result.answer_type_covered)
        self.assertIn("source:usgs", result.covered_constraints)

    def test_bridge_terms_exclude_original_question_terms(self):
        assessor = CoverageAssessor()

        result = assessor.assess(
            question="Who led Example Org in Taiwan in 2024?",
            documents=[
                Doc(
                    title="Leadership note",
                    text="The board named Alice Chen as director in a governance update.",
                    useful_tokens=["Alice Chen"],
                )
            ],
        )

        lowered_question = {term.lower() for term in ["Example", "Org", "Taiwan", "2024"]}
        self.assertTrue(result.bridge_terms)
        self.assertFalse(any(term.lower() in lowered_question for term in result.bridge_terms))


if __name__ == "__main__":
    unittest.main()
