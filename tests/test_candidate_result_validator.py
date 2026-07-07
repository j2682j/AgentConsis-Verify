from __future__ import annotations

import unittest

from tools.validation import CandidateResultValidator


class CandidateResultValidatorTests(unittest.TestCase):
    def test_valid_candidate_with_evidence_passes(self):
        result = CandidateResultValidator().validate(
            "42",
            question="What is the answer?",
            evidence_text="The computed answer is 42.",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.cleaned_answer, "42")

    def test_refusal_candidate_is_invalid(self):
        result = CandidateResultValidator().validate(
            "unknown",
            question="What is the answer?",
            evidence_text="Evidence exists.",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "refusal")

    def test_missing_evidence_binding_is_invalid(self):
        result = CandidateResultValidator().validate(
            "Alice",
            question="Who is named?",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "missing_evidence_binding")

    def test_markdown_wrapping_is_cleaned(self):
        result = CandidateResultValidator().validate(
            "**Alice**",
            question="Who is named?",
            source_binding={"source": "handler"},
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.cleaned_answer, "Alice")


if __name__ == "__main__":
    unittest.main()
