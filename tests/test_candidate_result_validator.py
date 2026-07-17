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

    def test_requested_compact_list_is_not_rejected_as_verbose(self) -> None:
        answer = "3/4,1/4,3/4,3/4,2/4,1/2,5/35,7/21,30/5,30/5,3/4,1/15,1/3,4/9,1/8,32/23,103/170"
        result = CandidateResultValidator().validate(
            answer,
            question="Return a comma separated list with no whitespace in document order.",
            source_binding={"handler": "fraction_document"},
        )

        self.assertTrue(result.valid, result.reasons)


if __name__ == "__main__":
    unittest.main()
