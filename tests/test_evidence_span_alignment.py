import unittest

from tools.evidence.fact_extraction import EvidenceFact, FactGroundingValidator
from tools.evidence.span_alignment import EvidenceSpanAligner


class EvidenceSpanAlignmentTests(unittest.TestCase):
    def test_aligns_punctuation_difference_to_original_text(self) -> None:
        source = "The mall's volume is 0.1777 m3, according to the paper."
        result = EvidenceSpanAligner().align(
            "The mall’s volume is 0.1777 m3 according to the paper",
            source,
        )
        self.assertTrue(result.valid)
        self.assertEqual(
            result.aligned_span,
            "The mall's volume is 0.1777 m3, according to the paper",
        )
        self.assertEqual(source[result.start_offset : result.end_offset], result.aligned_span)

    def test_rejects_changed_numeric_value(self) -> None:
        result = EvidenceSpanAligner().align(
            "The volume is 0.1778 m3.",
            "The volume is 0.1777 m3.",
        )
        self.assertFalse(result.valid)

    def test_marks_repeated_exact_span_ambiguous(self) -> None:
        result = EvidenceSpanAligner().align("Alpha", "Alpha appears. Alpha returns.")
        self.assertFalse(result.valid)
        self.assertTrue(result.ambiguous)

    def test_grounding_validator_uses_aligned_source_quote(self) -> None:
        source = "KGOT has studios in the Dimond Center."
        fact = EvidenceFact(
            fact_id="F1",
            subject="KGOT",
            relation="has studios in",
            object="Dimond Center",
            evidence_spans=["KGOT has studios in the Dimond Center"],
            source_id="D1",
        )
        result = FactGroundingValidator().validate(fact, source_text=source)
        self.assertEqual(result.grounding_status, "grounded")
        self.assertEqual(result.evidence_spans, ["KGOT has studios in the Dimond Center"])
        self.assertIn("evidence_alignment", result.qualifiers)


if __name__ == "__main__":
    unittest.main()
