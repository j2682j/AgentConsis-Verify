from __future__ import annotations

import unittest

from tools.search_result_builder.source_analyze.content_requirement import (
    ContentRequirementVerifier,
)


class ContentRequirementVerifierTests(unittest.TestCase):
    def test_pdf_text_requires_pdf_extraction(self):
        verifier = ContentRequirementVerifier()
        result = verifier.verify(
            required_content="pdf_text",
            content="[PDF Page 1] The requested result is 0.1777 m3.",
            method="pdf_pymupdf",
            content_type="application/pdf",
            status_code=200,
            content_complete=True,
            source_kind="academic",
        )

        self.assertTrue(result.requirement_met)
        self.assertEqual(result.state, "requirement_met")

    def test_full_page_rejects_truncated_content(self):
        verifier = ContentRequirementVerifier()
        result = verifier.verify(
            required_content="full_page",
            content="Partial document text",
            method="structured_html",
            status_code=200,
            content_complete=False,
        )

        self.assertFalse(result.requirement_met)
        self.assertEqual(result.missing_content, ["full_page"])


if __name__ == "__main__":
    unittest.main()
