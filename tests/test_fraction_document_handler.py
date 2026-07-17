from __future__ import annotations

import unittest
from pathlib import Path

from tools.deterministic_handlers.router import DeterministicHandlerRouter


class FractionDocumentHandlerTests(unittest.TestCase):
    def setUp(self):
        self.image = Path(
            r"C:\SCP\data\gaia\2023\validation\9318445f-fe6a-4e1b-acbf-c68228c9906a.png"
        )
        if not self.image.is_file():
            self.skipTest("GAIA fraction fixture is not available")

    def test_extracts_literal_fractions_and_reduces_sample_problems(self):
        question = (
            "As a comma separated list with no whitespace, using the provided image "
            "provide all the fractions that use / as the fraction line and the answers "
            "to the sample problems. Order the list by appearance."
        )
        router = DeterministicHandlerRouter(similarity_fn=lambda left, right: 0.0)

        result = router.run(
            question=question,
            attachment={
                "file_path": str(self.image),
                "extension": ".png",
            },
            handler_name="fraction_document",
        )

        self.assertTrue(result.ok, result.to_dict())
        self.assertEqual(
            result.answer,
            "3/4,1/4,3/4,3/4,2/4,1/2,5/35,7/21,30/5,30/5,"
            "3/4,1/15,1/3,4/9,1/8,32/23,103/170",
        )
        self.assertEqual(result.structured_result["validation"]["sample_count"], 7)


if __name__ == "__main__":
    unittest.main()
