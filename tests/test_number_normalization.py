from __future__ import annotations

import unittest

from utils.network_utils import answer_equivalence, normalize_number


class NumberNormalizationTests(unittest.TestCase):
    def test_preserves_integer_trailing_zeros(self):
        self.assertEqual(normalize_number("17000"), "17000")
        self.assertEqual(normalize_number("1000"), "1000")

    def test_removes_only_fractional_trailing_zeros(self):
        self.assertEqual(normalize_number("17.000"), "17")
        self.assertEqual(normalize_number("1.2300"), "1.23")
        self.assertEqual(normalize_number("0.000"), "0")

    def test_normalizes_grouped_numbers(self):
        self.assertEqual(normalize_number("17,000"), "17000")

    def test_different_integer_magnitudes_are_not_equivalent(self):
        self.assertFalse(answer_equivalence("17000", "17"))


if __name__ == "__main__":
    unittest.main()
