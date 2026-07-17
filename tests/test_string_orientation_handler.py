from __future__ import annotations

import unittest

from tools.deterministic_handlers.base import HandlerInput, render_handler_evidence
from tools.deterministic_handlers.handlers.string_transform import StringTransformRouterHandler


class StringOrientationHandlerTests(unittest.TestCase):
    def test_reversed_instruction_is_intermediate_value(self) -> None:
        question = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
        handler = StringTransformRouterHandler()
        handler_input = HandlerInput(question=question)

        match = handler.match_input(handler_input)
        result = handler.run(handler.build_input(handler_input))

        self.assertTrue(match.matched)
        self.assertEqual(
            result.answer,
            'If you understand this sentence, write the opposite of the word "left" as the answer.',
        )
        self.assertEqual(result.output_type, "intermediate_value")
        self.assertEqual(result.semantic_role, "decoded_instruction")
        self.assertIn("not a final answer", render_handler_evidence(result))

    def test_normal_question_is_not_detected_as_reversed(self) -> None:
        handler = StringTransformRouterHandler()
        result = handler.orientation_detector.detect("What is the capital city of France?")
        self.assertFalse(result.reversed_text)


if __name__ == "__main__":
    unittest.main()
