from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from tools.deterministic_solver.handlers.string_handler import StringHandler

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract
from .solver_backed import SolverBackedRouterHandler


@dataclass(frozen=True)
class TextOrientationResult:
    """Describe whether a text becomes linguistically clearer when reversed."""

    original: str
    decoded: str
    reversed_text: bool
    original_word_ratio: float
    decoded_word_ratio: float


class TextOrientationDetector:
    """Detect whole-text reversal using lexical orientation, not task phrases."""

    _WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    _FUNCTION_WORDS = {
        "a", "an", "and", "as", "at", "be", "by", "do", "for", "from",
        "has", "have", "how", "if", "in", "is", "it", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when", "where", "which",
        "who", "why", "with", "write", "you", "your",
    }

    def detect(self, text: str) -> TextOrientationResult:
        original = str(text or "").strip()
        decoded = original[::-1].strip()
        original_ratio = self._lexical_ratio(original)
        decoded_ratio = self._lexical_ratio(decoded)
        reversed_text = (
            len(self._WORD_RE.findall(original)) >= 4
            and decoded_ratio >= 0.35
            and decoded_ratio >= original_ratio + 0.25
        )
        return TextOrientationResult(
            original=original,
            decoded=decoded,
            reversed_text=reversed_text,
            original_word_ratio=round(original_ratio, 6),
            decoded_word_ratio=round(decoded_ratio, 6),
        )

    def _lexical_ratio(self, text: str) -> float:
        words = [word.lower() for word in self._WORD_RE.findall(text)]
        if not words:
            return 0.0
        recognized = sum(
            1
            for word in words
            if word in self._FUNCTION_WORDS or self._looks_like_english_word(word)
        )
        return recognized / len(words)

    def _looks_like_english_word(self, word: str) -> bool:
        if len(word) < 3 or not re.search(r"[aeiouy]", word):
            return False
        # Reversed English often creates implausible initial/final clusters.  This
        # language-shape check only confirms the function-word orientation signal.
        return not (
            re.match(r"^(?:ht|dn|rw|tn|siht|eht)", word)
            or re.search(r"(?:ht|dn|rw)$", word)
        )


class StringTransformRouterHandler(SolverBackedRouterHandler):
    name = "string_transform"
    capability_description = (
        "Perform exact string transformations and decode text whose entire character "
        "orientation is reversed."
    )
    supported_attachment_types: set[str] = {".txt", ".json"}
    routing_terms = {"uppercase", "lowercase", "reverse", "spaces", "characters", "words", "string", "title"}
    missing_inputs = ["quoted_or_inline_text", "string_operation"]
    input_schema = io_contract(
        name,
        [
            input_field("quoted_or_inline_text", "str", True, "Text to transform.", "question|attachment"),
            input_field("string_operation", "str", True, "String operation such as uppercase, reverse, or word count.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self) -> None:
        super().__init__(StringHandler())
        self.orientation_detector = TextOrientationDetector()

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        orientation = self.orientation_detector.detect(handler_input.question)
        if orientation.reversed_text:
            return HandlerMatch(
                handler_name=self.name,
                matched=True,
                confidence=0.99,
                reason="whole_text_orientation_reversal",
                missing_inputs=[],
            )
        return super().match_input(handler_input)

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        inputs = super().build_input(handler_input)
        inputs["orientation"] = self.orientation_detector.detect(handler_input.question)
        return inputs

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        orientation = inputs.get("orientation")
        if not isinstance(orientation, TextOrientationResult):
            orientation = self.orientation_detector.detect(str(inputs.get("question") or ""))
        if orientation.reversed_text:
            return HandlerResult(
                handler_name=self.name,
                status="ok",
                answer=orientation.decoded,
                confidence=0.99,
                output_type="intermediate_value",
                semantic_role="decoded_instruction",
                supporting_inputs=[orientation.original],
                structured_result={
                    "task_type": "whole_text_reversal",
                    "operation": "reverse_entire_text",
                    "original_word_ratio": orientation.original_word_ratio,
                    "decoded_word_ratio": orientation.decoded_word_ratio,
                    "decoded_instruction": orientation.decoded,
                },
                next_action_hint="Answer the decoded instruction; do not use the decoded sentence itself as the final answer.",
            )
        return super().run(inputs)


__all__ = [
    "StringTransformRouterHandler",
    "TextOrientationDetector",
    "TextOrientationResult",
]
