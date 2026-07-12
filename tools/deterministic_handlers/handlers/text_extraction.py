from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field


class TextExtractionRouterHandler:
    name = "text_extraction"
    capability_description = (
        "Extract exact values from closed text, including quoted text, nth word, nth sentence, "
        "URLs, emails, IDs, capitalized names, and occurrence counts."
    )
    supported_attachment_types: set[str] = {".txt", ".md", ".json", ".csv", ".tsv"}
    routing_terms = {"extract", "find", "count", "occurrence", "word", "sentence", "url", "email", "quoted", "id"}
    input_schema = io_contract(
        name,
        [
            input_field("source_text", "str", True, "Closed text to extract from.", "question|attachment|search"),
            input_field("operation", "str", True, "Extraction operation.", "question"),
            input_field("ordinal", "int", False, "Ordinal index for nth word/sentence.", "question"),
            input_field("target", "str", False, "Target text for occurrence count.", "question"),
        ],
        [
            *default_outputs(),
            output_field("matched_text", "str", False, "Extracted text span."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        text = self._source_text(handler_input)
        operation = self._operation(handler_input.question)
        missing = []
        if not text:
            missing.append("source_text")
        if not operation:
            missing.append("extraction_operation")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.9 if not missing else 0.3,
            reason="source_text_and_extraction_operation_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        return {
            "question": handler_input.question,
            "source_text": self._source_text(handler_input),
            "operation": self._operation(handler_input.question),
            "ordinal": self._ordinal(handler_input.question),
            "target": self._target(handler_input.question),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        text = str(inputs.get("source_text") or "")
        operation = str(inputs.get("operation") or "")
        if not text:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["source_text"],
            )
        if operation == "nth_word":
            words = re.findall(r"\b\w+\b", text)
            index = int(inputs.get("ordinal") or 0)
            answer = words[index] if 0 <= index < len(words) else ""
        elif operation == "nth_sentence":
            sentences = self._sentences(text)
            index = int(inputs.get("ordinal") or 0)
            answer = sentences[index] if 0 <= index < len(sentences) else ""
        elif operation == "url":
            matches = re.findall(r"https?://[^\s)>\"]+", text)
            answer = matches[0] if matches else ""
        elif operation == "email":
            matches = re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
            answer = matches[0] if matches else ""
        elif operation == "quoted":
            matches = re.findall(r'"([^"]+)"|' + r"'([^']+)'", text)
            values = [left or right for left, right in matches if left or right]
            answer = values[0] if values else ""
        elif operation == "count_occurrences":
            target = str(inputs.get("target") or "")
            answer = str(len(re.findall(re.escape(target), text, re.IGNORECASE))) if target else ""
        else:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["extraction_operation"],
                structured_result={"operation": operation},
            )
        if not answer:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["matching_text"],
                structured_result={"operation": operation},
            )
        structured = {
            "task_type": f"text_{operation}",
            "operation": operation,
            "ordinal": inputs.get("ordinal"),
            "target": inputs.get("target"),
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Task: text_{operation}\n"
                f"Answer: {answer}\n"
                "Instruction: prefer this exact deterministic extraction for closed-text tasks."
            ),
            structured_result=structured,
            confidence=0.91,
            output_type="final_answer",
            semantic_role=f"text_{operation}",
            supporting_inputs=[
                str(inputs.get("target") or ""),
                str(inputs.get("ordinal") or ""),
            ],
        )

    def _source_text(self, handler_input: HandlerInput) -> str:
        return "\n".join(
            part
            for part in [handler_input.attachment_result, handler_input.search_result, handler_input.question]
            if str(part or "").strip()
        )

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        ordinal_words = "first|second|third|fourth|fifth"
        if (
            re.search(r"\b\d+(?:st|nd|rd|th)?\s+word\b", lowered)
            or re.search(rf"\b(?:{ordinal_words})\s+word\b", lowered)
            or "nth word" in lowered
        ):
            return "nth_word"
        if (
            re.search(r"\b\d+(?:st|nd|rd|th)?\s+sentence\b", lowered)
            or re.search(rf"\b(?:{ordinal_words})\s+sentence\b", lowered)
            or "nth sentence" in lowered
        ):
            return "nth_sentence"
        if "url" in lowered or "link" in lowered:
            return "url"
        if "email" in lowered or "e-mail" in lowered:
            return "email"
        if "quoted" in lowered or "inside quotes" in lowered:
            return "quoted"
        if "how many times" in lowered or "occurrences" in lowered or "count" in lowered:
            return "count_occurrences"
        return ""

    def _ordinal(self, question: str) -> int:
        lowered = str(question or "").lower()
        mapping = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
        for word, index in mapping.items():
            if re.search(rf"\b{word}\b", lowered):
                return index
        match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", lowered)
        return int(match.group(1)) - 1 if match else 0

    def _target(self, question: str) -> str:
        matches = re.findall(r'"([^"]+)"|' + r"'([^']+)'", question or "")
        values = [left or right for left, right in matches if left or right]
        return values[0] if values else ""

    def _sentences(self, text: str) -> list[str]:
        return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


__all__ = ["TextExtractionRouterHandler"]
