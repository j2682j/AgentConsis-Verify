from __future__ import annotations

import re

from utils.network_utils import normalize_text

from ..config import QuestionAnalysis


class QuestionAnalyzer:
    """
    Build lightweight question analysis used by search candidate generation.

    Args:
        - None.

    Returns:
        - QuestionAnalyzer: Rule-based question analyzer.
    """

    ROLE_TERMS = {
        "author",
        "person",
        "prime minister",
        "president",
        "company",
        "institution",
        "organization",
        "paper",
        "article",
        "book",
        "video",
        "title",
        "website",
        "source",
        "figure",
        "axis",
        "axes",
        "label",
        "word",
        "number",
        "date",
        "year",
        "answer",
    }

    STOP_ENTITY_HEADS = {
        "A",
        "An",
        "As",
        "At",
        "By",
        "For",
        "From",
        "How",
        "If",
        "In",
        "Of",
        "On",
        "Please",
        "The",
        "There",
        "This",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
    }

    def analyze(self, question: str) -> QuestionAnalysis:
        """
        Analyze the original question for typed extraction and filtering.

        Args:
            - question: Original task question.

        Returns:
            - QuestionAnalysis: Answer type, target terms, constraints, and banned terms.
        """
        text = normalize_text(question)
        lowered = text.lower()
        answer_type = self._answer_type(lowered)
        constraints = self._constraints(text)
        source_hints = self._source_hints(lowered)
        target_terms = self._target_terms(text)
        banned = self._banned_terms(text, constraints=constraints)

        return QuestionAnalysis(
            answer_type=answer_type,
            target_terms=target_terms,
            constraints=constraints,
            source_hints=source_hints,
            banned_answer_terms=banned,
            requires_verification=True,
            needs_calculation=self._needs_calculation(lowered),
            needs_multi_hop=self._needs_multi_hop(lowered),
        )

    def _answer_type(self, lowered: str) -> str:
        if "ec number" in lowered or "ec numbers" in lowered:
            return "code"
        if any(marker in lowered for marker in ("which of these words", "label word", "which word")):
            return "word"
        if any(marker in lowered for marker in ("what symbol", "which symbol", "character", "punctuation")):
            return "symbol"
        if "who" in lowered:
            return "person"
        if "where" in lowered and "where each" not in lowered:
            return "place"
        if "when" in lowered or "date" in lowered:
            return "date"
        if any(marker in lowered for marker in ("title", "book", "paper", "video", "song", "film", "movie")):
            return "title"
        if "website" in lowered or "url" in lowered:
            return "website"
        if any(marker in lowered for marker in ("how many", "number", "count", "amount", "percent", "percentage")):
            return "number"
        return "entity"

    def _constraints(self, text: str) -> list[str]:
        values: list[str] = []
        for value in re.findall(r"\b(?:19|20)\d{2}\b", text):
            self._append(values, value)
        for value in re.findall(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
            text,
            flags=re.IGNORECASE,
        ):
            self._append(values, normalize_text(value))
        for value in re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
            self._append(values, value)
        return values[:8]

    def _source_hints(self, lowered: str) -> list[str]:
        hints: list[str] = []
        source_markers = {
            "arxiv": "arxiv",
            "wikipedia": "wikipedia",
            "official": "official",
            "new international version": "NIV",
            "british museum": "British Museum",
            "usgs": "USGS",
        }
        for marker, hint in source_markers.items():
            if marker in lowered:
                self._append(hints, hint)
        return hints

    def _target_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        for quoted in re.findall(r'"([^"]{2,120})"|\'([^\']{2,120})\'', text):
            value = next((part for part in quoted if part), "")
            self._append(terms, normalize_text(value))

        for value in re.findall(r"\b[A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,5}\b", text):
            cleaned = self._trim_entity(value)
            if cleaned and cleaned.lower() not in self.ROLE_TERMS:
                self._append(terms, cleaned)

        keyword_patterns = (
            r"\bSPFMV\b",
            r"\bSPCSV\b",
            r"\b[A-Z]{2,}[A-Z0-9-]*\b",
            r"\barXiv\.org\b",
        )
        for pattern in keyword_patterns:
            for value in re.findall(pattern, text):
                self._append(terms, normalize_text(value))

        return terms[:8]

    def _banned_terms(self, text: str, *, constraints: list[str]) -> list[str]:
        banned: list[str] = []
        lowered = normalize_text(text).lower()
        for constraint in constraints:
            self._append(banned, constraint)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", constraint):
                self._append(banned, year)

        for term in self.ROLE_TERMS:
            if term in lowered:
                self._append(banned, term)

        for token in re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text):
            if token.lower() in {
                "the", "and", "for", "with", "from", "which", "what", "where",
                "when", "who", "this", "that", "these", "return", "answer",
                "submitted", "mentioned", "included", "according",
            }:
                continue
            if token[0].isupper() or token.lower() in self.ROLE_TERMS:
                self._append(banned, token)
        return banned[:80]

    def _needs_calculation(self, lowered: str) -> bool:
        return any(marker in lowered for marker in ("calculate", "sum", "average", "distance", "round", "how many"))

    def _needs_multi_hop(self, lowered: str) -> bool:
        return any(marker in lowered for marker in ("first", "then", "two most", "in the order", "which of these"))

    def _trim_entity(self, value: str) -> str:
        tokens = normalize_text(value).split()
        while tokens and tokens[0] in self.STOP_ENTITY_HEADS:
            tokens = tokens[1:]
        return " ".join(tokens).strip(" .,;:-")

    def _append(self, values: list[str], value: str) -> None:
        cleaned = normalize_text(value).strip(" .,;:-")
        if cleaned and cleaned.lower() not in {item.lower() for item in values}:
            values.append(cleaned)


__all__ = ["QuestionAnalyzer"]

