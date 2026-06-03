from __future__ import annotations

import re

from utils.network_utils import normalize_text

from .config import QuestionAnalysis


class QuestionAnalyzer:
    """
    對問題做輕量分析，提供 search pipeline 所需的 answer type 與關鍵詞。

    Args:
        - None.

    Returns:
        - QuestionAnalyzer: question analysis helper。
    """

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "answer",
        "question",
        "provide",
        "please",
        "this",
        "that",
        "are",
        "was",
        "were",
    }

    def analyze(self, question: str) -> QuestionAnalysis:
        """
        產生 QuestionAnalysis。

        Args:
            - question: 原始問題。

        Returns:
            - QuestionAnalysis: 問題分析結果。
        """
        text = normalize_text(question)
        lowered = text.lower()
        answer_type = self._answer_type(lowered)
        target_terms = self._target_terms(text)
        constraints = self._constraints(text)
        return QuestionAnalysis(
            answer_type=answer_type,
            target_terms=target_terms,
            constraints=constraints,
            source_hints=[],
            needs_multi_hop=bool(re.search(r"\b(?:after|before|then|first|last|oldest|newest|same|another)\b", lowered)),
        )

    def _answer_type(self, lowered_question: str) -> str:
        if any(marker in lowered_question for marker in ("url", "website", "web site")):
            return "website"
        if any(marker in lowered_question for marker in ("when", "date", "year")):
            return "date"
        if any(marker in lowered_question for marker in ("how many", "number", "count", "amount")):
            return "number"
        if "who" in lowered_question:
            return "person"
        if "where" in lowered_question:
            return "place"
        if any(marker in lowered_question for marker in ("title", "book", "paper", "movie", "song", "video")):
            return "title"
        if any(marker in lowered_question for marker in ("symbol", "character")):
            return "symbol"
        if "word" in lowered_question:
            return "word"
        return "entity"

    def _target_terms(self, question: str) -> list[str]:
        terms: list[str] = []
        for quoted in re.findall(r'"([^"]{2,120})"|\'([^\']{2,120})\'', question):
            value = next((part for part in quoted if part), "")
            self._append(terms, value)
        for match in re.findall(r"\b[A-Z][A-Za-z0-9&'.:-]+(?:\s+[A-Z0-9][A-Za-z0-9&'.:-]+){0,5}\b", question):
            self._append(terms, match)
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{2,}", question):
            if token.lower() not in self.STOPWORDS:
                self._append(terms, token)
        return terms[:16]

    def _constraints(self, question: str) -> list[str]:
        constraints: list[str] = []
        for value in re.findall(r"\b(?:19|20)\d{2}\b", question):
            self._append(constraints, value)
        for value in re.findall(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
            question,
            flags=re.IGNORECASE,
        ):
            self._append(constraints, value)
        return constraints[:8]

    def _append(self, values: list[str], value: str) -> None:
        cleaned = normalize_text(value).strip(" .,;:-")
        key = cleaned.lower()
        if cleaned and key not in {item.lower() for item in values}:
            values.append(cleaned)


__all__ = ["QuestionAnalyzer"]
