from __future__ import annotations

import re

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, QuestionAnalysis


class QuestionTermFilter:
    """
    Reject candidate answers copied from the question or matching invalid patterns.

    Args:
        - None.

    Returns:
        - QuestionTermFilter: Candidate filtering helper.
    """

    GENERIC_TERMS = {
        "abstract",
        "according",
        "article",
        "author",
        "book",
        "chapter",
        "figure",
        "house",
        "minister",
        "number",
        "official",
        "paper",
        "person",
        "place",
        "prime",
        "prime minister",
        "question",
        "result",
        "source",
        "table",
        "title",
        "website",
        "year",
    }

    def filter(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis,
        candidates: list[CandidateAnswer],
    ) -> tuple[list[CandidateAnswer], list[dict[str, str]]]:
        """
        Filter candidates and return rejection diagnostics.

        Args:
            - question: Original task question.
            - analysis: QuestionAnalysis generated from the question.
            - candidates: Raw candidates extracted from evidence.

        Returns:
            - tuple[list[CandidateAnswer], list[dict[str, str]]]: Kept candidates and rejected items.
        """
        kept: list[CandidateAnswer] = []
        rejected: list[dict[str, str]] = []
        question_key = self._key(question)
        banned_keys = {self._key(term) for term in analysis.banned_answer_terms if term}

        for candidate in candidates:
            reason = self._reject_reason(
                candidate=candidate,
                analysis=analysis,
                question_key=question_key,
                banned_keys=banned_keys,
            )
            if reason:
                rejected.append({"answer": candidate.answer, "reason": reason})
                continue
            kept.append(candidate)

        return kept, rejected

    def _reject_reason(
        self,
        *,
        candidate: CandidateAnswer,
        analysis: QuestionAnalysis,
        question_key: str,
        banned_keys: set[str],
    ) -> str:
        text = normalize_text(candidate.answer).strip()
        key = self._key(text)
        if not key:
            return "empty_candidate"
        if key in banned_keys:
            return "question_term"
        if key in self.GENERIC_TERMS:
            return "generic_term"
        if len(key) <= 3 and analysis.answer_type not in {"symbol", "code"}:
            return "too_short"
        if analysis.answer_type in {"number", "date"} and key in banned_keys:
            return "given_constraint"
        if analysis.answer_type != "date" and re.fullmatch(r"(?:19|20)\d{2}", key):
            return "standalone_year"
        if analysis.answer_type == "person" and not self._looks_like_person(text):
            return "answer_type_mismatch"
        if analysis.answer_type == "code" and not self._looks_like_code(text):
            return "answer_type_mismatch"
        if analysis.answer_type == "symbol" and len(text) > 20:
            return "answer_type_mismatch"
        if analysis.answer_type == "word" and not re.fullmatch(r"[A-Za-z][A-Za-z-]{3,40}", text):
            return "answer_type_mismatch"
        if f" {key} " in f" {question_key} " and analysis.answer_type not in {"symbol"}:
            return "copied_from_question"
        return ""

    def _looks_like_person(self, text: str) -> bool:
        if text.lower() in self.GENERIC_TERMS:
            return False
        return bool(re.fullmatch(r"[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4}", text))

    def _looks_like_code(self, text: str) -> bool:
        return bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+|[A-Z]{1,8}[-_ ]?\d{1,8}", text))

    def _key(self, text: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(text).lower()).strip(" .,;:-")


__all__ = ["QuestionTermFilter"]

