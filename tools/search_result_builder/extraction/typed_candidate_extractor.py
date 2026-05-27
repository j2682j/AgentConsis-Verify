from __future__ import annotations

import re

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, QuestionAnalysis, SearchSourceCandidate


class TypedCandidateExtractor:
    """
    Extract answer candidates with patterns selected by QuestionAnalysis.answer_type.

    Args:
        - None.

    Returns:
        - TypedCandidateExtractor: Typed candidate extraction service.
    """

    STOP_CANDIDATES = {
        "Abstract The",
        "According",
        "Amazon",
        "April",
        "As",
        "Facebook",
        "Google",
        "Here",
        "Home",
        "House",
        "Jun",
        "June",
        "LinkedIn",
        "Official",
        "Prime Minister",
        "Result",
        "Results",
        "Search",
        "Source",
        "There",
        "This",
        "Twitter",
        "Wikipedia",
        "YouTube",
    }

    def extract_candidates(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis,
        evidence_items: list[EvidenceItem],
        sources: list[SearchSourceCandidate],
        max_candidates: int = 8,
    ) -> list[CandidateAnswer]:
        """
        Extract typed candidates from evidence chunks.

        Args:
            - question: Original task question.
            - analysis: QuestionAnalysis for answer type and constraints.
            - evidence_items: Evidence chunks selected from search sources.
            - sources: Search sources used to create evidence.
            - max_candidates: Maximum candidates to return.

        Returns:
            - list[CandidateAnswer]: Ranked typed candidate answers.
        """
        source_by_id = {source.source_id: source for source in sources}
        grouped: dict[str, CandidateAnswer] = {}
        for evidence in evidence_items:
            for answer in self._extract_answers(evidence.text, analysis=analysis):
                answer = normalize_text(answer).strip(" .,;:-")
                if not self._valid_answer(answer, analysis=analysis):
                    continue

                key = self._answer_key(answer)
                candidate = grouped.get(key)
                if candidate is None:
                    candidate = CandidateAnswer(
                        answer=answer,
                        answer_type=analysis.answer_type,
                        support_count=0,
                        verification_score=0.0,
                    )
                    grouped[key] = candidate

                source = source_by_id.get(evidence.source_id)
                candidate.support_count += 1
                candidate.verification_score += evidence.relevance_score + self._type_bonus(answer, analysis)
                if evidence.evidence_id not in candidate.evidence_ids:
                    candidate.evidence_ids.append(evidence.evidence_id)
                if source and source.source_id not in candidate.source_ids:
                    candidate.source_ids.append(source.source_id)
                candidate.verified = True

        candidates = list(grouped.values())
        candidates.sort(
            key=lambda item: (
                item.verification_score,
                item.support_count,
                len(item.answer),
            ),
            reverse=True,
        )
        return candidates[:max_candidates]

    def _extract_answers(self, text: str, *, analysis: QuestionAnalysis) -> list[str]:
        answer_type = analysis.answer_type
        candidates: list[str] = []
        if answer_type == "website":
            candidates.extend(re.findall(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+", text))
        elif answer_type == "date":
            candidates.extend(
                re.findall(
                    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}|(?:19|20)\d{2})\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        elif answer_type == "number":
            candidates.extend(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", text))
        elif answer_type == "code":
            candidates.extend(re.findall(r"\b\d+\.\d+\.\d+\.\d+\b|\b[A-Z]{1,8}[-_ ]?\d{1,8}\b", text))
        elif answer_type == "symbol":
            candidates.extend(re.findall(r"`([^`]{1,20})`", text))
            candidates.extend(re.findall(r"\b(?:backtick|space|tab|hyphen|dash|underscore|comma|period|colon|semicolon)\b", text, flags=re.IGNORECASE))
        elif answer_type == "word":
            candidates.extend(re.findall(r'"([A-Za-z][A-Za-z-]{3,40})"', text))
            candidates.extend(re.findall(r"\b([A-Za-z][A-Za-z-]{3,40})\s+society\b", text, flags=re.IGNORECASE))
            candidates.extend(re.findall(r"\bsociety\s+(?:is|as|called|described as)\s+([A-Za-z][A-Za-z-]{3,40})\b", text, flags=re.IGNORECASE))
        elif answer_type == "person":
            candidates.extend(
                re.findall(
                    r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4}\b",
                    text,
                )
            )
        elif answer_type == "title":
            candidates.extend(re.findall(r'"([^"]{4,140})"', text))
            candidates.extend(
                re.findall(
                    r"\b[A-Z][A-Za-z0-9'&:.-]+(?:\s+[A-Z0-9][A-Za-z0-9'&:.-]+){1,10}\b",
                    text,
                )
            )
        else:
            candidates.extend(
                re.findall(
                    r"\b[A-Z][A-Za-z0-9'&:.-]+(?:\s+[A-Z0-9][A-Za-z0-9'&:.-]+){0,6}\b",
                    text,
                )
            )
        return self._dedupe(candidates)

    def _valid_answer(self, answer: str, *, analysis: QuestionAnalysis) -> bool:
        text = normalize_text(answer).strip(" .,;:-")
        if len(text) < 1 or len(text) > 160:
            return False
        if text in self.STOP_CANDIDATES:
            return False
        lowered = text.lower()
        if lowered in {
            "abstract",
            "article",
            "author",
            "common",
            "figure",
            "paper",
            "prime minister",
            "question",
            "result",
            "source",
            "the",
        }:
            return False
        if analysis.answer_type == "person" and len(text.split()) < 2:
            return False
        if analysis.answer_type == "word" and not re.fullmatch(r"[A-Za-z][A-Za-z-]{3,40}", text):
            return False
        return True

    def _type_bonus(self, answer: str, analysis: QuestionAnalysis) -> float:
        if analysis.answer_type == "person" and re.fullmatch(r"[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4}", answer):
            return 4.0
        if analysis.answer_type == "code" and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", answer):
            return 5.0
        if analysis.answer_type == "symbol" and answer.lower() in {"backtick", "space", "tab", "hyphen", "dash", "underscore"}:
            return 4.0
        return 1.0

    def _answer_key(self, answer: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(answer).lower()).strip(" .,;:-")

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = normalize_text(value).strip(" .,;:-")
            key = self._answer_key(cleaned)
            if key and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result


__all__ = ["TypedCandidateExtractor"]

