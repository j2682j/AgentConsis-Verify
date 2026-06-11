from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, SearchSignals


@dataclass
class RetrievalDecision:
    """
    Store the retrieval sufficiency decision for next-hop search.

    Args:
        - need_next_hop: Whether another retrieval hop is required.
        - reason: Main decision reason.
        - confidence: Evidence sufficiency confidence from 0.0 to 1.0.
        - missing_info: Missing or weak information signals.
        - scores: Component scores used by the controller.

    Returns:
        - RetrievalDecision: Next-hop search decision.
    """

    need_next_hop: bool
    reason: str
    confidence: float = 0.0
    missing_info: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class RetrievalController:
    """
    Decide whether first-hop evidence is sufficient or a next-hop search is needed.

    Args:
        - min_candidate_support: Minimum support count for a candidate answer.
        - sufficiency_threshold: Minimum weighted sufficiency score to stop retrieval.
        - min_target_coverage: Minimum target-term coverage required when target terms exist.
        - min_novel_terms: Minimum evidence terms not already present in the question.
        - min_evidence_chars: Minimum combined evidence length.

    Returns:
        - RetrievalController: Rule-based retrieval sufficiency controller.
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
        "this",
        "that",
        "are",
        "was",
        "were",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "please",
        "provide",
    }

    def __init__(
        self,
        *,
        min_candidate_support: int = 1,
        sufficiency_threshold: float = 0.55,
        min_target_coverage: float = 0.4,
        min_novel_terms: int = 3,
        min_evidence_chars: int = 250,
    ) -> None:
        self.min_candidate_support = min_candidate_support
        self.sufficiency_threshold = sufficiency_threshold
        self.min_target_coverage = min_target_coverage
        self.min_novel_terms = min_novel_terms
        self.min_evidence_chars = min_evidence_chars

    def assess(
        self,
        *,
        evidence_items: list[EvidenceItem],
        candidates: list[CandidateAnswer],
        question: str = "",
        search_signals: SearchSignals | None = None,
    ) -> RetrievalDecision:
        """
        Assess whether current evidence is enough to stop retrieval.

        Args:
            - evidence_items: Evidence chunks produced by SourceAnalysis.
            - candidates: Candidate answers produced by the search pipeline.
            - question: Original question.
            - search_signals: Search signals, especially semantic-impact target terms.

        Returns:
            - RetrievalDecision: Whether next-hop search is required.
        """
        if not evidence_items:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_evidence",
                confidence=0.0,
                missing_info=["evidence"],
                scores=self._empty_scores(),
            )

        supported_candidates = [
            candidate
            for candidate in candidates
            if candidate.support_count >= self.min_candidate_support
        ]
        if candidates and not supported_candidates:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_supported_candidate",
                confidence=0.0,
                missing_info=["candidate_answer"],
                scores=self._empty_scores(evidence_presence=1.0),
            )

        evidence_text = self._combined_evidence_text(evidence_items)
        evidence_presence = min(1.0, len(evidence_items) / 2.0)
        length_score = min(1.0, len(evidence_text) / max(1, self.min_evidence_chars))
        target_coverage = self._target_term_coverage(
            evidence_text=evidence_text,
            search_signals=search_signals,
        )
        answer_signal = self._answer_signal_score(
            question=question,
            evidence_text=evidence_text,
            search_signals=search_signals,
        )
        novelty_score, novel_terms = self._novelty_score(question=question, evidence_text=evidence_text)
        useful_density = self._useful_density(evidence_items)

        sufficiency_score = (
            0.25 * evidence_presence
            + 0.20 * length_score
            + 0.25 * target_coverage
            + 0.15 * answer_signal
            + 0.10 * novelty_score
            + 0.05 * useful_density
        )
        sufficiency_score = round(max(0.0, min(sufficiency_score, 1.0)), 6)

        missing_info: list[str] = []
        if search_signals and search_signals.target_terms and target_coverage < self.min_target_coverage:
            missing_info.append("target_term_coverage")
        if self._infer_answer_type(question, search_signals) != "unknown" and answer_signal <= 0.0:
            missing_info.append("answer_signal")
        if novel_terms < self.min_novel_terms:
            missing_info.append("evidence_novelty")

        scores = {
            "sufficiency_score": sufficiency_score,
            "evidence_presence": round(evidence_presence, 6),
            "evidence_length": round(length_score, 6),
            "target_term_coverage": round(target_coverage, 6),
            "answer_signal": round(answer_signal, 6),
            "evidence_novelty": round(novelty_score, 6),
            "useful_density": round(useful_density, 6),
            "novel_terms": float(novel_terms),
        }

        if sufficiency_score < self.sufficiency_threshold or missing_info:
            return RetrievalDecision(
                need_next_hop=True,
                reason="insufficient_evidence",
                confidence=sufficiency_score,
                missing_info=missing_info,
                scores=scores,
            )

        return RetrievalDecision(
            need_next_hop=False,
            reason="sufficient_evidence",
            confidence=sufficiency_score,
            missing_info=[],
            scores=scores,
        )

    def _combined_evidence_text(self, evidence_items: list[EvidenceItem]) -> str:
        parts: list[str] = []
        for item in evidence_items:
            parts.extend([item.title, item.text, " ".join(item.matched_terms)])
        return normalize_text(" ".join(part for part in parts if part))

    def _target_term_coverage(
        self,
        *,
        evidence_text: str,
        search_signals: SearchSignals | None,
    ) -> float:
        target_terms = [
            normalize_text(term)
            for term in (search_signals.target_terms if search_signals else [])
            if normalize_text(term)
        ]
        if not target_terms:
            return 1.0

        evidence_key = self._match_key(evidence_text)
        covered = 0
        for term in target_terms:
            term_key = self._match_key(term)
            if not term_key:
                continue
            if term_key.strip() in evidence_key:
                covered += 1
                continue
            term_tokens = self._keywords(term)
            if term_tokens and any(token in self._keywords(evidence_text) for token in term_tokens):
                covered += 1
        return covered / max(1, len(target_terms))

    def _answer_signal_score(
        self,
        *,
        question: str,
        evidence_text: str,
        search_signals: SearchSignals | None,
    ) -> float:
        answer_type = self._infer_answer_type(question, search_signals)
        if answer_type == "unknown":
            return 1.0
        patterns = {
            "date": [
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b",
                r"\b(?:18|19|20)\d{2}\b",
            ],
            "number": [
                r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
                r"\b\d+(?:\.\d+)?\s*(?:km|mi|miles|hours|minutes|percent|%)\b",
            ],
            "person": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
            ],
            "place": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
            ],
            "url": [
                r"https?://\S+",
                r"\b[a-z0-9-]+\.(?:com|org|net|edu|gov|io|ai|co)\b",
            ],
            "code": [
                r"\b[A-Z]{2,}[-_A-Z0-9]*\b",
                r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\b",
            ],
        }
        lowered = evidence_text.lower()
        return 1.0 if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns[answer_type]) else 0.0

    def _novelty_score(self, *, question: str, evidence_text: str) -> tuple[float, int]:
        question_terms = set(self._keywords(question))
        evidence_terms = set(self._keywords(evidence_text))
        novel_terms = evidence_terms - question_terms
        count = len(novel_terms)
        return min(1.0, count / max(1, self.min_novel_terms)), count

    def _useful_density(self, evidence_items: list[EvidenceItem]) -> float:
        if not evidence_items:
            return 0.0
        average = sum(len(item.matched_terms) for item in evidence_items) / len(evidence_items)
        return min(1.0, average / 3.0)

    def _infer_answer_type(
        self,
        question: str,
        search_signals: SearchSignals | None,
    ) -> str:
        if search_signals and search_signals.answer_type and search_signals.answer_type != "unknown":
            return search_signals.answer_type
        lowered = normalize_text(question).lower()
        if any(marker in lowered for marker in ("when", "date", "year", "month", "day")):
            return "date"
        if any(marker in lowered for marker in ("how many", "how much", "number", "distance", "percentage", "percent")):
            return "number"
        if any(marker in lowered for marker in ("website", "url", "domain")):
            return "url"
        if re.search(r"\bwho\b", lowered):
            return "person"
        if any(marker in lowered for marker in ("where", "city", "country", "location", "place")):
            return "place"
        if any(marker in lowered for marker in ("code", "symbol", "identifier", "ec number", "zip code")):
            return "code"
        return "unknown"

    def _keywords(self, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower()):
            if token in self.STOPWORDS or len(token) <= 2 or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _match_key(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).lower())
        return f" {' '.join(cleaned.split())} "

    def _empty_scores(self, **overrides: float) -> dict[str, float]:
        scores = {
            "sufficiency_score": 0.0,
            "evidence_presence": 0.0,
            "evidence_length": 0.0,
            "target_term_coverage": 0.0,
            "answer_signal": 0.0,
            "evidence_novelty": 0.0,
            "useful_density": 0.0,
            "novel_terms": 0.0,
        }
        scores.update(overrides)
        return scores


__all__ = ["RetrievalController", "RetrievalDecision"]
