from __future__ import annotations

import re

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, FactCard, QuestionAnalysis, SearchSourceCandidate, VerifiedCandidate


class CandidateVerifier:
    """
    Verify candidate answers against extracted evidence and produce compact fact cards.

    Args:
        - None.

    Returns:
        - CandidateVerifier: Rule-based verifier for search candidates.
    """

    NEGATIVE_MARKERS = (
        "not ",
        "incorrect",
        "unrelated",
        "different",
        "rather than",
        "instead of",
        "no match",
        "couldn't find",
        "could not find",
        "no results",
    )

    def verify(
        self,
        *,
        question: str,
        candidates: list[CandidateAnswer],
        evidence_items: list[EvidenceItem],
        sources: list[SearchSourceCandidate],
        analysis: QuestionAnalysis | None = None,
    ) -> tuple[list[VerifiedCandidate], list[FactCard]]:
        """
        Build verified candidates and fact cards from current evidence.

        Args:
            - question: Original task question.
            - candidates: Candidate answers extracted from evidence.
            - evidence_items: Evidence chunks selected from search sources.
            - sources: Search sources used to create the evidence chunks.

        Returns:
            - tuple[list[VerifiedCandidate], list[FactCard]]: Verified candidates and fact cards.
        """
        if not candidates:
            return [], []

        question_terms = self._keywords(question)
        banned_terms = {
            self._answer_key(term)
            for term in (analysis.banned_answer_terms if analysis else [])
        }
        source_quality = self._source_quality_by_id(sources)
        fact_cards: list[FactCard] = []
        verified: list[VerifiedCandidate] = []

        for index, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{index}"
            item = VerifiedCandidate(
                candidate_id=candidate_id,
                answer=candidate.answer,
                answer_type=candidate.answer_type,
                evidence_ids=list(candidate.evidence_ids),
                risk_flags=self._risk_flags(candidate.answer),
            )
            has_verification_support = False
            if self._answer_key(candidate.answer) in banned_terms:
                item.risk_flags.append("question_term_candidate")
            if analysis and candidate.answer_type != analysis.answer_type:
                item.risk_flags.append("answer_type_mismatch")

            for evidence in evidence_items:
                relation, confidence, constraints = self._classify_relation(
                    question_terms=question_terms,
                    candidate=candidate,
                    evidence=evidence,
                    source_quality=source_quality.get(evidence.source_id, 0.4),
                )
                if relation == "irrelevant":
                    continue

                fact = FactCard(
                    fact_id=f"F{len(fact_cards) + 1}",
                    claim=self._claim_from_evidence(evidence.text),
                    relation=relation,
                    candidate_id=candidate_id,
                    source_id=evidence.source_id,
                    evidence_id=evidence.evidence_id,
                    confidence=round(confidence, 3),
                    constraint_matches=constraints,
                )
                fact_cards.append(fact)

                if relation == "support":
                    item.support_count += 1
                    item.support_fact_ids.append(fact.fact_id)
                    if evidence.query_id.startswith("V"):
                        has_verification_support = True
                elif relation == "refute":
                    item.refute_count += 1
                    item.refute_fact_ids.append(fact.fact_id)
                else:
                    item.neutral_count += 1

            if not has_verification_support:
                item.risk_flags.append("no_verification_support")
            item.confidence = self._candidate_confidence(item)
            verified.append(item)

        verified.sort(
            key=lambda value: (
                value.confidence,
                value.support_count,
                -value.refute_count,
                -len(value.risk_flags),
            ),
            reverse=True,
        )
        return verified, fact_cards

    def _classify_relation(
        self,
        *,
        question_terms: set[str],
        candidate: CandidateAnswer,
        evidence: EvidenceItem,
        source_quality: float,
    ) -> tuple[str, float, list[str]]:
        text = normalize_text(evidence.text)
        lower = text.lower()
        answer_key = normalize_text(candidate.answer).lower()
        constraints = sorted(term for term in question_terms if term in lower)[:8]
        has_answer = bool(answer_key and answer_key in lower)
        constraint_score = min(len(constraints) / 4, 1.0)
        base = 0.35 + constraint_score * 0.35 + source_quality * 0.2

        if has_answer and self._has_negative_context(lower, answer_key):
            return "refute", min(base + 0.2, 1.0), constraints
        if has_answer and constraints:
            return "support", min(base + 0.25, 1.0), constraints
        if has_answer:
            return "neutral", min(base, 0.65), constraints
        if len(constraints) >= 3:
            return "neutral", min(0.25 + constraint_score * 0.25, 0.55), constraints
        return "irrelevant", 0.0, []

    def _candidate_confidence(self, candidate: VerifiedCandidate) -> float:
        score = (
            candidate.support_count * 0.35
            - candidate.refute_count * 0.45
            + min(candidate.neutral_count, 2) * 0.05
        )
        if candidate.support_count >= 2:
            score += 0.15
        if "no_verification_support" not in candidate.risk_flags and candidate.support_count:
            score += 0.2
        if "question_term_candidate" in candidate.risk_flags:
            score -= 0.8
        if "answer_type_mismatch" in candidate.risk_flags:
            score -= 0.5
        if candidate.risk_flags:
            score -= 0.25
        score = max(0.0, min(1.0, score))
        if "no_verification_support" in candidate.risk_flags:
            score = min(score, 0.45)
        return round(score, 3)

    def _has_negative_context(self, text: str, answer_key: str) -> bool:
        position = text.find(answer_key)
        if position < 0:
            return False
        window = text[max(0, position - 80): position + len(answer_key) + 80]
        return any(marker in window for marker in self.NEGATIVE_MARKERS)

    def _claim_from_evidence(self, text: str, max_chars: int = 220) -> str:
        cleaned = normalize_text(text)
        sentence_match = re.search(r"^(.{40,220}?[.!?])\s", cleaned)
        claim = sentence_match.group(1) if sentence_match else cleaned[:max_chars]
        return claim.strip(" -")

    def _risk_flags(self, answer: str) -> list[str]:
        text = normalize_text(answer)
        flags: list[str] = []
        if re.search(r"\b[A-Z_]+_[A-Z0-9_]+\b", text):
            flags.append("placeholder_like_answer")
        if re.fullmatch(r"[A-Za-z]{1,3}", text):
            flags.append("short_generic_candidate")
        if text.lower() in {"unknown", "none", "n/a", "not enough information"}:
            flags.append("uncertain_answer")
        if len(text.split()) > 20:
            flags.append("too_verbose_candidate")
        return flags

    def _answer_key(self, answer: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(answer).lower()).strip(" .,;:-")

    def _keywords(self, question: str) -> set[str]:
        lowered = re.sub(r"[^\w\s]", " ", normalize_text(question).lower())
        stopwords = {
            "the", "and", "for", "with", "from", "what", "which", "who", "when",
            "where", "why", "how", "answer", "return", "provide", "please", "order",
            "your", "that", "this", "are", "was", "were", "the", "into", "about",
        }
        return {
            token
            for token in lowered.split()
            if len(token) > 2 and token not in stopwords
        }

    def _source_quality_by_id(self, sources: list[SearchSourceCandidate]) -> dict[str, float]:
        quality: dict[str, float] = {}
        for source in sources:
            domain = source.domain.lower()
            value = 0.45
            if any(marker in domain for marker in ("wikipedia.org", ".gov", ".edu", ".org")):
                value += 0.25
            if source.fetched:
                value += 0.1
            if source.rank <= 2:
                value += 0.1
            quality[source.source_id] = min(value, 1.0)
        return quality


__all__ = ["CandidateVerifier"]

