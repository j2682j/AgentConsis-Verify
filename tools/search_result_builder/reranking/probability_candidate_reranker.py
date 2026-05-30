from __future__ import annotations

import os
from typing import Any

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem
from ..search_query_generate.token_prob_compute import MODEL_NAME, TokenProbabilityAnalyzer


class ProbabilityCandidateReranker:
    """
    Rerank a small candidate set with HF token probability as a lightweight feature.

    Args:
        - analyzer: Optional TokenProbabilityAnalyzer-compatible object for tests or reuse.
        - model_name: HuggingFace causal LM name. Defaults to PROBABILITY_CANDIDATE_MODEL or token-prob default.
        - max_candidates: Maximum candidates to score.
        - max_evidence_sentences: Maximum evidence snippets per candidate.

    Returns:
        - ProbabilityCandidateReranker: Optional probability-assisted candidate reranker.
    """

    def __init__(
        self,
        *,
        analyzer: Any | None = None,
        model_name: str | None = None,
        max_candidates: int = 5,
        max_evidence_sentences: int = 3,
        probability_weight: float = 0.25,
    ) -> None:
        self.analyzer = analyzer
        self.model_name = model_name or os.getenv("PROBABILITY_CANDIDATE_MODEL", MODEL_NAME)
        self.max_candidates = max(1, max_candidates)
        self.max_evidence_sentences = max(1, max_evidence_sentences)
        self.probability_weight = max(0.0, min(probability_weight, 1.0))

    def rerank(
        self,
        *,
        question: str,
        candidates: list[CandidateAnswer],
        evidence_items: list[EvidenceItem],
    ) -> tuple[list[CandidateAnswer], dict[str, Any]]:
        """
        Score at most top-k candidates and return a probability-assisted ordering.

        Args:
            - question: Original task question.
            - candidates: Rule-based candidates after question-term filtering.
            - evidence_items: Evidence snippets used to score candidate spans.

        Returns:
            - list[CandidateAnswer]: Candidates ordered by combined rule/probability score.
            - dict[str, Any]: Diagnostics for JSON logs.
        """
        selected = candidates[: self.max_candidates]
        remaining = candidates[self.max_candidates :]
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "model_name": self.model_name,
            "max_candidates": self.max_candidates,
            "max_evidence_sentences": self.max_evidence_sentences,
            "scored": [],
            "error": "",
        }
        if not selected or not evidence_items:
            diagnostics["enabled"] = bool(selected and evidence_items)
            return candidates, diagnostics

        try:
            analyzer = self._analyzer()
            base_max = max((candidate.verification_score for candidate in selected), default=0.0) or 1.0
            scored: list[tuple[float, int, CandidateAnswer]] = []
            for index, candidate in enumerate(selected):
                probability_score, details = self._score_candidate(
                    analyzer=analyzer,
                    question=question,
                    candidate=candidate,
                    evidence_items=evidence_items,
                )
                base_score = max(candidate.verification_score, 0.0) / base_max
                combined_score = (
                    (1.0 - self.probability_weight) * base_score
                    + self.probability_weight * probability_score
                )
                candidate.probability_score = round(probability_score, 6)
                candidate.probability_details = {
                    **details,
                    "base_score": round(base_score, 6),
                    "combined_score": round(combined_score, 6),
                    "probability_weight": self.probability_weight,
                }
                diagnostics["scored"].append(
                    {
                        "answer": candidate.answer,
                        "base_score": candidate.probability_details["base_score"],
                        "probability_score": candidate.probability_score,
                        "combined_score": candidate.probability_details["combined_score"],
                        "evidence_ids": details.get("evidence_ids", []),
                    }
                )
                scored.append((combined_score, -index, candidate))

            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return [item[2] for item in scored] + remaining, diagnostics
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return candidates, diagnostics

    def _analyzer(self) -> Any:
        if self.analyzer is None:
            self.analyzer = TokenProbabilityAnalyzer(model_name=self.model_name)
        return self.analyzer

    def _score_candidate(
        self,
        *,
        analyzer: Any,
        question: str,
        candidate: CandidateAnswer,
        evidence_items: list[EvidenceItem],
    ) -> tuple[float, dict[str, Any]]:
        snippets = self._candidate_snippets(candidate.answer, evidence_items)
        if not snippets:
            return 0.0, {"reason": "candidate_not_found_in_evidence", "evidence_ids": []}

        scores: list[float] = []
        details: list[dict[str, Any]] = []
        for evidence, snippet, text_unit in snippets[: self.max_evidence_sentences]:
            scored_unit = analyzer.score_text_unit(snippet, text_unit)
            probability_score = self._salience_score(scored_unit.logprob_avg, scored_unit.logprob_min)
            scores.append(probability_score)
            details.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "text_unit": text_unit,
                    "logprob_avg": scored_unit.logprob_avg,
                    "logprob_min": scored_unit.logprob_min,
                    "score": round(probability_score, 6),
                }
            )

        best = max(scores) if scores else 0.0
        return round(best, 6), {
            "reason": "span_probability",
            "evidence_ids": [item["evidence_id"] for item in details],
            "span_scores": details,
            "question_hint": normalize_text(question)[:160],
        }

    def _candidate_snippets(
        self,
        answer: str,
        evidence_items: list[EvidenceItem],
    ) -> list[tuple[EvidenceItem, str, str]]:
        answer = normalize_text(answer)
        if not answer:
            return []

        matches: list[tuple[EvidenceItem, str, str]] = []
        for evidence in evidence_items:
            text = normalize_text(evidence.text)
            text_unit = self._matching_text_unit(text, answer)
            if not text_unit:
                continue
            snippet = self._sentence_window(text, text_unit)
            matches.append((evidence, snippet, text_unit))
        return matches

    def _matching_text_unit(self, text: str, answer: str) -> str:
        position = text.lower().find(answer.lower())
        if position < 0:
            return ""
        return text[position: position + len(answer)]

    def _sentence_window(self, text: str, text_unit: str) -> str:
        position = text.find(text_unit)
        if position < 0:
            return text[:500]

        left = max(text.rfind(".", 0, position), text.rfind("?", 0, position), text.rfind("!", 0, position))
        right_candidates = [
            value for value in (
                text.find(".", position + len(text_unit)),
                text.find("?", position + len(text_unit)),
                text.find("!", position + len(text_unit)),
            )
            if value >= 0
        ]
        start = 0 if left < 0 else left + 1
        end = min(right_candidates) + 1 if right_candidates else min(len(text), position + 260)
        snippet = normalize_text(text[start:end])
        if text_unit not in snippet:
            snippet = normalize_text(text[max(0, position - 180): position + len(text_unit) + 180])
        return snippet

    def _salience_score(self, logprob_avg: float | None, logprob_min: float | None) -> float:
        values = [abs(value) for value in (logprob_avg, logprob_min) if value is not None]
        if not values:
            return 0.0
        return max(0.0, min(max(values) / 20.0, 1.0))


__all__ = ["ProbabilityCandidateReranker"]
