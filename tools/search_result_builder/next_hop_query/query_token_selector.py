from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..query.search_intent_planner import SearchIntentPlan
from ..query.semantic_impact import SemanticImpactScorer
from ..query.span_repair import SalientSpan, SpanRepairer


@dataclass(frozen=True)
class SelectedQueryTokens:
    """
    Store externally selected question-side tokens for next-hop query building.

    Args:
        - role_tokens: Tokens or short phrases bound to the expected answer role.
        - salient_tokens: High semantic-impact spans from the original question.
        - selected_tokens: Deduplicated final question-side tokens.
        - metadata: Diagnostics for the token selection path.

    Returns:
        - SelectedQueryTokens: Question-side tokens used by the composer.
    """

    role_tokens: list[str] = field(default_factory=list)
    salient_tokens: list[str] = field(default_factory=list)
    selected_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryTokenSelector:
    """
    Select original-question tokens with semantic impact and answer-role binding.

    Args:
        - semantic_scorer: Encoder-based token deletion scorer.
        - span_repairer: Converts token salience into readable spans.
        - max_salient_spans: Maximum semantic-impact spans retained.
        - max_selected_tokens: Maximum question-side tokens exposed to query composition.

    Returns:
        - QueryTokenSelector: Stateless selector for next-hop query composition.
    """

    ROLE_TOKEN_PATTERNS: dict[str, list[str]] = {
        "volume": [
            r"\bhow\s+(?:large|big|much)\b",
            r"\b(?:volume|capacity|m\^?3|cubic meters?)\b",
        ],
        "distance": [
            r"\bhow\s+far\b",
            r"\b(?:distance|kilometers?|km|miles?|meters?)\b",
        ],
        "duration": [
            r"\bhow\s+long\b",
            r"\b(?:duration|hours?|minutes?|seconds?|days?)\b",
        ],
        "count": [
            r"\bhow\s+many\b",
            r"\b(?:number|count|total)\b",
        ],
        "number": [
            r"\bhow\s+(?:many|much)\b",
            r"\b(?:number|amount|total|count)\b",
        ],
        "date": [
            r"\b(?:when|what date|which date|what year|which year)\b",
            r"\b(?:date|year|month|day)\b",
        ],
        "person": [
            r"\bwho\b",
            r"\b(?:first name|last name|full name|writer|author|quoted by|person|name)\b",
        ],
        "location": [
            r"\bwhere\b",
            r"\b(?:location|place|city|country)\b",
        ],
        "organization": [
            r"\b(?:organization|company|agency|institution|publisher)\b",
        ],
        "title": [
            r"\b(?:title|name)\b",
        ],
        "species": [
            r"\b(?:species|bird species)\b",
        ],
        "text_span": [
            r"\b(?:stand for|stands for|code|abbreviation|answer)\b",
        ],
        "zip_code": [
            r"\b(?:zip code|postal code|five-digit)\b",
        ],
    }

    def __init__(
        self,
        *,
        semantic_scorer: SemanticImpactScorer | None = None,
        span_repairer: SpanRepairer | None = None,
        max_salient_spans: int = 3,
        max_selected_tokens: int = 4,
    ) -> None:
        self.semantic_scorer = semantic_scorer or SemanticImpactScorer(
            max_salient_tokens=10
        )
        self.span_repairer = span_repairer or SpanRepairer(
            max_salient_spans=max_salient_spans
        )
        self.max_salient_spans = max(1, max_salient_spans)
        self.max_selected_tokens = max(1, max_selected_tokens)
        self._cache: dict[tuple[str, str], SelectedQueryTokens] = {}

    def select(
        self,
        *,
        question: str,
        intent_plan: SearchIntentPlan | None = None,
    ) -> SelectedQueryTokens:
        """
        Select tokens from the original question for next-hop composition.

        Args:
            - question: Original user task.
            - intent_plan: Planner output carrying answer_role and state.

        Returns:
            - SelectedQueryTokens: Role-bound and semantic-impact spans.
        """
        normalized_question = normalize_text(question)
        answer_role = normalize_text(
            str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
        ).casefold() or "unknown"
        cache_key = (normalized_question, answer_role)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        role_tokens = self._role_tokens(normalized_question, answer_role)
        salient_spans, salience_metadata = self._salient_spans(normalized_question)
        salient_tokens = [span.text for span in salient_spans[: self.max_salient_spans]]
        selected = self._dedupe(role_tokens + salient_tokens)[: self.max_selected_tokens]
        if not selected:
            selected = self._fallback_tokens(normalized_question)[: self.max_selected_tokens]

        result = SelectedQueryTokens(
            role_tokens=role_tokens,
            salient_tokens=salient_tokens,
            selected_tokens=selected,
            metadata={
                "method": "semantic_impact_answer_role_binding",
                "answer_role": answer_role,
                **salience_metadata,
            },
        )
        self._cache[cache_key] = result
        return result

    def _role_tokens(self, question: str, answer_role: str) -> list[str]:
        role = self._normalize_role(answer_role)
        candidates: list[str] = []
        for pattern in self.ROLE_TOKEN_PATTERNS.get(role, []):
            for match in re.finditer(pattern, question, flags=re.IGNORECASE):
                token = normalize_text(match.group(0)).strip(" ?.,;:")
                if token:
                    candidates.append(token)

        if not candidates and role not in {"", "unknown"}:
            role_text = role.replace("_", " ")
            if role_text not in {"number", "text span"}:
                candidates.append(role_text)
        return self._dedupe(candidates)[:2]

    def _salient_spans(self, question: str) -> tuple[list[SalientSpan], dict[str, Any]]:
        if not question:
            return [], {"salience_error": "empty_question"}
        try:
            tokens = self.semantic_scorer.score_tokens(question)
            kept_tokens = self.semantic_scorer.filter_tokens(tokens)
            spans = self.span_repairer.build_spans(question, kept_tokens)
            return spans, {
                "salience_error": "",
                "semantic_impact_spans": [
                    {
                        "text": span.text,
                        "score": span.score,
                        "start": span.start,
                        "end": span.end,
                    }
                    for span in spans[: self.max_salient_spans]
                ],
            }
        except Exception as exc:
            return [], {"salience_error": f"{type(exc).__name__}: {exc}"}

    def _fallback_tokens(self, question: str) -> list[str]:
        tokens: list[str] = []
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]{2,}", question):
            key = token.casefold()
            if key in SemanticImpactScorer.STOPWORDS:
                continue
            if key in SemanticImpactScorer.GENERIC_QUERY_TERMS:
                continue
            tokens.append(token)
        return self._dedupe(tokens)

    def _normalize_role(self, answer_role: str) -> str:
        role = normalize_text(answer_role).casefold()
        return {
            "place": "location",
            "short_phrase": "text_span",
            "boolean": "text_span",
        }.get(role, role)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(str(value or "")).strip(" \"'`.,;:")
            key = text.casefold()
            if not text or key in seen:
                continue
            result.append(text)
            seen.add(key)
        return result


__all__ = ["QueryTokenSelector", "SelectedQueryTokens"]
