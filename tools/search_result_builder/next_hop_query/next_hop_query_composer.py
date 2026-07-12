from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..config import EvidenceItem
from ..query.search_intent_planner import SearchIntentPlan
from .query_token_selector import QueryTokenSelector
from .rag_filter import RAGFilterResult


@dataclass(frozen=True)
class NextHopComposition:
    """
    Store the externally composed next-hop query and its selected tokens.

    Args:
        - query: Final next-hop query.
        - selected_query_tokens: Question-side tokens retained from the original task.
        - selected_evidence_spans: Evidence-side bridge spans selected from Labeler output.
        - metadata: Diagnostics for report/export.

    Returns:
        - NextHopComposition: Query composition result.
    """

    query: str
    selected_query_tokens: list[str] = field(default_factory=list)
    selected_evidence_spans: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NextHopQueryComposer:
    """
    Compose next-hop queries from external question-token selection and evidence spans.

    Args:
        - query_token_selector: Selects original-question tokens with semantic impact and role binding.
        - max_evidence_spans: Maximum Labeler / SpanRecovery spans appended to the query.
        - max_query_chars: Maximum output query length.

    Returns:
        - NextHopQueryComposer: Replacement for model-based EfficientRAG filter generation.
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
        "source",
    }

    def __init__(
        self,
        *,
        query_token_selector: QueryTokenSelector | None = None,
        max_evidence_spans: int = 3,
        max_query_chars: int = 260,
    ) -> None:
        self.query_token_selector = query_token_selector or QueryTokenSelector()
        self.max_evidence_spans = max(1, max_evidence_spans)
        self.max_query_chars = max(80, max_query_chars)

    def build_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
        intent_plan: SearchIntentPlan | None = None,
    ) -> RAGFilterResult:
        """
        Build a RAGFilterResult-compatible next-hop query without running the filter model.

        Args:
            - question: Original task question.
            - evidence_items: Useful evidence spans from CONTINUE chunks.
            - intent_plan: Planner state carrying answer role and missing terms.

        Returns:
            - RAGFilterResult: Next-hop query and trace-compatible metadata.
        """
        selected = self.query_token_selector.select(
            question=question,
            intent_plan=intent_plan,
        )
        evidence_spans = self._select_evidence_spans(
            question=question,
            evidence_items=evidence_items,
        )
        if not evidence_spans:
            return RAGFilterResult(
                query="",
                kept_question_tokens=[],
                kept_evidence_tokens=[],
                fallback_used=False,
                metadata={
                    "method": "external_semantic_role_next_hop",
                    "filter_model_used": False,
                    "query_token_selection": selected.to_dict(),
                    "selected_query_tokens": [],
                    "selected_evidence_spans": [],
                    "filter_input": "",
                    "empty_reason": "no_selected_bridge_spans",
                },
            )
        query_tokens = (
            list(selected.role_tokens)
            if selected.role_tokens and evidence_spans
            else list(selected.selected_tokens)
        )
        query = self._compose_query(
            selected_query_tokens=query_tokens,
            evidence_spans=evidence_spans,
            intent_plan=intent_plan,
        )

        return RAGFilterResult(
            query=query,
            kept_question_tokens=list(query_tokens),
            kept_evidence_tokens=list(evidence_spans),
            fallback_used=False,
            metadata={
                "method": "external_semantic_role_next_hop",
                "filter_model_used": False,
                "query_token_selection": selected.to_dict(),
                "selected_query_tokens": list(query_tokens),
                "selected_evidence_spans": list(evidence_spans),
                "filter_input": self._debug_input(
                    question_tokens=query_tokens,
                    evidence_spans=evidence_spans,
                ),
            },
        )

    def _select_evidence_spans(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> list[str]:
        question_key = self._match_key(question)
        candidates: list[tuple[int, int, int, str]] = []
        order = 0
        for item in evidence_items:
            for span in list(item.matched_terms or []):
                cleaned = self._clean_span(span)
                if not cleaned:
                    continue
                key = self._match_key(cleaned)
                if not key:
                    continue
                new_info = 1 if key not in question_key else 0
                word_count = len(self._keywords(cleaned))
                candidates.append((new_info, min(word_count, 4), -order, cleaned))
                order += 1

        selected: list[str] = []
        seen: set[str] = set()
        has_new_information = any(item[0] > 0 for item in candidates)
        for new_info, _, _, span in sorted(candidates, reverse=True):
            if has_new_information and new_info <= 0:
                continue
            key = self._match_key(span)
            if not key or key in seen:
                continue
            if self._contained_by_existing(key, seen):
                continue
            selected.append(span)
            seen.add(key)
            if len(selected) >= self.max_evidence_spans:
                break
        return selected

    def _compose_query(
        self,
        *,
        selected_query_tokens: list[str],
        evidence_spans: list[str],
        intent_plan: SearchIntentPlan | None,
    ) -> str:
        parts: list[str] = []
        if intent_plan is not None and intent_plan.preferred_domain:
            parts.append(f"site:{intent_plan.preferred_domain}")
        parts.extend(selected_query_tokens)
        parts.extend(evidence_spans)
        return self._clean_query(" ".join(self._dedupe(parts)))

    def _debug_input(
        self,
        *,
        question_tokens: list[str],
        evidence_spans: list[str],
    ) -> str:
        return self._clean_query(
            "Query : "
            + " ".join(question_tokens)
            + " Info : "
            + " ".join(evidence_spans)
        )

    def _clean_query(self, query: str) -> str:
        text = normalize_text(query)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\b(?:Query|Info)\s*:\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" \"'`.,;:-")
        return text[: self.max_query_chars]

    def _clean_span(self, span: str) -> str:
        text = normalize_text(str(span or "")).strip(" \"'`.,;:")
        if len(text) < 3 or len(text) > 100:
            return ""
        tokens = self._keywords(text)
        if not tokens:
            return ""
        if all(token in self.STOPWORDS for token in tokens):
            return ""
        if re.fullmatch(r"[\W_]+", text):
            return ""
        return text

    def _contained_by_existing(self, key: str, seen: set[str]) -> bool:
        return any(key in existing or existing in key for existing in seen)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(str(value or "")).strip(" \"'`.,;:")
            key = self._match_key(text)
            if not text or not key or key in seen:
                continue
            result.append(text)
            seen.add(key)
        return result

    def _keywords(self, text: str) -> list[str]:
        tokens = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'_.-]{1,}",
            normalize_text(text).casefold(),
        )
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip("'_.-")
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _match_key(self, text: str) -> str:
        return " ".join(self._keywords(text))


__all__ = ["NextHopComposition", "NextHopQueryComposer"]
