from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import re
from typing import Any

from utils.network_utils import normalize_text

from .span_role_classifier import ANSWER_SUPPORT, BRIDGE, NOISE


@dataclass(frozen=True)
class FinalizedSpan:
    """
    Final span after role-aware cleanup.

    Args:
     - original_text: Span text classified by the role classifier.
     - finalized_text: Span text written back to evidence or next-hop.
     - role: ANSWER_SUPPORT / BRIDGE / NOISE.
     - accepted: Whether the span remains usable after finalization.
     - reason: Finalization strategy or rejection reason.

    Returns:
     - FinalizedSpan: Role-aware span finalization result.
    """

    original_text: str
    finalized_text: str
    role: str
    accepted: bool
    reason: str
    goal_id: str = ""
    candidate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleAwareSpanFinalizationResult:
    """
    Batch result for role-aware span finalization.

    Args:
     - finalized: Per-span finalization records.
     - diagnostics: Compact batch summary.

    Returns:
     - RoleAwareSpanFinalizationResult: Finalized spans and diagnostics.
    """

    finalized: list[FinalizedSpan] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finalized": [item.to_dict() for item in self.finalized],
            "diagnostics": dict(self.diagnostics),
        }


class RoleAwareSpanFinalizer:
    """
    Convert classified spans into evidence-ready or query-ready spans.

    Args:
     - max_answer_chars: Maximum answer-support context length.
     - max_bridge_chars: Maximum bridge phrase length.

    Returns:
     - RoleAwareSpanFinalizer: Role-aware span cleanup helper.
    """

    _NUMERIC_WITH_UNIT_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*"
        r"(?:m\^?3|m3|km|kg|cm|mm|miles?|meters?|metres?|hours?|minutes?|"
        r"seconds?|albums?|episodes?|years?|%|percent|cubic\s+met(?:er|re)s?)"
        r"(?:\s+(?:and\s+)?[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*"
        r"(?:hours?|minutes?|seconds?))*\b",
        re.IGNORECASE,
    )
    _BARE_NUMBER_RE = re.compile(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?")
    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
    _EDGE_PUNCT_RE = re.compile(
        r"^[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+|[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+$"
    )
    _SCRIPT_FRAGMENT_RE = re.compile(
        r"\b(?:i|you|he|she|we|they|it|me|my|your|his|her|our|their|"
        r"because|wherever|slowly|scary|never|come|well|that's|it's|i'll)\b",
        re.IGNORECASE,
    )
    _PAGE_CATEGORY_TERMS = {
        "album of the year",
        "discography",
        "songs",
        "lyrics",
        "crossword",
        "word finder",
        "scrabble",
        "anagram",
        "privacy",
        "cookie",
        "captcha",
        "navigation",
        "login",
    }
    _BRIDGE_GENERIC_TERMS = {
        "album",
        "albums",
        "song",
        "songs",
        "discography",
        "keyword",
        "keywords",
        "list",
        "page",
        "result",
        "results",
    }

    def __init__(
        self,
        *,
        max_answer_chars: int = 220,
        max_bridge_chars: int = 90,
    ) -> None:
        self.max_answer_chars = max(80, max_answer_chars)
        self.max_bridge_chars = max(20, max_bridge_chars)

    def finalize_batch(
        self,
        *,
        items: list[dict[str, str]],
        context: str,
        source_title: str = "",
    ) -> RoleAwareSpanFinalizationResult:
        """
        Finalize classified role items for a single document.

        Args:
         - items: Role classifier outputs for one document.
         - context: Document chunk text.
         - source_title: Source title for title-aware bridge cleanup.

        Returns:
         - RoleAwareSpanFinalizationResult: Finalized span records.
        """
        finalized: list[FinalizedSpan] = []
        for item in items:
            span = normalize_text(item.get("text", ""))
            role = normalize_text(item.get("role", "")).upper()
            goal_id = normalize_text(item.get("goal_id", ""))
            candidate_id = normalize_text(item.get("id", ""))
            if role == ANSWER_SUPPORT:
                finalized_item = self._finalize_answer(span, context)
            elif role == BRIDGE:
                finalized_item = self._finalize_bridge(span, context, source_title)
            else:
                finalized_item = FinalizedSpan(
                    original_text=span,
                    finalized_text="",
                    role=NOISE,
                    accepted=False,
                    reason="noise_role",
                )
            finalized.append(
                replace(
                    finalized_item,
                    goal_id=goal_id if finalized_item.role != NOISE else "",
                    candidate_id=candidate_id,
                )
            )

        reason_counts: dict[str, int] = {}
        accepted_counts: dict[str, int] = {}
        goal_counts: dict[str, int] = {}
        for item in finalized:
            reason_counts[item.reason] = reason_counts.get(item.reason, 0) + 1
            if item.accepted:
                accepted_counts[item.role] = accepted_counts.get(item.role, 0) + 1
                if item.goal_id:
                    goal_counts[item.goal_id] = goal_counts.get(item.goal_id, 0) + 1
        return RoleAwareSpanFinalizationResult(
            finalized=finalized,
            diagnostics={
                "input_count": len(items),
                "accepted_count": sum(1 for item in finalized if item.accepted),
                "reason_counts": reason_counts,
                "accepted_role_counts": accepted_counts,
                "goal_assignment_counts": goal_counts,
            },
        )

    def _finalize_answer(self, span: str, context: str) -> FinalizedSpan:
        cleaned = self._clean(span)
        if not cleaned:
            return self._reject(span, ANSWER_SUPPORT, "empty_answer_span")
        if self._NUMERIC_WITH_UNIT_RE.search(cleaned):
            phrase = self._containing_clause(cleaned, context, self.max_answer_chars)
            return FinalizedSpan(
                original_text=span,
                finalized_text=phrase or cleaned,
                role=ANSWER_SUPPORT,
                accepted=True,
                reason="answer_numeric_context",
            )
        phrase = self._containing_clause(cleaned, context, self.max_answer_chars)
        return FinalizedSpan(
            original_text=span,
            finalized_text=phrase or cleaned,
            role=ANSWER_SUPPORT,
            accepted=True,
            reason="answer_context",
        )

    def _finalize_bridge(
        self,
        span: str,
        context: str,
        source_title: str,
    ) -> FinalizedSpan:
        cleaned = self._clean(span)
        if not cleaned:
            return self._reject(span, BRIDGE, "empty_bridge_span")
        if self._NUMERIC_WITH_UNIT_RE.search(cleaned):
            return self._reject(span, BRIDGE, "answer_like_numeric_unit")
        if self._BARE_NUMBER_RE.fullmatch(cleaned):
            return self._reject(span, BRIDGE, "bare_number")
        if len(cleaned) > self.max_bridge_chars:
            return self._reject(span, BRIDGE, "bridge_too_long")
        if self._looks_like_script_fragment(cleaned):
            return self._reject(span, BRIDGE, "script_fragment")

        title_bridge = self._title_bridge_phrase(cleaned, source_title)
        if title_bridge:
            cleaned = title_bridge
        if self._looks_like_page_category(cleaned):
            return self._reject(span, BRIDGE, "page_category")
        if self._is_generic_bridge(cleaned):
            return self._reject(span, BRIDGE, "generic_bridge")
        return FinalizedSpan(
            original_text=span,
            finalized_text=cleaned,
            role=BRIDGE,
            accepted=True,
            reason="bridge_query_phrase",
        )

    def _title_bridge_phrase(self, span: str, source_title: str) -> str:
        title = normalize_text(source_title)
        if not title or span.casefold() not in title.casefold():
            return ""
        segments = [
            self._clean(segment)
            for segment in re.split(r"\s[-|:]\s|\s[|]\s", title)
        ]
        useful_segments = [
            segment
            for segment in segments
            if segment
            and span.casefold() in segment.casefold()
            and not self._looks_like_page_category(segment)
            and len(segment) <= self.max_bridge_chars
        ]
        if useful_segments:
            return useful_segments[0]
        return ""

    def _containing_clause(self, span: str, context: str, max_chars: int) -> str:
        text = normalize_text(context)
        if not text:
            return ""
        index = text.casefold().find(span.casefold())
        if index < 0:
            return ""
        start = max(
            text.rfind(".", 0, index),
            text.rfind(";", 0, index),
            text.rfind(":", 0, index),
            text.rfind("\n", 0, index),
        )
        start = 0 if start < 0 else start + 1
        end_candidates = [
            pos
            for pos in (
                text.find(".", index + len(span)),
                text.find(";", index + len(span)),
                text.find("\n", index + len(span)),
            )
            if pos >= 0
        ]
        end = min(end_candidates) + 1 if end_candidates else len(text)
        phrase = self._clean(text[start:end])
        if len(phrase) <= max_chars:
            return phrase
        left = max(0, index - max_chars // 3)
        right = min(len(text), index + len(span) + (max_chars * 2 // 3))
        return self._clean(text[left:right])[:max_chars]

    def _looks_like_script_fragment(self, span: str) -> bool:
        words = self._WORD_RE.findall(span)
        if len(words) >= 4 and span[:1].islower():
            return True
        if len(words) >= 5 and self._SCRIPT_FRAGMENT_RE.search(span):
            return True
        if len(words) >= 8 and self._SCRIPT_FRAGMENT_RE.search(span):
            return True
        if len(words) <= 3 and self._SCRIPT_FRAGMENT_RE.search(span):
            return True
        return False

    def _looks_like_page_category(self, span: str) -> bool:
        lowered = span.casefold()
        return any(term in lowered for term in self._PAGE_CATEGORY_TERMS)

    def _is_generic_bridge(self, span: str) -> bool:
        words = [word.casefold() for word in self._WORD_RE.findall(span)]
        if not words:
            return True
        return all(word in self._BRIDGE_GENERIC_TERMS for word in words)

    def _clean(self, text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = self._EDGE_PUNCT_RE.sub("", cleaned).strip()
        return normalize_text(cleaned)

    def _reject(self, span: str, role: str, reason: str) -> FinalizedSpan:
        return FinalizedSpan(
            original_text=normalize_text(span),
            finalized_text="",
            role=role,
            accepted=False,
            reason=reason,
        )


__all__ = [
    "FinalizedSpan",
    "RoleAwareSpanFinalizationResult",
    "RoleAwareSpanFinalizer",
]
