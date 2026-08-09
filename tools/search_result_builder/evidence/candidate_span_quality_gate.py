from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text

from .span_role_classifier import CandidateSpan


@dataclass(frozen=True)
class CandidateSpanQualityResult:
    """
    Filtered span candidates before role classification.

    Args:
     - candidates: Candidate spans that are suitable for role classification.
     - dropped: Spans removed before classification with compact reasons.
     - diagnostics: Runtime summary for reports and debugging.

    Returns:
     - CandidateSpanQualityResult: Span quality gate output.
    """

    candidates: list[CandidateSpan] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "dropped": list(self.dropped),
            "diagnostics": dict(self.diagnostics),
        }


class CandidateSpanQualityGate:
    """
    Keep only structurally meaningful spans before SLM role classification.

    Args:
     - max_candidates: Maximum candidates passed to the role classifier.
     - max_span_chars: Maximum normalized span length.

    Returns:
     - CandidateSpanQualityGate: Lightweight candidate filtering gate.
    """

    _NUMERIC_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])"
    )
    _NUMERIC_WITH_UNIT_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*"
        r"(?:m\^?3|m3|km|kg|cm|mm|miles?|meters?|metres?|hours?|minutes?|"
        r"seconds?|albums?|episodes?|years?|%|percent|cubic\s+met(?:er|re)s?)\b",
        re.IGNORECASE,
    )
    _YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
    _URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
    _EDGE_PUNCT_RE = re.compile(
        r"^[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+|[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+$"
    )

    _PAGE_CHROME_TERMS = {
        "advertisement",
        "all rights reserved",
        "captcha",
        "cookie",
        "cookies",
        "copyright",
        "crossword solver",
        "external links",
        "home",
        "javascript",
        "login",
        "menu",
        "navigation",
        "newsletter",
        "privacy",
        "privacy policy",
        "related articles",
        "search results",
        "sign in",
        "subscribe",
        "terms of service",
        "the crossword solver",
    }
    _FRAGMENT_TERMS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "he",
        "i",
        "i'll",
        "in",
        "is",
        "it",
        "it's",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "then",
        "there",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "which",
        "who",
        "with",
        "you",
    }
    _GENERIC_SINGLE_TERMS = {
        "about",
        "article",
        "book",
        "browse",
        "category",
        "content",
        "data",
        "details",
        "download",
        "episode",
        "file",
        "full",
        "help",
        "image",
        "information",
        "list",
        "official",
        "page",
        "part",
        "results",
        "search",
        "section",
        "series",
        "source",
        "summary",
        "title",
        "topic",
        "video",
        "view",
    }

    def __init__(
        self,
        *,
        max_candidates: int = 40,
        max_span_chars: int = 160,
    ) -> None:
        # Fourth and last of the budgets bounding the same spans, after
        # PassageEvidenceUnitBuilder.max_units, RetrievalControl's max_total,
        # and the classifier's per-call bound. It has to track them: while the
        # first held rounds to 10 spans this never bound, and raising that one
        # to 40 without this simply moved the truncation here. level1_final_07
        # shows the shape -- candidate units per round rose from 8.5 to 30.2,
        # and the spans reaching the classifier still stopped dead at 15.
        self.max_candidates = max(1, max_candidates)
        self.max_span_chars = max(30, max_span_chars)

    def filter_candidates(
        self,
        candidates: list[CandidateSpan],
    ) -> CandidateSpanQualityResult:
        """
        Normalize and filter candidate spans.

        Args:
         - candidates: Grounded and expanded candidates.

        Returns:
         - CandidateSpanQualityResult: Kept and dropped candidate spans.
        """
        kept: list[CandidateSpan] = []
        dropped: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = self._normalize_span(candidate.text)
            if not normalized:
                dropped.append(self._drop(candidate, "empty_after_normalization"))
                continue
            if len(normalized) > self.max_span_chars:
                dropped.append(self._drop(candidate, "too_long", normalized))
                continue
            if self._is_fragment_or_generic(normalized):
                dropped.append(self._drop(candidate, "fragment_or_generic", normalized))
                continue

            allowed, reason = self._is_quality_span(normalized)
            if not allowed:
                dropped.append(self._drop(candidate, reason, normalized))
                continue

            key = normalized.casefold()
            if key in seen:
                dropped.append(self._drop(candidate, "duplicate_after_normalization", normalized))
                continue

            kept.append(
                CandidateSpan(
                    id=candidate.id,
                    text=normalized,
                    local_context=normalize_text(candidate.local_context),
                    source_title=normalize_text(candidate.source_title),
                )
            )
            seen.add(key)
            if len(kept) >= self.max_candidates:
                break

        reason_counts: dict[str, int] = {}
        for item in dropped:
            reason = item.get("reason", "")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        diagnostics = {
            "input_count": len(candidates),
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "drop_reason_counts": reason_counts,
            "max_candidates": self.max_candidates,
        }
        return CandidateSpanQualityResult(
            candidates=kept,
            dropped=dropped[:50],
            diagnostics=diagnostics,
        )

    def _normalize_span(self, text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = self._EDGE_PUNCT_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return normalize_text(cleaned)

    def _is_fragment_or_generic(self, span: str) -> bool:
        lowered = span.casefold()
        words = self._WORD_RE.findall(span)
        if self._NUMERIC_RE.search(span) or self._URL_RE.search(span):
            return False
        if lowered in self._FRAGMENT_TERMS:
            return True
        return len(words) == 1 and words[0].casefold() in self._GENERIC_SINGLE_TERMS

    def _is_quality_span(self, span: str) -> tuple[bool, str]:
        lowered = span.casefold()
        words = self._WORD_RE.findall(span)
        if not span:
            return False, "empty"
        if any(term in lowered for term in self._PAGE_CHROME_TERMS):
            return False, "page_chrome"
        if len(span) < 3 and not self._NUMERIC_RE.search(span):
            return False, "too_short"
        if self._NUMERIC_RE.fullmatch(span):
            return False, "bare_number"
        if self._NUMERIC_WITH_UNIT_RE.search(span) or self._YEAR_RE.search(span):
            return True, "kept"
        if self._NUMERIC_RE.search(span):
            return True, "kept"
        if len(words) >= 2:
            return True, "kept"
        if words and len(words[0]) >= 4 and words[0][:1].isupper():
            return True, "kept"
        return False, "low_information"

    def _drop(
        self,
        candidate: CandidateSpan,
        reason: str,
        normalized_text: str | None = None,
    ) -> dict[str, str]:
        return {
            "id": normalize_text(candidate.id),
            "text": normalize_text(candidate.text),
            "normalized_text": normalize_text(normalized_text or candidate.text),
            "reason": reason,
        }


__all__ = [
    "CandidateSpanQualityGate",
    "CandidateSpanQualityResult",
]
