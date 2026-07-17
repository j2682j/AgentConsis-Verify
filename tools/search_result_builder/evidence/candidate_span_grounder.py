from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text

from .span_role_classifier import CandidateSpan


_SPACY_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class GroundedCandidateSpan:
    """
    Span candidate aligned to its local context and expanded to a phrase.

    Args:
     - candidate: Expanded candidate span for the next pipeline stage.
     - original_text: Original text before grounding and expansion.
     - grounded: Whether the span was found in local_context.
     - start: Start offset in normalized local_context.
     - end: End offset in normalized local_context.
     - expansion_reason: Expansion strategy used for this span.
     - diagnostics: Compact details for experiment logs.

    Returns:
     - GroundedCandidateSpan: Grounded and expanded span candidate.
    """

    candidate: CandidateSpan
    original_text: str
    grounded: bool
    start: int = -1
    end: int = -1
    expansion_reason: str = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "original_text": self.original_text,
            "grounded": self.grounded,
            "start": self.start,
            "end": self.end,
            "expansion_reason": self.expansion_reason,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CandidateSpanGroundingResult:
    """
    Batch result for candidate span grounding and expansion.

    Args:
     - candidates: Expanded CandidateSpan objects.
     - grounded_spans: Per-span grounding details.
     - diagnostics: Batch summary for reports.

    Returns:
     - CandidateSpanGroundingResult: Expanded candidates and diagnostics.
    """

    candidates: list[CandidateSpan] = field(default_factory=list)
    grounded_spans: list[GroundedCandidateSpan] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "grounded_spans": [span.to_dict() for span in self.grounded_spans],
            "diagnostics": dict(self.diagnostics),
        }


class CandidateSpanExpander:
    """
    Expand grounded spans into short phrase-level units.

    Args:
     - spacy_model: spaCy model used for entity and noun-chunk expansion.
     - max_span_chars: Maximum expanded span length.
     - max_context_chars: Maximum context scanned by spaCy.

    Returns:
     - CandidateSpanExpander: Phrase expansion helper.
    """

    _NUMERIC_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])"
    )
    _NUMERIC_WITH_UNIT_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*"
        r"(?:m\^?3|m3|km|kg|cm|mm|miles?|meters?|metres?|hours?|minutes?|"
        r"seconds?|albums?|episodes?|years?|%|percent|cubic\s+met(?:er|re)s?)"
        r"(?:\s+(?:and\s+)?[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*"
        r"(?:hours?|minutes?|seconds?))*\b",
        re.IGNORECASE,
    )
    _QUOTE_RE = re.compile(r'"([^"]{3,160})"|\'([^\']{3,160})\'')
    _ENTITY_LABELS = {
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    }

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_sm",
        max_span_chars: int = 160,
        max_context_chars: int = 4000,
    ) -> None:
        self.spacy_model = spacy_model
        self.max_span_chars = max(30, max_span_chars)
        self.max_context_chars = max(500, max_context_chars)

    def expand(
        self,
        *,
        span: str,
        context: str,
        source_title: str = "",
        start: int = -1,
        end: int = -1,
    ) -> tuple[str, str]:
        text = normalize_text(span)
        cleaned_context = normalize_text(context)
        if not text or start < 0 or end < 0 or not cleaned_context:
            return text, "none"

        expanded = self._expand_numeric_unit(text, cleaned_context)
        if expanded != text:
            return expanded, "numeric_unit"
        if self._NUMERIC_RE.fullmatch(text):
            return text, "bare_number"

        expanded = self._expand_source_title_phrase(text, source_title)
        if expanded != text:
            return expanded, "source_title_phrase"

        expanded = self._expand_quoted_phrase(text, cleaned_context, start, end)
        if expanded != text:
            return expanded, "quoted_title"

        expanded = self._expand_spacy_span(text, cleaned_context, start, end)
        if expanded != text:
            return expanded

        return text, "none"

    def _expand_numeric_unit(self, span: str, context: str) -> str:
        if not self._NUMERIC_RE.search(span):
            return span
        for match in self._NUMERIC_WITH_UNIT_RE.finditer(context):
            value = normalize_text(match.group(0))
            if span.casefold() in value.casefold() and len(value) <= self.max_span_chars:
                return value
        return span

    def _expand_source_title_phrase(self, span: str, source_title: str) -> str:
        title = normalize_text(source_title)
        if not title or span.casefold() not in title.casefold():
            return span
        segments = re.split(r"\s[-|:]\s|\s[|]\s", title)
        for segment in segments:
            cleaned = normalize_text(segment)
            if span.casefold() in cleaned.casefold() and 3 <= len(cleaned) <= self.max_span_chars:
                return cleaned
        return span

    def _expand_quoted_phrase(
        self,
        span: str,
        context: str,
        start: int,
        end: int,
    ) -> str:
        for match in self._QUOTE_RE.finditer(context):
            quote_start, quote_end = match.span()
            if quote_start <= start and quote_end >= end:
                phrase = normalize_text(match.group(1) or match.group(2) or "")
                if 3 <= len(phrase) <= self.max_span_chars:
                    return phrase
        return span

    def _expand_spacy_span(
        self,
        span: str,
        context: str,
        start: int,
        end: int,
    ) -> tuple[str, str]:
        nlp = self._load_spacy()
        if not nlp:
            return span, "none"
        try:
            doc = nlp(context[: self.max_context_chars])
        except Exception:
            return span, "none"

        best_entity = self._best_covering_span(
            [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents],
            start,
            end,
            entity_only=True,
        )
        if best_entity:
            return context[best_entity[0] : best_entity[1]], "ner_entity"

        try:
            noun_chunks = [
                (chunk.start_char, chunk.end_char, "NOUN_CHUNK")
                for chunk in doc.noun_chunks
            ]
        except Exception:
            noun_chunks = []
        best_chunk = self._best_covering_span(
            noun_chunks,
            start,
            end,
            entity_only=False,
        )
        if best_chunk:
            return context[best_chunk[0] : best_chunk[1]], "noun_chunk"
        return span, "none"

    def _best_covering_span(
        self,
        spans: list[tuple[int, int, str]],
        start: int,
        end: int,
        *,
        entity_only: bool,
    ) -> tuple[int, int, str] | None:
        choices: list[tuple[int, int, str]] = []
        for item_start, item_end, label in spans:
            if entity_only and label not in self._ENTITY_LABELS:
                continue
            if item_start <= start and item_end >= end:
                length = item_end - item_start
                if 2 <= length <= self.max_span_chars:
                    choices.append((item_start, item_end, label))
        if not choices:
            return None
        return min(choices, key=lambda item: item[1] - item[0])

    def _load_spacy(self) -> Any | None:
        if self.spacy_model in _SPACY_CACHE:
            return _SPACY_CACHE[self.spacy_model]
        try:
            import spacy  # type: ignore

            nlp = spacy.load(self.spacy_model)
        except Exception:
            try:
                import spacy  # type: ignore

                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = None
        _SPACY_CACHE[self.spacy_model] = nlp
        return nlp


class CandidateSpanGrounder:
    """
    Ground candidate spans in context and expand them before quality filtering.

    Args:
     - expander: Optional phrase expander.

    Returns:
     - CandidateSpanGrounder: Candidate grounding coordinator.
    """

    _EDGE_PUNCT_RE = re.compile(
        r"^[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+|[\s\"'`“”‘’()\[\]{}<>:;,.!?|/\\-]+$"
    )

    def __init__(
        self,
        *,
        expander: CandidateSpanExpander | None = None,
    ) -> None:
        self.expander = expander or CandidateSpanExpander()

    def expand_candidates(
        self,
        candidates: list[CandidateSpan],
    ) -> CandidateSpanGroundingResult:
        grounded_spans: list[GroundedCandidateSpan] = []
        output_candidates: list[CandidateSpan] = []
        reason_counts: dict[str, int] = {}
        grounded_count = 0
        expanded_count = 0

        for candidate in candidates:
            original = self._normalize_span(candidate.text)
            context = normalize_text(candidate.local_context)
            source_title = normalize_text(candidate.source_title)
            start, end, method = self._ground(original, context)
            grounded = start >= 0 and end >= 0
            if grounded:
                grounded_count += 1
                expanded, reason = self.expander.expand(
                    span=original,
                    context=context,
                    source_title=source_title,
                    start=start,
                    end=end,
                )
            else:
                expanded, reason = original, "ungrounded"

            expanded = self._normalize_span(expanded)
            if expanded and expanded.casefold() != original.casefold():
                expanded_count += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            expanded_candidate = CandidateSpan(
                id=candidate.id,
                text=expanded or original,
                local_context=context,
                source_title=source_title,
            )
            output_candidates.append(expanded_candidate)
            grounded_spans.append(
                GroundedCandidateSpan(
                    candidate=expanded_candidate,
                    original_text=original,
                    grounded=grounded,
                    start=start,
                    end=end,
                    expansion_reason=reason,
                    diagnostics={
                        "grounding_method": method,
                        "expanded": expanded.casefold() != original.casefold(),
                    },
                )
            )

        diagnostics = {
            "input_count": len(candidates),
            "grounded_count": grounded_count,
            "ungrounded_count": len(candidates) - grounded_count,
            "expanded_count": expanded_count,
            "expansion_reason_counts": reason_counts,
            "examples": [
                {
                    "id": span.candidate.id,
                    "original": span.original_text,
                    "expanded": span.candidate.text,
                    "grounded": span.grounded,
                    "reason": span.expansion_reason,
                }
                for span in grounded_spans[:20]
            ],
        }
        return CandidateSpanGroundingResult(
            candidates=output_candidates,
            grounded_spans=grounded_spans,
            diagnostics=diagnostics,
        )

    def _ground(self, span: str, context: str) -> tuple[int, int, str]:
        if not span or not context:
            return -1, -1, "empty"
        index = context.find(span)
        if index >= 0:
            return index, index + len(span), "exact"
        index = context.casefold().find(span.casefold())
        if index >= 0:
            return index, index + len(span), "case_insensitive"

        normalized_span = self._normalize_for_match(span)
        if not normalized_span:
            return -1, -1, "unmatched"
        for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'._-]*", context):
            token = match.group(0)
            if self._normalize_for_match(token) == normalized_span:
                return match.start(), match.end(), "token_normalized"
        return -1, -1, "unmatched"

    def _normalize_span(self, text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = self._EDGE_PUNCT_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return normalize_text(cleaned)

    def _normalize_for_match(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", normalize_text(text).casefold())


__all__ = [
    "CandidateSpanExpander",
    "CandidateSpanGrounder",
    "CandidateSpanGroundingResult",
    "GroundedCandidateSpan",
]
