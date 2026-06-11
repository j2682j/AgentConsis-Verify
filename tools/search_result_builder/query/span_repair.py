from __future__ import annotations

import re
from dataclasses import dataclass

from utils.network_utils import normalize_text

from .semantic_impact import SemanticImpactScorer, TokenSalient


@dataclass
class SalientSpan:
    """
    Repaired phrase span built from high-impact tokens.

    Args:
        - text: Original question text covered by the repaired span.
        - start: Character start offset in the question.
        - end: Character end offset in the question.
        - score: Aggregated salience score from source tokens.
        - tokens: Tokenizer tokens that triggered the span.
        - token_indices: Token positions that triggered the span.

    Returns:
        - SalientSpan: Search-oriented salient phrase.
    """

    text: str
    start: int
    end: int
    score: float
    tokens: list[str]
    token_indices: list[int]


class SpanRepairer:
    """
    Merge and repair token-level salience into query-ready spans.

    Args:
        - max_salient_spans: Maximum number of spans to keep.
        - merge_gap_chars: Maximum gap used when merging nearby salient tokens.
        - min_token_chars: Minimum span length unless numeric content exists.

    Returns:
        - SpanRepairer: Span repair and selection helper.
    """

    WEAK_SINGLE_TERMS = {
        "algebra",
        "algebraic",
        "attached",
        "base",
        "camera",
        "day",
        "doesn",
        "document",
        "each",
        "exchange",
        "guarantee",
        "guarantees",
        "here",
        "line",
        "rest",
        "round",
        "selected",
        "studie",
    }
    PHRASE_EDGE_TERMS = {
        "about",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    MONTH_PATTERN = (
        r"January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
        r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
    )
    PUNCTUATION_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

    def __init__(
        self,
        *,
        max_salient_spans: int = 5,
        merge_gap_chars: int = 2,
        min_token_chars: int = 2,
    ) -> None:
        self.max_salient_spans = max_salient_spans
        self.merge_gap_chars = merge_gap_chars
        self.min_token_chars = min_token_chars
        self.stopwords = SemanticImpactScorer.STOPWORDS
        self.generic_query_terms = SemanticImpactScorer.GENERIC_QUERY_TERMS

    def build_spans(
        self,
        question: str,
        tokens: list[TokenSalient],
    ) -> list[SalientSpan]:
        """
        Merge, repair, deduplicate, and select final salient spans.

        Args:
            - question: Original question text.
            - tokens: Filtered salient tokens.

        Returns:
            - list[SalientSpan]: Query-ready salient spans.
        """
        return self.select_top_spans(self.merge_salient_tokens(question, tokens))

    def merge_salient_tokens(
        self,
        question: str,
        tokens: list[TokenSalient],
    ) -> list[SalientSpan]:
        """
        Merge adjacent high-impact tokens into readable phrase spans.

        Args:
            - question: Original question text.
            - tokens: Filtered salient tokens.

        Returns:
            - list[SalientSpan]: Repaired salient spans before final top-k selection.
        """
        if not tokens:
            return []

        ordered = sorted(tokens, key=lambda item: (item.start, item.end))
        groups: list[list[TokenSalient]] = []
        current: list[TokenSalient] = []
        for token in ordered:
            if not current:
                current = [token]
                continue
            previous = current[-1]
            gap_text = question[previous.end : token.start]
            gap_ok = (
                token.start - previous.end <= self.merge_gap_chars
                or re.fullmatch(r"[\s._:/-]*", gap_text or "") is not None
            )
            if gap_ok:
                current.append(token)
            else:
                groups.append(current)
                current = [token]
        if current:
            groups.append(current)

        spans: list[SalientSpan] = []
        for group in groups:
            start = min(token.start for token in group)
            end = max(token.end for token in group)
            start, end, text = self._repair_span(question, start, end)
            if not self._valid_span_text(text):
                continue
            spans.append(
                SalientSpan(
                    text=text,
                    start=start,
                    end=end,
                    score=round(sum(token.score for token in group), 6),
                    tokens=[token.token for token in group],
                    token_indices=[token.token_index for token in group],
                )
            )
        return self._dedupe_spans(spans)

    def select_top_spans(self, spans: list[SalientSpan]) -> list[SalientSpan]:
        """
        Select the highest-impact non-contained spans.

        Args:
            - spans: Repaired salient spans.

        Returns:
            - list[SalientSpan]: Final top-k spans for query generation.
        """
        deduped = self._dedupe_spans(spans)
        deduped.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        selected: list[SalientSpan] = []
        for span in deduped:
            span_key = self._normalize_for_match(span.text)
            if not span_key:
                continue
            contained = False
            for existing in list(selected):
                existing_key = self._normalize_for_match(existing.text)
                if span_key in existing_key and existing.score >= span.score:
                    contained = True
                    break
                if existing_key in span_key and span.score >= existing.score:
                    selected.remove(existing)
                    break
            if not contained:
                selected.append(span)
            if len(selected) >= self.max_salient_spans:
                break
        return selected

    def _valid_span_text(self, text: str) -> bool:
        cleaned = normalize_text(text).strip(" ,.;:!?()[]{}")
        if len(cleaned) < self.min_token_chars:
            return False
        lowered = cleaned.lower()
        if lowered in self.stopwords or lowered in self.generic_query_terms:
            return False
        if self.PUNCTUATION_RE.fullmatch(cleaned):
            return False
        if re.fullmatch(r"\d{1,2}", cleaned):
            return False
        if len(cleaned) < 4 and not any(char.isdigit() for char in cleaned):
            return False
        if lowered in self.WEAK_SINGLE_TERMS:
            return False
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", cleaned)
        if len(words) == 1:
            word = words[0]
            if (
                word.lower() in self.WEAK_SINGLE_TERMS
                or word.lower() in self.stopwords
                or word.lower() in self.generic_query_terms
            ):
                return False
            if len(word) <= 3 and not (word.isupper() or any(char.isdigit() for char in word)):
                return False
        return True

    def _repair_span(self, question: str, start: int, end: int) -> tuple[int, int, str]:
        start, end = self._expand_word_boundaries(question, start, end)

        quoted = self._quoted_span_containing(question, start, end)
        if quoted is not None:
            q_start, q_end = quoted
            if self._is_reasonable_expansion(question, start, end, q_start, q_end, max_chars=140):
                start, end = q_start, q_end

        start, end = self._expand_date_phrase(question, start, end)
        start, end = self._expand_capitalized_phrase(question, start, end)
        start, end = self._expand_domain_phrase(question, start, end)
        start, end = self._trim_span_edges(question, start, end)
        text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
        return start, end, text

    def _expand_word_boundaries(self, question: str, start: int, end: int) -> tuple[int, int]:
        start = max(0, start)
        end = min(len(question), end)
        while start > 0 and self._is_word_char(question[start - 1]):
            start -= 1
        while end < len(question) and self._is_word_char(question[end]):
            end += 1
        return start, end

    def _is_word_char(self, char: str) -> bool:
        return bool(re.match(r"[A-Za-z0-9_'-]", char))

    def _quoted_span_containing(self, question: str, start: int, end: int) -> tuple[int, int] | None:
        left = question.rfind('"', 0, start + 1)
        if left < 0:
            return None
        right = question.find('"', max(end, left + 1))
        if right < 0:
            return None
        if left <= start and end <= right + 1:
            return left + 1, right
        return None

    def _expand_date_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        patterns = [
            rf"\b(?:{self.MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}\b",
            rf"\b(?:{self.MONTH_PATTERN})\s+\d{{4}}\b",
            r"\b\d{4}\s*(?:-|to|and)\s*\d{4}\b",
        ]
        return self._expand_by_patterns(question, start, end, patterns)

    def _expand_capitalized_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        word = r"[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|[A-Z]{2,}[A-Za-z0-9-]*|\d+"
        connector = r"(?:of|the|and|or|for|to|in|on|at|from|with|by)"
        pattern = rf"\b{word}(?:(?:\s+{connector})?\s+{word})*\b"
        return self._expand_by_patterns(question, start, end, [pattern], max_chars=90)

    def _expand_domain_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        patterns = [
            r"\bWord\s+of\s+the\s+Day\b",
            r"\bFeatured\s+Article\b",
            r"\bofficial\s+script\b",
            r"\bminimum\s+perigee\b",
            r"\bSeries\s+\d+,\s*Episode\s+\d+\b",
            r"\b[a-zA-Z0-9_-]{8,}\b",
        ]
        return self._expand_by_patterns(question, start, end, patterns, max_chars=80)

    def _expand_by_patterns(
        self,
        question: str,
        start: int,
        end: int,
        patterns: list[str],
        *,
        max_chars: int = 120,
    ) -> tuple[int, int]:
        best_start, best_end = start, end
        for pattern in patterns:
            for match in re.finditer(pattern, question):
                m_start, m_end = match.span()
                if not self._overlaps(start, end, m_start, m_end):
                    continue
                if not self._is_reasonable_expansion(question, start, end, m_start, m_end, max_chars=max_chars):
                    continue
                if m_end - m_start > best_end - best_start:
                    best_start, best_end = m_start, m_end
        return best_start, best_end

    def _trim_span_edges(self, question: str, start: int, end: int) -> tuple[int, int]:
        edge_terms = "|".join(sorted(self.PHRASE_EDGE_TERMS))
        while start < end:
            segment = question[start:end]
            match = re.match(rf"^\W*(?:{edge_terms})\b\s*", segment, flags=re.IGNORECASE)
            if not match:
                break
            start += match.end()

        while start < end:
            segment = question[start:end]
            match = re.search(rf"\s+\b(?:{edge_terms})\b\W*$", segment, flags=re.IGNORECASE)
            if not match:
                break
            end = start + match.start()
        return start, end

    def _overlaps(self, start: int, end: int, other_start: int, other_end: int) -> bool:
        return start < other_end and other_start < end

    def _is_reasonable_expansion(
        self,
        question: str,
        start: int,
        end: int,
        candidate_start: int,
        candidate_end: int,
        *,
        max_chars: int,
    ) -> bool:
        if candidate_start > start or candidate_end < end:
            return False
        candidate = normalize_text(question[candidate_start:candidate_end])
        if len(candidate) > max_chars:
            return False
        if candidate.count(" ") > 18:
            return False
        return True

    def _dedupe_spans(self, spans: list[SalientSpan]) -> list[SalientSpan]:
        best_by_key: dict[str, SalientSpan] = {}
        for span in spans:
            key = self._normalize_for_match(span.text)
            if not key:
                continue
            existing = best_by_key.get(key)
            if existing is None or span.score > existing.score:
                best_by_key[key] = span
        return list(best_by_key.values())

    def _normalize_for_match(self, value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower())
        return f" {' '.join(cleaned.split())} "


__all__ = ["SalientSpan", "SpanRepairer"]
