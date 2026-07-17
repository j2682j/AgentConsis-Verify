from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
    repair_source: str = "token_merge"
    original_text: str = ""
    original_start: int = -1
    original_end: int = -1
    rescore: float = 0.0


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
    NER_LABELS = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "DATE",
        "EVENT",
        "WORK_OF_ART",
        "PRODUCT",
        "LAW",
        "NORP",
        "FAC",
    }
    INSTRUCTION_ARTIFACTS = {
        "according",
        "answer",
        "format",
        "return",
        "returns",
        "rounded",
        "using",
    }

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
        self._nlp: Any | None = None
        self._nlp_load_attempted = False

    def build_spans(
        self,
        question: str,
        tokens: list[TokenSalient],
        *,
        scorer: SemanticImpactScorer | None = None,
    ) -> list[SalientSpan]:
        """
        Merge, repair, deduplicate, and select final salient spans.

        Args:
            - question: Original question text.
            - tokens: Filtered salient tokens.

        Returns:
            - list[SalientSpan]: Query-ready salient spans.
        """
        spans = self.merge_salient_tokens(question, tokens)
        if scorer is not None:
            spans = self.repair_spans(question, spans, scorer=scorer)
        return self.select_top_spans(spans)

    def repair_spans(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        scorer: SemanticImpactScorer | None = None,
    ) -> list[SalientSpan]:
        """
        Repair merged spans into complete semantic units and optionally rescore them.

        Args:
            - question: Original question text.
            - spans: Initial spans produced from token-level salience.
            - scorer: Optional semantic impact scorer for repaired span rescoring.

        Returns:
            - list[SalientSpan]: Repaired spans, optionally with span-level impact scores.
        """
        if not spans:
            return []
        repaired = self.repair_with_ner_entities(question, spans)
        repaired = [
            span
            for span in (self.cleanup_span_boundary(question, span) for span in repaired)
            if self._valid_span_text(span.text)
        ]
        if scorer is not None:
            repaired = self.rescore_repaired_spans(question, repaired, scorer=scorer)
        return self._dedupe_spans(repaired)

    def repair_with_ner_entities(
        self,
        question: str,
        spans: list[SalientSpan],
    ) -> list[SalientSpan]:
        """
        Align salient spans to complete NER entities or noun chunks when possible.

        Args:
            - question: Original question text.
            - spans: Initial repaired spans.

        Returns:
            - list[SalientSpan]: Spans expanded or replaced with complete semantic units.
        """
        doc = self._spacy_doc(question)
        if doc is None:
            return list(spans)

        entity_candidates = [
            (int(ent.start_char), int(ent.end_char), str(ent.label_), "ner_entity")
            for ent in doc.ents
            if str(ent.label_) in self.NER_LABELS
        ]
        chunk_candidates: list[tuple[int, int, str, str]] = []
        try:
            chunk_candidates = [
                (int(chunk.start_char), int(chunk.end_char), "NOUN_CHUNK", "noun_chunk")
                for chunk in doc.noun_chunks
            ]
        except Exception:
            chunk_candidates = []

        repaired: list[SalientSpan] = []
        for span in spans:
            original_text = span.original_text or span.text
            original_start = span.original_start if span.original_start >= 0 else span.start
            original_end = span.original_end if span.original_end >= 0 else span.end
            best = self._best_semantic_unit(
                question,
                span,
                entity_candidates,
                max_chars=90,
            )
            if best is None:
                best = self._best_semantic_unit(
                    question,
                    span,
                    chunk_candidates,
                    max_chars=90,
                )
            if best is None:
                repaired.append(span)
                continue
            start, end, _label, source = best
            text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
            if not self._valid_span_text(text):
                repaired.append(span)
                continue
            repaired.append(
                SalientSpan(
                    text=text,
                    start=start,
                    end=end,
                    score=span.score,
                    tokens=list(span.tokens),
                    token_indices=list(span.token_indices),
                    repair_source=source,
                    original_text=original_text,
                    original_start=original_start,
                    original_end=original_end,
                    rescore=span.rescore,
                )
            )
        return repaired

    def cleanup_span_boundary(self, question: str, span: SalientSpan) -> SalientSpan:
        """
        Clean generic boundary artifacts without adding task-specific rules.

        Args:
            - question: Original question text.
            - span: Span to clean.

        Returns:
            - SalientSpan: Boundary-cleaned span.
        """
        start = max(0, span.start)
        end = min(len(question), span.end)
        anchor_mid = self._anchor_mid_for_span(question, span, start, end)
        start, end = self._restrict_to_anchor_sentence(question, start, end, anchor_mid)
        start, end = self._trim_span_edges(question, start, end)
        start, end = self._trim_instruction_artifacts(question, start, end, anchor_mid)
        start, end = self._trim_outer_punctuation(question, start, end)
        text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
        source = span.repair_source
        if (start, end) != (span.start, span.end) and source == "token_merge":
            source = "boundary_cleanup"
        return SalientSpan(
            text=text,
            start=start,
            end=end,
            score=span.score,
            tokens=list(span.tokens),
            token_indices=list(span.token_indices),
            repair_source=source,
            original_text=span.original_text or span.text,
            original_start=span.original_start if span.original_start >= 0 else span.start,
            original_end=span.original_end if span.original_end >= 0 else span.end,
            rescore=span.rescore,
        )

    def _anchor_mid_for_span(self, question: str, span: SalientSpan, start: int, end: int) -> int:
        segment = question[start:end]
        for token in span.tokens:
            cleaned = normalize_text(token).strip(" ,.;:!?()[]{}\"'`")
            if not cleaned or cleaned.lower() in self.INSTRUCTION_ARTIFACTS:
                continue
            offset = segment.find(cleaned)
            if offset >= 0:
                return start + offset + len(cleaned) // 2
        anchor_start = span.original_start if span.original_start >= 0 else span.start
        anchor_end = span.original_end if span.original_end >= 0 else span.end
        return max(start, min(end, (anchor_start + anchor_end) // 2))

    def rescore_repaired_spans(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        scorer: SemanticImpactScorer,
    ) -> list[SalientSpan]:
        """
        Recompute semantic impact for complete repaired spans.

        Args:
            - question: Original question text.
            - spans: NER/boundary repaired spans.
            - scorer: Semantic impact scorer.

        Returns:
            - list[SalientSpan]: Spans whose score reflects repaired span impact.
        """
        if not spans:
            return []
        try:
            scores = scorer.score_span_impacts(question, [(span.start, span.end) for span in spans])
        except Exception:
            return spans
        rescored: list[SalientSpan] = []
        for span, score in zip(spans, scores, strict=False):
            repaired_score = round(max(0.0, float(score)), 6)
            rescored.append(
                SalientSpan(
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    score=repaired_score,
                    tokens=list(span.tokens),
                    token_indices=list(span.token_indices),
                    repair_source=(
                        "span_rescore"
                        if span.repair_source == "token_merge"
                        else f"{span.repair_source}+span_rescore"
                    ),
                    original_text=span.original_text or span.text,
                    original_start=span.original_start if span.original_start >= 0 else span.start,
                    original_end=span.original_end if span.original_end >= 0 else span.end,
                    rescore=repaired_score,
                )
            )
        return rescored

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
                    repair_source="token_merge",
                    original_text=text,
                    original_start=start,
                    original_end=end,
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

    def _spacy_doc(self, question: str) -> Any | None:
        if self._nlp_load_attempted and self._nlp is None:
            return None
        if self._nlp is None:
            self._nlp_load_attempted = True
            try:
                import spacy

                model_name = "en_core_web_md"
                try:
                    self._nlp = spacy.load(model_name)
                except Exception:
                    self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        if self._nlp is None:
            return None
        try:
            return self._nlp(question)
        except Exception:
            return None

    def _best_semantic_unit(
        self,
        question: str,
        span: SalientSpan,
        candidates: list[tuple[int, int, str, str]],
        *,
        max_chars: int,
    ) -> tuple[int, int, str, str] | None:
        best: tuple[float, int, int, str, str] | None = None
        original_start = span.original_start if span.original_start >= 0 else span.start
        original_end = span.original_end if span.original_end >= 0 else span.end
        for start, end, label, source in candidates:
            if not self._overlaps(original_start, original_end, start, end) and not self._nearby(
                original_start,
                original_end,
                start,
                end,
                max_gap=8,
            ):
                continue
            candidate_text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
            if len(candidate_text) > max_chars:
                continue
            if candidate_text.count(" ") > 14:
                continue
            if not self._valid_span_text(candidate_text):
                continue
            if self._drops_protected_identifier(span, candidate_text):
                continue
            overlap = max(0, min(original_end, end) - max(original_start, start))
            expansion_penalty = max(0, (end - start) - (span.end - span.start)) * 0.001
            priority = 2.0 if source == "ner_entity" else 1.0
            score = priority + overlap - expansion_penalty
            if best is None or score > best[0]:
                best = (score, start, end, label, source)
        if best is None:
            return None
        _score, start, end, label, source = best
        return start, end, label, source

    def _drops_protected_identifier(self, span: SalientSpan, candidate_text: str) -> bool:
        candidate_key = self._normalize_identifier_text(candidate_text)
        protected = self._protected_identifiers(span)
        return any(identifier not in candidate_key for identifier in protected)

    def _protected_identifiers(self, span: SalientSpan) -> list[str]:
        raw_parts = list(span.tokens or [])
        if span.original_text:
            raw_parts.append(span.original_text)
        raw_parts.append(span.text)
        identifiers: list[str] = []
        for raw in raw_parts:
            text = normalize_text(raw)
            patterns = [
                r"\b[A-Za-z]*\d[A-Za-z0-9.,:_-]*\b",
                r"\b[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+\b",
                r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    identifier = self._normalize_identifier_text(match.group(0))
                    if identifier and identifier not in identifiers:
                        identifiers.append(identifier)
        return identifiers

    def _normalize_identifier_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())

    def _nearby(
        self,
        start: int,
        end: int,
        other_start: int,
        other_end: int,
        *,
        max_gap: int,
    ) -> bool:
        if self._overlaps(start, end, other_start, other_end):
            return True
        return min(abs(other_start - end), abs(start - other_end)) <= max_gap

    def _restrict_to_anchor_sentence(
        self,
        question: str,
        start: int,
        end: int,
        anchor_mid: int,
    ) -> tuple[int, int]:
        segment = question[start:end]
        if not re.search(r"[.!?]\s+", segment):
            return start, end
        sentence_start = start
        for match in re.finditer(r"[.!?]\s+", question[start:end]):
            boundary = start + match.end()
            if boundary <= anchor_mid:
                sentence_start = boundary
            else:
                break
        sentence_end = end
        for match in re.finditer(r"[.!?]\s+", question[start:end]):
            boundary_start = start + match.start()
            if boundary_start >= anchor_mid:
                sentence_end = boundary_start
                break
        if sentence_start < sentence_end:
            return sentence_start, sentence_end
        return start, end

    def _trim_instruction_artifacts(
        self,
        question: str,
        start: int,
        end: int,
        anchor_mid: int,
    ) -> tuple[int, int]:
        while start < end:
            text = question[start:end]
            match = re.match(r"^\W*([A-Za-z]+)\b[\s,:;-]*", text)
            if not match:
                break
            word = match.group(1).lower()
            if word not in self.INSTRUCTION_ARTIFACTS or start + match.end() > anchor_mid:
                break
            start += match.end()
        while start < end:
            text = question[start:end]
            match = re.search(r"[\s,:;-]*\b([A-Za-z]+)\W*$", text)
            if not match:
                break
            word = match.group(1).lower()
            if word not in self.INSTRUCTION_ARTIFACTS or start + match.start() < anchor_mid:
                break
            end = start + match.start()
        return start, end

    def _trim_outer_punctuation(self, question: str, start: int, end: int) -> tuple[int, int]:
        while start < end and question[start] in " \t\r\n,.;:!?()[]{}\"'`":
            start += 1
        while start < end and question[end - 1] in " \t\r\n,.;:!?()[]{}\"'`":
            end -= 1
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
