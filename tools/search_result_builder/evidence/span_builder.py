from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class EvidenceSpan:
    """
    記錄 useful token 對齊回原文後的 evidence span。
    Args:
        - term: 觸發此 span 的 useful token。
        - start: span 在原文中的起始位置。
        - end: span 在原文中的結束位置。
        - span_text: useful token 在原文中還原出的完整詞或片段。
        - context: 擷取出的原文上下文。
    Returns:
        - EvidenceSpan: 可供 EvidenceRunner 組裝 prompt 的原文片段。
    """

    term: str
    start: int
    end: int
    context: str
    span_text: str = ""


@dataclass(frozen=True)
class _SentenceSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _MatchedTerm:
    term: str
    start: int
    end: int


class SpanBuilder:
    """
    將 Labeler 的 useful tokens 對齊回原始 chunk，並抽取鄰近 sentence/context span。
    Args:
        - context_window_sentences: token 所在句子前後要補多少句。
        - min_context_chars: context 太短時補句子的最低字元數。
        - max_context_chars: 單一 context span 的最大字元數。
        - merge_distance_chars: 多個 context spans 距離多近時合併。
    Returns:
        - SpanBuilder: 可重複使用的 token-to-span 重建器。
    """

    _SENTENCE_PATTERN = re.compile(r"[^.!?;\n]+(?:[.!?;]+|\n|$)", re.UNICODE)

    def __init__(
        self,
        *,
        context_window_sentences: int = 1,
        min_context_chars: int = 120,
        max_context_chars: int = 500,
        merge_distance_chars: int = 120,
    ) -> None:
        self.context_window_sentences = max(0, context_window_sentences)
        self.min_context_chars = max(1, min_context_chars)
        self.max_context_chars = max(40, max_context_chars)
        self.merge_distance_chars = max(0, merge_distance_chars)

    def build_context(
        self,
        text: str,
        matched_terms: list[Any],
        *,
        fallback_chars: int = 500,
    ) -> tuple[str, list[EvidenceSpan]]:
        """
        依 useful tokens 建立最終 evidence context。
        Args:
            - text: 原始 retrieved chunk。
            - matched_terms: Labeler 標出的 useful tokens 或 spans。
            - fallback_chars: 沒有找到 token 時保留的原文長度。
        Returns:
            - tuple[str, list[EvidenceSpan]]: 組合後 context 與 span metadata。
        """
        source_text = normalize_text(text)
        if not source_text:
            return "", []

        terms = self._clean_terms(matched_terms)
        if not terms:
            return self._truncate(source_text, fallback_chars), []

        matches = self._match_terms(source_text, terms)
        if not matches:
            return self._truncate(source_text, fallback_chars), []

        sentences = self._sentence_spans(source_text)
        spans = [
            self._context_for_match(source_text, sentences, match)
            for match in matches
        ]
        merged = self._merge_spans(source_text, spans)
        contexts = [span.context for span in merged if span.context]
        return "\n...\n".join(contexts), merged

    def _clean_terms(self, terms: list[Any]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = normalize_text(str(term or "")).strip(" ,.;:!?()[]{}")
            key = self._normalize_for_match(cleaned)
            if not key or key in seen or self._is_weak_anchor(key):
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _is_weak_anchor(self, normalized_term: str) -> bool:
        if len(normalized_term) <= 1:
            return True
        if normalized_term.isdigit() and len(normalized_term) < 4:
            return True
        if len(normalized_term) < 4 and " " not in normalized_term:
            return True
        return False

    def _match_terms(self, text: str, terms: list[str]) -> list[_MatchedTerm]:
        normalized_text, index_map = self._normalize_with_index_map(text)
        matches: list[_MatchedTerm] = []
        seen: set[tuple[int, int, str]] = set()
        for term in terms:
            normalized_term = self._normalize_for_match(term)
            if not normalized_term:
                continue
            start = normalized_text.find(normalized_term)
            while start >= 0:
                end = start + len(normalized_term)
                original_start = index_map[start]
                original_end = index_map[end - 1] + 1
                key = (original_start, original_end, normalized_term)
                if key not in seen:
                    expanded_start, expanded_end = self._expand_match_bounds(
                        text,
                        original_start,
                        original_end,
                    )
                    matches.append(
                        _MatchedTerm(
                            term=term,
                            start=expanded_start,
                            end=expanded_end,
                        )
                    )
                    seen.add(key)
                start = normalized_text.find(normalized_term, start + 1)
        matches.sort(key=lambda item: (item.start, item.end, item.term.lower()))
        return matches

    def _normalize_with_index_map(self, text: str) -> tuple[str, list[int]]:
        chars: list[str] = []
        index_map: list[int] = []
        previous_space = False
        for index, character in enumerate(text):
            normalized = self._strip_accents(character.casefold())
            if normalized.isalnum():
                chars.append(normalized)
                index_map.extend([index] * len(normalized))
                previous_space = False
            elif not previous_space:
                chars.append(" ")
                index_map.append(index)
                previous_space = True
        normalized_text = "".join(chars).strip()
        leading_spaces = len("".join(chars)) - len("".join(chars).lstrip())
        if leading_spaces:
            index_map = index_map[leading_spaces:]
        return normalized_text, index_map[: len(normalized_text)]

    def _normalize_for_match(self, value: str) -> str:
        normalized, _ = self._normalize_with_index_map(value)
        return normalized.strip()

    def _strip_accents(self, value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    def _sentence_spans(self, text: str) -> list[_SentenceSpan]:
        spans: list[_SentenceSpan] = []
        for match in self._SENTENCE_PATTERN.finditer(text):
            sentence = normalize_text(match.group(0))
            if not sentence:
                continue
            start = match.start()
            end = match.end()
            spans.append(_SentenceSpan(text=sentence, start=start, end=end))
        if not spans:
            spans.append(_SentenceSpan(text=text, start=0, end=len(text)))
        return spans

    def _context_for_match(
        self,
        text: str,
        sentences: list[_SentenceSpan],
        match: _MatchedTerm,
    ) -> EvidenceSpan:
        sentence_index = self._sentence_index_for_match(sentences, match)
        if sentence_index is None:
            start = max(0, match.start - self.max_context_chars // 2)
            end = min(len(text), match.end + self.max_context_chars // 2)
            context = self._trim_to_limit(text[start:end])
            return EvidenceSpan(
                term=match.term,
                start=start,
                end=end,
                context=context,
                span_text=normalize_text(text[match.start : match.end]),
            )

        start_index = max(0, sentence_index - self.context_window_sentences)
        end_index = min(
            len(sentences) - 1,
            sentence_index + self.context_window_sentences,
        )
        while (
            self._span_length(sentences, start_index, end_index)
            < self.min_context_chars
            and (start_index > 0 or end_index < len(sentences) - 1)
        ):
            if start_index > 0:
                start_index -= 1
            if (
                self._span_length(sentences, start_index, end_index)
                >= self.min_context_chars
            ):
                break
            if end_index < len(sentences) - 1:
                end_index += 1

        start = sentences[start_index].start
        end = sentences[end_index].end
        context = self._trim_to_limit(text[start:end])
        return EvidenceSpan(
            term=match.term,
            start=start,
            end=end,
            context=context,
            span_text=normalize_text(text[match.start : match.end]),
        )

    def _sentence_index_for_match(
        self,
        sentences: list[_SentenceSpan],
        match: _MatchedTerm,
    ) -> int | None:
        for index, sentence in enumerate(sentences):
            if sentence.start <= match.start < sentence.end:
                return index
            if max(sentence.start, match.start) < min(sentence.end, match.end):
                return index
        return None

    def _span_length(
        self,
        sentences: list[_SentenceSpan],
        start_index: int,
        end_index: int,
    ) -> int:
        return max(0, sentences[end_index].end - sentences[start_index].start)

    def _merge_spans(
        self,
        text: str,
        spans: list[EvidenceSpan],
    ) -> list[EvidenceSpan]:
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda span: (span.start, span.end))
        merged: list[EvidenceSpan] = []
        current = sorted_spans[0]
        terms = [current.term]
        for span in sorted_spans[1:]:
            if span.start - current.end <= self.merge_distance_chars:
                current = EvidenceSpan(
                    term=", ".join(terms + [span.term]),
                    start=min(current.start, span.start),
                    end=max(current.end, span.end),
                    context=self._trim_to_limit(
                        text[min(current.start, span.start) : max(current.end, span.end)]
                    ),
                    span_text=normalize_text(
                        text[min(current.start, span.start) : max(current.end, span.end)]
                    ),
                )
                terms = [term.strip() for term in current.term.split(",")]
            else:
                merged.append(current)
                current = span
                terms = [span.term]
        merged.append(current)
        return merged

    def _trim_to_limit(self, text: str) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= self.max_context_chars:
            return cleaned
        return cleaned[: self.max_context_chars].rstrip() + "..."

    def _truncate(self, text: str, max_chars: int) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    def _expand_match_bounds(
        self,
        text: str,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        start, end = self._expand_to_word_bounds(text, start, end)
        start, end = self._expand_named_phrase(text, start, end)
        return start, end

    def _expand_to_word_bounds(
        self,
        text: str,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        while end < len(text) and self._is_word_char(text[end]):
            end += 1
        return start, end

    def _expand_named_phrase(
        self,
        text: str,
        start: int,
        end: int,
        *,
        max_chars: int = 80,
    ) -> tuple[int, int]:
        left = start
        right = end
        while left > 0 and right - left < max_chars:
            candidate_start = self._previous_token_start(text, left)
            if candidate_start is None:
                break
            token = text[candidate_start:left].strip(" \t\r\n,;:()[]{}")
            if not self._is_phrase_token(token):
                break
            left = candidate_start
        while right < len(text) and right - left < max_chars:
            candidate_end = self._next_token_end(text, right)
            if candidate_end is None:
                break
            token = text[right:candidate_end].strip(" \t\r\n,;:()[]{}")
            if not self._is_phrase_token(token):
                break
            right = candidate_end
        return self._trim_outer_space(text, left, right)

    def _previous_token_start(self, text: str, end: int) -> int | None:
        index = end
        while index > 0 and text[index - 1].isspace():
            index -= 1
        if index > 0 and text[index - 1] in "'-":
            index -= 1
        start = index
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        if start == index:
            return None
        return start

    def _next_token_end(self, text: str, start: int) -> int | None:
        index = start
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] in "'-":
            index += 1
        end = index
        while end < len(text) and self._is_word_char(text[end]):
            end += 1
        if end == index:
            return None
        return end

    def _is_word_char(self, character: str) -> bool:
        return character.isalnum() or character in {"'", "-", "_"}

    def _is_phrase_token(self, token: str) -> bool:
        if not token:
            return False
        stripped = token.strip("'\"")
        if not stripped:
            return False
        if any(character.isdigit() for character in stripped):
            return True
        return stripped[:1].isupper() and len(stripped) > 1

    def _trim_outer_space(
        self,
        text: str,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end


__all__ = ["EvidenceSpan", "SpanBuilder"]
