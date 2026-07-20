from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
import unicodedata

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class SpanAlignmentResult:
    original_span: str
    aligned_span: str = ""
    start_offset: int = -1
    end_offset: int = -1
    method: str = "none"
    token_overlap: float = 0.0
    valid: bool = False
    ambiguous: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EvidenceSpanAligner:
    """Align a generated evidence quote to one contiguous source-text span."""

    _TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
    _NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?!\w)")
    _NEGATIONS = frozenset(
        {"no", "not", "never", "none", "neither", "without", "absent", "n't"}
    )
    _UNITS = frozenset(
        {
            "m", "m2", "m3", "km", "cm", "mm", "kg", "g", "lb", "lbs",
            "ft", "feet", "inch", "inches", "liter", "liters", "litre",
            "litres", "percent", "percentage", "hz", "khz", "mhz", "ghz",
            "second", "seconds", "minute", "minutes", "hour", "hours",
            "year", "years", "million", "billion",
        }
    )

    def __init__(self, *, min_token_overlap: float = 0.85) -> None:
        self.min_token_overlap = max(0.5, min(1.0, float(min_token_overlap)))

    def align(self, span: str, source_text: str) -> SpanAlignmentResult:
        original = normalize_text(span)
        source = str(source_text or "")
        if not original or not normalize_text(source):
            return SpanAlignmentResult(original_span=original, reason="empty_input")

        exact = self._exact_matches(original, source)
        if exact:
            return self._result_from_matches(
                original,
                source,
                exact,
                method="exact",
                overlap=1.0,
            )

        source_tokens = self._tokens_with_offsets(source)
        target_tokens = self._token_values(original)
        if not source_tokens or not target_tokens:
            return SpanAlignmentResult(
                original_span=original,
                reason="no_alignable_tokens",
            )

        normalized_matches = self._normalized_token_matches(
            target_tokens,
            source_tokens,
        )
        if normalized_matches:
            return self._result_from_matches(
                original,
                source,
                normalized_matches,
                method="normalized_tokens",
                overlap=1.0,
            )

        candidates = self._fuzzy_windows(original, target_tokens, source, source_tokens)
        if not candidates:
            return SpanAlignmentResult(
                original_span=original,
                reason="no_candidate_above_threshold",
            )
        best_score = candidates[0][0]
        best = [item for item in candidates if best_score - item[0] < 0.02]
        if len({(item[1], item[2]) for item in best}) > 1:
            return SpanAlignmentResult(
                original_span=original,
                method="lcs_token_window",
                token_overlap=round(best_score, 6),
                ambiguous=True,
                reason="multiple_similar_source_spans",
            )
        _, start, end = candidates[0]
        aligned = source[start:end]
        if not self._preserves_critical_tokens(original, aligned):
            return SpanAlignmentResult(
                original_span=original,
                aligned_span=aligned,
                start_offset=start,
                end_offset=end,
                method="lcs_token_window",
                token_overlap=round(best_score, 6),
                reason="critical_token_mismatch",
            )
        return SpanAlignmentResult(
            original_span=original,
            aligned_span=aligned,
            start_offset=start,
            end_offset=end,
            method="lcs_token_window",
            token_overlap=round(best_score, 6),
            valid=True,
        )

    def _exact_matches(self, span: str, source: str) -> list[tuple[int, int]]:
        needle = span.casefold()
        haystack = source.casefold()
        matches: list[tuple[int, int]] = []
        start = 0
        while needle:
            index = haystack.find(needle, start)
            if index < 0:
                break
            matches.append((index, index + len(span)))
            start = index + max(1, len(needle))
        return matches

    def _normalized_token_matches(
        self,
        target: list[str],
        source: list[tuple[str, int, int]],
    ) -> list[tuple[int, int]]:
        width = len(target)
        matches: list[tuple[int, int]] = []
        for index in range(0, len(source) - width + 1):
            if [item[0] for item in source[index : index + width]] != target:
                continue
            matches.append((source[index][1], source[index + width - 1][2]))
        return matches

    def _fuzzy_windows(
        self,
        original: str,
        target: list[str],
        source_text: str,
        source: list[tuple[str, int, int]],
    ) -> list[tuple[float, int, int]]:
        width = len(target)
        candidates: list[tuple[float, int, int]] = []
        for candidate_width in range(max(1, width - 2), min(len(source), width + 2) + 1):
            for index in range(0, len(source) - candidate_width + 1):
                window = source[index : index + candidate_width]
                values = [item[0] for item in window]
                score = self._token_similarity(target, values)
                if score < self.min_token_overlap:
                    continue
                start, end = window[0][1], window[-1][2]
                aligned = source_text[start:end]
                if self._preserves_critical_tokens(original, aligned):
                    candidates.append((score, start, end))
        candidates.sort(key=lambda item: (-item[0], item[2] - item[1], item[1]))
        return candidates

    def _token_similarity(self, target: list[str], candidate: list[str]) -> float:
        target_counter = Counter(target)
        candidate_counter = Counter(candidate)
        overlap = sum((target_counter & candidate_counter).values()) / max(1, len(target))
        sequence = SequenceMatcher(None, target, candidate).ratio()
        return min(overlap, sequence)

    def _preserves_critical_tokens(self, original: str, aligned: str) -> bool:
        if self._numbers(original) != self._numbers(aligned):
            return False
        original_tokens = set(self._token_values(original))
        aligned_tokens = set(self._token_values(aligned))
        if (original_tokens & self._NEGATIONS) != (aligned_tokens & self._NEGATIONS):
            return False
        if (original_tokens & self._UNITS) != (aligned_tokens & self._UNITS):
            return False
        return True

    def _result_from_matches(
        self,
        original: str,
        source: str,
        matches: list[tuple[int, int]],
        *,
        method: str,
        overlap: float,
    ) -> SpanAlignmentResult:
        if len(matches) != 1:
            return SpanAlignmentResult(
                original_span=original,
                method=method,
                token_overlap=overlap,
                ambiguous=True,
                reason="multiple_exact_source_spans",
            )
        start, end = matches[0]
        aligned = source[start:end]
        return SpanAlignmentResult(
            original_span=original,
            aligned_span=aligned,
            start_offset=start,
            end_offset=end,
            method=method,
            token_overlap=overlap,
            valid=self._preserves_critical_tokens(original, aligned),
            reason=(
                "" if self._preserves_critical_tokens(original, aligned)
                else "critical_token_mismatch"
            ),
        )

    def _tokens_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        return [
            (self._normalize_token(match.group(0)), match.start(), match.end())
            for match in self._TOKEN_RE.finditer(text)
            if self._normalize_token(match.group(0))
        ]

    def _token_values(self, text: str) -> list[str]:
        return [
            token
            for token, _, _ in self._tokens_with_offsets(text)
            if token
        ]

    def _numbers(self, text: str) -> tuple[str, ...]:
        return tuple(
            re.sub(r"\s+", "", match.group(0)).casefold()
            for match in self._NUMBER_RE.finditer(unicodedata.normalize("NFKC", text))
        )

    @staticmethod
    def _normalize_token(value: str) -> str:
        return unicodedata.normalize("NFKC", value).casefold().replace("’", "'")


__all__ = ["EvidenceSpanAligner", "SpanAlignmentResult"]
