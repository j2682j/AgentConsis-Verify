from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..query.semantic_impact import SemanticImpactScorer


@dataclass(frozen=True)
class EvidenceUnit:
    """
    Store a candidate textual unit before Labeler input construction.

    Args:
        - text: Unit text.
        - unit_type: sentence / line / table_like / key_value.
        - quality_flags: General text-quality flags.
        - relevance_score: Semantic relevance to the question.
        - novelty_score: Lexical novelty against the question.
        - rank_score: Final ranking score.
        - selected: Whether the unit is retained for Labeler input.

    Returns:
        - EvidenceUnit: Scored evidence unit.
    """

    text: str
    unit_type: str = "sentence"
    quality_flags: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    rank_score: float = 0.0
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceUnitSelection:
    """
    Store selected evidence units for Labeler input.

    Args:
        - text: Joined selected evidence units.
        - units: All scored candidate units.
        - selected_units: Units retained for Labeler input.
        - dropped_units: Units removed or not selected.
        - diagnostics: Aggregate selector diagnostics.

    Returns:
        - EvidenceUnitSelection: Prepared Labeler passage units.
    """

    text: str
    units: list[EvidenceUnit] = field(default_factory=list)
    selected_units: list[EvidenceUnit] = field(default_factory=list)
    dropped_units: list[EvidenceUnit] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class EvidenceUnitSelector:
    """
    Select clean, relevant, and novel evidence units before Labeler inference.

    Args:
        - semantic_scorer: Optional encoder scorer for query-conditioned relevance.
        - max_units: Maximum retained units.
        - max_chars: Maximum joined Labeler passage length.
        - min_unit_chars: Minimum candidate unit length.

    Returns:
        - EvidenceUnitSelector: General evidence-unit selector.
    """

    FORM_TERMS = {
        "captcha",
        "cloudflare",
        "cookie",
        "cookies",
        "e-mail",
        "email",
        "forgot password",
        "javascript",
        "login",
        "not a robot",
        "not a bot",
        "password",
        "performing security verification",
        "privacy",
        "register here",
        "security service",
        "sign in",
        "sign-in",
        "user name",
    }
    NAV_TERMS = {
        "about",
        "advertisement",
        "contact us",
        "current events",
        "help",
        "main page",
        "menu",
        "navigation",
        "privacy policy",
        "random article",
        "share",
        "subscribe",
        "terms of use",
        "upload file",
    }
    METADATA_KEYS = {
        "author.name",
        "content:",
        "datepublished",
        "datemodified",
        "description:",
        "headings:",
        "metadata:",
        "og:description",
        "og:title",
        "publisher.name",
        "structured data:",
        "title:",
    }

    def __init__(
        self,
        *,
        semantic_scorer: SemanticImpactScorer | None = None,
        max_units: int = 4,
        max_chars: int = 700,
        min_unit_chars: int = 24,
    ) -> None:
        self.semantic_scorer = semantic_scorer or SemanticImpactScorer()
        self.max_units = max(1, max_units)
        self.max_chars = max(160, max_chars)
        self.min_unit_chars = max(8, min_unit_chars)

    def select(
        self,
        *,
        question: str,
        current_query: str,
        source_title: str,
        selected_passage: str,
        raw_text: str,
    ) -> EvidenceUnitSelection:
        """
        Select top evidence units for Labeler input.

        Args:
            - question: Original task question.
            - current_query: Current retrieval query.
            - source_title: Source title for diagnostics only.
            - selected_passage: Sentence-selector output.
            - raw_text: Raw retrieved chunk text.

        Returns:
            - EvidenceUnitSelection: Joined clean units and diagnostics.
        """
        del source_title
        source_text = str(selected_passage or raw_text or "")
        units = self._build_units(source_text)
        if not units:
            return EvidenceUnitSelection(
                text="",
                diagnostics={
                    "evidence_unit_selector_used": True,
                    "evidence_unit_empty_fallback": True,
                    "evidence_unit_empty_reason": "no_units",
                },
            )

        scored = self._score_units(
            units=units,
            question=question,
            current_query=current_query,
        )
        selected = self._select_top_units(scored)
        retried_raw_text = False
        raw_source = str(raw_text or "")
        if (
            not selected
            and raw_source
            and raw_source.strip() != str(source_text or "").strip()
        ):
            raw_units = self._build_units(raw_source)
            raw_scored = self._score_units(
                units=raw_units,
                question=question,
                current_query=current_query,
            )
            raw_selected = self._select_top_units(raw_scored)
            if raw_selected:
                units = raw_units
                scored = raw_scored
                selected = raw_selected
                retried_raw_text = True
        selected_keys = {unit.text.casefold() for unit in selected}
        dropped = [unit for unit in scored if unit.text.casefold() not in selected_keys]
        text = normalize_text(
            "\n".join(
                f"Evidence unit {index}: {unit.text}"
                for index, unit in enumerate(selected, start=1)
            )
        )
        diagnostics = self._diagnostics(
            units=scored,
            selected=selected,
            dropped=dropped,
            empty_fallback=not bool(selected),
            retried_raw_text=retried_raw_text,
        )
        return EvidenceUnitSelection(
            text=text,
            units=scored,
            selected_units=selected,
            dropped_units=dropped,
            diagnostics=diagnostics,
        )

    def _build_units(self, text: str) -> list[EvidenceUnit]:
        candidates: list[str] = []
        for line in re.split(r"[\r\n]+", text):
            line = normalize_text(line)
            if not line:
                continue
            if len(line) > 260 and not self._structured_type(line):
                candidates.extend(self._split_sentences(line))
            else:
                candidates.append(line)

        if len(candidates) <= 1:
            candidates = self._split_sentences(text)

        result: list[EvidenceUnit] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = normalize_text(candidate).strip(" \"'`")
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            unit_type = self._unit_type(cleaned)
            flags = self._quality_flags(cleaned, unit_type=unit_type)
            result.append(
                EvidenceUnit(
                    text=cleaned,
                    unit_type=unit_type,
                    quality_flags=flags,
                )
            )
        return result

    def _split_sentences(self, text: str) -> list[str]:
        source = normalize_text(text)
        if not source:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", source)
        if len(parts) == 1:
            parts = re.split(r"\s{2,}|;\s+", source)
        return [normalize_text(part) for part in parts if normalize_text(part)]

    def _score_units(
        self,
        *,
        units: list[EvidenceUnit],
        question: str,
        current_query: str,
    ) -> list[EvidenceUnit]:
        reference = normalize_text(" ".join([question, current_query]))
        texts = [unit.text for unit in units]
        similarities = self._semantic_similarities(reference, texts)
        scored: list[EvidenceUnit] = []
        for index, unit in enumerate(units):
            relevance = (
                max(0.0, min(1.0, similarities[index]))
                if index < len(similarities)
                else 0.0
            )
            novelty = self._novelty(unit.text, question)
            structured_bonus = 0.12 if unit.unit_type in {"table_like", "key_value"} else 0.0
            boilerplate_penalty = self._boilerplate_penalty(unit.quality_flags)
            quality_bonus = self._quality_bonus(unit.text, unit.quality_flags)
            rank_score = relevance + novelty + structured_bonus + quality_bonus - boilerplate_penalty
            scored.append(
                EvidenceUnit(
                    text=unit.text,
                    unit_type=unit.unit_type,
                    quality_flags=list(unit.quality_flags),
                    relevance_score=round(relevance, 6),
                    novelty_score=round(novelty, 6),
                    rank_score=round(rank_score, 6),
                    selected=False,
                )
            )
        return scored

    def _select_top_units(self, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
        candidates = [
            unit
            for unit in units
            if not self._drop_unit(unit)
        ]
        candidates.sort(
            key=lambda unit: (
                unit.unit_type in {"table_like", "key_value"},
                unit.rank_score,
                len(unit.text),
            ),
            reverse=True,
        )
        selected: list[EvidenceUnit] = []
        used_chars = 0
        for unit in candidates:
            if len(selected) >= self.max_units:
                break
            if used_chars + len(unit.text) > self.max_chars and selected:
                continue
            selected.append(
                EvidenceUnit(
                    text=unit.text,
                    unit_type=unit.unit_type,
                    quality_flags=list(unit.quality_flags),
                    relevance_score=unit.relevance_score,
                    novelty_score=unit.novelty_score,
                    rank_score=unit.rank_score,
                    selected=True,
                )
            )
            used_chars += len(unit.text)
        return selected

    def _drop_unit(self, unit: EvidenceUnit) -> bool:
        flags = set(unit.quality_flags)
        if flags & {"form_like", "navigation_like", "symbol_heavy"}:
            return True
        if "metadata_like" in flags:
            return True
        if "too_short" in flags and unit.unit_type == "sentence":
            return True
        if "low_information" in flags and unit.unit_type == "sentence":
            return True
        return False

    def _quality_flags(self, text: str, *, unit_type: str) -> list[str]:
        lowered = text.casefold()
        tokens = self._tokens(text)
        flags: list[str] = []
        if len(text) < self.min_unit_chars:
            flags.append("too_short")
        if self._contains_any(lowered, self.METADATA_KEYS):
            flags.append("metadata_like")
        if self._contains_any(lowered, self.FORM_TERMS):
            flags.append("form_like")
        if self._contains_any(lowered, self.NAV_TERMS):
            flags.append("navigation_like")
        if tokens and len(set(tokens)) / len(tokens) < 0.45:
            flags.append("repeated_text")
        if tokens and self._alpha_ratio(tokens) < 0.35 and unit_type not in {"table_like", "key_value"}:
            flags.append("low_information")
        if self._punctuation_ratio(text) > 0.35:
            flags.append("symbol_heavy")
        return flags

    def _unit_type(self, text: str) -> str:
        structured = self._structured_type(text)
        if structured:
            return structured
        return "sentence"

    def _structured_type(self, text: str) -> str:
        if re.search(r"\b[A-Za-z][A-Za-z0-9 _/-]{2,40}\s*:\s*\S+", text):
            return "key_value"
        if "|" in text or "\t" in text:
            return "table_like"
        tokens = self._tokens(text)
        digit_tokens = sum(1 for token in tokens if any(char.isdigit() for char in token))
        if digit_tokens >= 2 and len(tokens) >= 4:
            return "table_like"
        return ""

    def _semantic_similarities(self, reference: str, texts: list[str]) -> list[float]:
        if not reference or not texts:
            return [0.0] * len(texts)
        try:
            return self.semantic_scorer.semantic_similarities(reference, texts)
        except Exception:
            return [0.0] * len(texts)

    def _novelty(self, text: str, question: str) -> float:
        unit_terms = set(self._content_tokens(text))
        if not unit_terms:
            return 0.0
        question_terms = set(self._content_tokens(question))
        new_terms = [term for term in unit_terms if term not in question_terms]
        return len(new_terms) / len(unit_terms)

    def _boilerplate_penalty(self, flags: list[str]) -> float:
        penalties = {
            "metadata_like": 0.35,
            "form_like": 0.8,
            "navigation_like": 0.45,
            "low_information": 0.25,
            "repeated_text": 0.2,
            "symbol_heavy": 0.6,
            "too_short": 0.15,
        }
        return sum(penalties.get(flag, 0.0) for flag in flags)

    def _quality_bonus(self, text: str, flags: list[str]) -> float:
        if flags:
            return 0.0
        tokens = self._tokens(text)
        if len(tokens) >= 8:
            return 0.08
        return 0.03

    def _diagnostics(
        self,
        *,
        units: list[EvidenceUnit],
        selected: list[EvidenceUnit],
        dropped: list[EvidenceUnit],
        empty_fallback: bool,
        retried_raw_text: bool,
    ) -> dict[str, Any]:
        dropped_flags: dict[str, int] = {}
        for unit in dropped:
            for flag in unit.quality_flags:
                dropped_flags[flag] = dropped_flags.get(flag, 0) + 1
        blocking_flags = {
            "form_like",
            "metadata_like",
            "navigation_like",
            "symbol_heavy",
        }
        should_fallback = bool(
            empty_fallback
            and not any(flag in dropped_flags for flag in blocking_flags)
        )
        selected_types = [unit.unit_type for unit in selected]
        avg_relevance = (
            sum(unit.relevance_score for unit in selected) / len(selected)
            if selected
            else 0.0
        )
        avg_novelty = (
            sum(unit.novelty_score for unit in selected) / len(selected)
            if selected
            else 0.0
        )
        return {
            "evidence_unit_selector_used": True,
            "evidence_unit_count": len(units),
            "evidence_unit_selected_count": len(selected),
            "evidence_unit_dropped_count": len(dropped),
            "evidence_unit_empty_fallback": empty_fallback,
            "evidence_unit_should_fallback": should_fallback,
            "evidence_unit_empty_reason": (
                "no_selected_units" if empty_fallback else ""
            ),
            "evidence_unit_retried_raw_text": retried_raw_text,
            "evidence_unit_selected_types": selected_types,
            "evidence_unit_dropped_flags": dropped_flags,
            "evidence_unit_avg_relevance": round(avg_relevance, 6),
            "evidence_unit_avg_novelty": round(avg_novelty, 6),
            "evidence_unit_selected": [unit.to_dict() for unit in selected],
        }

    def _contains_any(self, lowered: str, terms: set[str]) -> bool:
        return any(term in lowered for term in terms)

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]*", normalize_text(text))

    def _content_tokens(self, text: str) -> list[str]:
        result: list[str] = []
        for token in self._tokens(text.casefold()):
            if len(token) <= 2:
                continue
            if token in SemanticImpactScorer.STOPWORDS:
                continue
            result.append(token)
        return result

    def _alpha_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        alpha = sum(1 for token in tokens if re.search(r"[A-Za-z]", token))
        return alpha / len(tokens)

    def _punctuation_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        punct = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return punct / len(text)


__all__ = [
    "EvidenceUnit",
    "EvidenceUnitSelection",
    "EvidenceUnitSelector",
]
