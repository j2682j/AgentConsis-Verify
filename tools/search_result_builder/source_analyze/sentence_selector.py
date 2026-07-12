from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class SelectedPassage:
    """
    保存送入 Labeler 前被選出的短 passage。

    Args:
        - text: 選句後重新組成的 passage。
        - sentences: 被選中的原始句子。
        - original_char_count: 原始 chunk 字元數。
        - selected_char_count: 選句後字元數。
        - selected_count: 被選中的句子數。
        - truncated: 是否因長度限制被截斷。
        - reasons: 選句使用到的主要訊號。

    Returns:
        - SelectedPassage: Labeler 可直接使用的短 passage。
    """

    text: str
    sentences: list[str] = field(default_factory=list)
    original_char_count: int = 0
    selected_char_count: int = 0
    selected_count: int = 0
    truncated: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LabelerSentenceSelector:
    """
    在 Labeler 前先挑出和問題、query、限制條件最相關的句子。

    Args:
        - max_sentences: 最多保留句子數。
        - max_chars: 選句後 passage 的最大字元數。
        - min_sentence_chars: 過短句子的過濾門檻。

    Returns:
        - LabelerSentenceSelector: 輕量選句器。
    """

    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_.-]{2,}")
    _NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])")
    _DATE_RE = re.compile(
        r"\b(?:18|19|20)\d{2}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b",
        re.IGNORECASE,
    )
    _CAPITAL_PHRASE_RE = re.compile(
        r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5}\b"
    )
    _STOPWORDS = {
        "about",
        "after",
        "also",
        "answer",
        "because",
        "before",
        "between",
        "could",
        "from",
        "have",
        "into",
        "many",
        "question",
        "search",
        "source",
        "that",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }

    def __init__(
        self,
        *,
        max_sentences: int = 5,
        max_chars: int = 1200,
        min_sentence_chars: int = 25,
    ) -> None:
        self.max_sentences = max(1, max_sentences)
        self.max_chars = max(200, max_chars)
        self.min_sentence_chars = max(10, min_sentence_chars)

    def select(
        self,
        *,
        question: str,
        query: str,
        text: str,
        source_title: str = "",
        answer_role: str = "unknown",
        constraints: list[str] | None = None,
    ) -> SelectedPassage:
        """
        從 chunk 中選出最適合 Labeler 判斷 useful span 的短 passage。

        Args:
            - question: 原始問題。
            - query: 目前 retrieval query。
            - text: 原始 chunk 文字。
            - source_title: 來源標題。
            - answer_role: Planner 推出的答案角色。
            - constraints: 題目中必須保留的限制詞。

        Returns:
            - SelectedPassage: 選句後 passage 與精簡 diagnostics。
        """
        normalized_text = normalize_text(text)
        if not normalized_text:
            return SelectedPassage(text="", reasons=["empty_text"])

        sentences = self._split_sentences(normalized_text)
        if not sentences:
            selected = normalized_text[: self.max_chars]
            return SelectedPassage(
                text=selected,
                sentences=[selected],
                original_char_count=len(normalized_text),
                selected_char_count=len(selected),
                selected_count=1 if selected else 0,
                truncated=len(normalized_text) > len(selected),
                reasons=["fallback_prefix"],
            )

        query_terms = set(self._keywords(" ".join([question, query, source_title])))
        constraint_terms = [
            term
            for term in self._clean_items(constraints or [])
            if len(self._keywords(term)) > 0 or any(char.isdigit() for char in term)
        ]
        scored: list[tuple[float, int, str, list[str]]] = []
        for index, sentence in enumerate(sentences):
            if len(sentence) < self.min_sentence_chars and index != 0:
                continue
            score, reasons = self._score_sentence(
                sentence=sentence,
                query_terms=query_terms,
                constraints=constraint_terms,
                answer_role=answer_role,
            )
            if source_title and self._has_overlap(sentence, source_title):
                score += 0.5
                reasons.append("source_title_overlap")
            scored.append((score, index, sentence, reasons))

        if not scored:
            selected = normalized_text[: self.max_chars]
            return SelectedPassage(
                text=selected,
                sentences=[selected],
                original_char_count=len(normalized_text),
                selected_char_count=len(selected),
                selected_count=1 if selected else 0,
                truncated=len(normalized_text) > len(selected),
                reasons=["fallback_prefix"],
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = sorted(scored[: self.max_sentences], key=lambda item: item[1])
        selected_sentences: list[str] = []
        selected_reasons: list[str] = []
        total = 0
        for _, _, sentence, reasons in chosen:
            addition = len(sentence) + (1 if selected_sentences else 0)
            if selected_sentences and total + addition > self.max_chars:
                continue
            if not selected_sentences and len(sentence) > self.max_chars:
                sentence = sentence[: self.max_chars]
                addition = len(sentence)
            selected_sentences.append(sentence)
            selected_reasons.extend(reasons)
            total += addition

        selected_text = normalize_text(" ".join(selected_sentences))
        return SelectedPassage(
            text=selected_text,
            sentences=selected_sentences,
            original_char_count=len(normalized_text),
            selected_char_count=len(selected_text),
            selected_count=len(selected_sentences),
            truncated=len(selected_text) < len(normalized_text),
            reasons=self._clean_items(selected_reasons) or ["top_retrieval_sentences"],
        )

    def _split_sentences(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        sentences = [normalize_text(part) for part in parts if normalize_text(part)]
        if len(sentences) <= 1 and len(normalized) > self.max_chars:
            sentences = [
                normalize_text(normalized[index : index + self.max_chars])
                for index in range(0, len(normalized), self.max_chars)
            ]
        return sentences

    def _score_sentence(
        self,
        *,
        sentence: str,
        query_terms: set[str],
        constraints: list[str],
        answer_role: str,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        sentence_terms = set(self._keywords(sentence))
        overlap = len(sentence_terms & query_terms)
        score = float(overlap)
        if overlap:
            reasons.append("query_term_overlap")
        constraint_hits = sum(1 for term in constraints if self._contains(sentence, term))
        if constraint_hits:
            score += 2.0 * constraint_hits
            reasons.append("constraint_coverage")
        if self._role_match(sentence, answer_role):
            score += 1.5
            reasons.append(f"answer_role:{normalize_text(answer_role) or 'unknown'}")
        if len(sentence_terms) >= 6:
            score += 0.25
        return score, reasons

    def _role_match(self, sentence: str, answer_role: str) -> bool:
        role = normalize_text(answer_role).casefold()
        if role in {"number", "count", "volume", "duration", "distance"}:
            return bool(self._NUMBER_RE.search(sentence))
        if role == "date":
            return bool(self._DATE_RE.search(sentence))
        if role in {"person", "organization", "location", "title", "species", "text_span"}:
            return bool(self._CAPITAL_PHRASE_RE.search(sentence))
        return False

    def _has_overlap(self, text: str, other: str) -> bool:
        return bool(set(self._keywords(text)) & set(self._keywords(other)))

    def _contains(self, text: str, term: str) -> bool:
        text_key = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold())
        term_key = re.sub(r"[^a-z0-9]+", " ", normalize_text(term).casefold()).strip()
        return bool(term_key and f" {term_key} " in f" {text_key} ")

    def _keywords(self, text: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in self._WORD_RE.findall(normalize_text(text)):
            key = token.casefold().strip("'_.-")
            if not key or key in self._STOPWORDS or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    def _clean_items(self, items: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items or []:
            text = normalize_text(str(item or ""))
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result


__all__ = ["LabelerSentenceSelector", "SelectedPassage"]
