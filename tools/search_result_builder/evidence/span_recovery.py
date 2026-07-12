from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text


_SPACY_NLP = None
_SPACY_LOAD_ATTEMPTED = False


@dataclass(frozen=True)
class RecoveredSpans:
    """
    Store fallback answer and bridge spans recovered without labeler spans.

    Args:
        - answer_spans: Candidate spans that match the expected answer role.
        - bridge_spans: Clue spans useful for next-hop retrieval.
        - answer_role: Normalized expected answer role.
        - reasons: Diagnostics explaining how spans were recovered.

    Returns:
        - RecoveredSpans: Fallback span recovery result.
    """

    answer_spans: list[str] = field(default_factory=list)
    bridge_spans: list[str] = field(default_factory=list)
    answer_role: str = "unknown"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpanRecovery:
    """
    Recover answer-like or clue-like spans when labeler useful spans are missing.

    Args:
        - max_answer_spans: Maximum recovered answer spans.
        - max_bridge_spans: Maximum recovered bridge spans.
        - max_scan_chars: Maximum document text inspected per chunk.

    Returns:
        - SpanRecovery: Lightweight fallback span recovery helper.
    """

    _NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])")
    _VOLUME_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:m\^?3|m3|cubic\s+met(?:er|re)s?|lit(?:er|re)s?|l)\b",
        re.IGNORECASE,
    )
    _DURATION_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
        re.IGNORECASE,
    )
    _DISTANCE_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|miles?|meters?|metres?)\b",
        re.IGNORECASE,
    )
    _DATE_RE = re.compile(
        r"\b(?:18|19|20)\d{2}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b|"
        r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b",
        re.IGNORECASE,
    )
    _QUOTED_RE = re.compile(r'"([^"]{3,120})"|\'([^\']{3,120})\'')
    _CAPITAL_PHRASE_RE = re.compile(
        r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,6}\b"
    )
    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}")
    _STOPWORDS = {
        "about",
        "after",
        "also",
        "answer",
        "article",
        "before",
        "between",
        "content",
        "document",
        "from",
        "have",
        "into",
        "many",
        "page",
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
        max_answer_spans: int = 6,
        max_bridge_spans: int = 8,
        max_scan_chars: int = 3500,
    ) -> None:
        self.max_answer_spans = max(1, max_answer_spans)
        self.max_bridge_spans = max(1, max_bridge_spans)
        self.max_scan_chars = max(500, max_scan_chars)

    def recover(
        self,
        *,
        question: str,
        title: str,
        text: str,
        intent_plan: Any | None = None,
        answer_role: str = "",
    ) -> RecoveredSpans:
        """
        Recover fallback spans from document text and intent state.

        Args:
            - question: Original task question.
            - title: Retrieved source title.
            - text: Retrieved document chunk text.
            - intent_plan: Optional SearchIntentPlan-like state.
            - answer_role: Optional expected answer role override.

        Returns:
            - RecoveredSpans: Answer-like spans and bridge clue spans.
        """
        source_text = normalize_text(" ".join(part for part in [title, text] if part))
        if not source_text:
            return RecoveredSpans(reasons=["empty_recovery_text"])
        scan_text = source_text[: self.max_scan_chars]
        role = self._answer_role(question, intent_plan=intent_plan, answer_role=answer_role)
        answer_spans = self._answer_spans(scan_text, role)
        bridge_spans = self._bridge_spans(
            question=question,
            title=title,
            text=scan_text,
            intent_plan=intent_plan,
            answer_spans=answer_spans,
        )
        reasons: list[str] = []
        if answer_spans:
            reasons.append(f"recovered_answer_span:{role}")
        if bridge_spans:
            reasons.append("recovered_bridge_span")
        if not reasons:
            reasons.append("no_recoverable_span")
        return RecoveredSpans(
            answer_spans=answer_spans[: self.max_answer_spans],
            bridge_spans=bridge_spans[: self.max_bridge_spans],
            answer_role=role,
            reasons=reasons,
        )

    def recover_restricted(
        self,
        *,
        question: str,
        title: str = "",
        selected_passage: str,
        intent_plan: Any | None = None,
        answer_role: str = "",
        sequence_tag: str = "",
    ) -> RecoveredSpans:
        """
        只在 sentence selection 後的 passage 裡補回 Labeler 未標出的 span。

        Args:
            - question: 原始問題。
            - title: 來源標題。
            - selected_passage: Labeler 實際看到的短 passage。
            - intent_plan: SearchIntentPlan 狀態。
            - answer_role: 預期答案角色。
            - sequence_tag: Labeler sequence head 的輸出。

        Returns:
            - RecoveredSpans: 受限範圍內補出的 answer 或 bridge spans。
        """
        passage = normalize_text(selected_passage)
        if not passage:
            return RecoveredSpans(reasons=["restricted_empty_passage"])
        role = self._answer_role(question, intent_plan=intent_plan, answer_role=answer_role)
        tag = normalize_text(sequence_tag)
        answer_spans: list[str] = []
        bridge_spans: list[str] = []
        if tag in {"<TERMINATE>", "<FINISH>"}:
            answer_spans = self._answer_spans(passage, role)
        elif tag == "<CONTINUE>":
            bridge_spans = self._bridge_spans(
                question=question,
                title=title,
                text=passage,
                intent_plan=intent_plan,
                answer_spans=[],
            )
        else:
            bridge_spans = self._bridge_spans(
                question=question,
                title=title,
                text=passage,
                intent_plan=intent_plan,
                answer_spans=[],
            )

        reasons: list[str] = ["restricted_span_recovery"]
        if answer_spans:
            reasons.append(f"restricted_answer_span:{role}")
        if bridge_spans:
            reasons.append("restricted_bridge_span")
        if not answer_spans and not bridge_spans:
            reasons.append("restricted_no_span")
        return RecoveredSpans(
            answer_spans=answer_spans[: self.max_answer_spans],
            bridge_spans=bridge_spans[: self.max_bridge_spans],
            answer_role=role,
            reasons=reasons,
        )

    def _answer_spans(self, text: str, answer_role: str) -> list[str]:
        if answer_role == "volume":
            return self._regex_spans(self._VOLUME_RE, text)
        if answer_role == "duration":
            return self._regex_spans(self._DURATION_RE, text)
        if answer_role == "distance":
            return self._regex_spans(self._DISTANCE_RE, text)
        if answer_role == "date":
            return self._regex_spans(self._DATE_RE, text)
        if answer_role in {"number", "count"}:
            return [
                span
                for span in self._regex_spans(self._NUMBER_RE, text)
                if not self._is_year_like(span)
            ]
        if answer_role in {"person", "organization", "location"}:
            return self._entity_spans(text, answer_role)
        if answer_role in {"title", "text_span", "species"}:
            return self._title_like_spans(text)
        return []

    def _bridge_spans(
        self,
        *,
        question: str,
        title: str,
        text: str,
        intent_plan: Any | None,
        answer_spans: list[str],
    ) -> list[str]:
        spans: list[str] = []
        combined = normalize_text(" ".join([title, text]))
        for term in self._intent_terms(intent_plan):
            if self._contains_term(combined, term):
                spans.append(term)
        for phrase in self._quoted_terms(question):
            if self._contains_term(combined, phrase):
                spans.append(phrase)
        spans.extend(self._title_like_spans(title)[:4])
        spans.extend(self._capital_phrases(text)[:8])
        if len(self._dedupe(spans)) < 3:
            spans.extend(self._entity_spans(text, "any")[:6])
        spans = [
            span
            for span in spans
            if self._is_informative(span)
            and self._normalize_key(span) not in {self._normalize_key(item) for item in answer_spans}
        ]
        return self._dedupe(spans)[: self.max_bridge_spans]

    def _entity_spans(self, text: str, answer_role: str) -> list[str]:
        nlp = self._load_spacy()
        if nlp is None:
            return self._capital_phrases(text)
        label_sets = {
            "person": {"PERSON"},
            "organization": {"ORG"},
            "location": {"GPE", "LOC", "FAC"},
            "any": {"PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART", "EVENT"},
        }
        allowed = label_sets.get(answer_role, label_sets["any"])
        spans = [
            normalize_text(entity.text)
            for entity in nlp(text[: self.max_scan_chars]).ents
            if entity.label_ in allowed and self._is_informative(entity.text)
        ]
        return self._dedupe(spans)

    def _title_like_spans(self, text: str) -> list[str]:
        spans: list[str] = []
        spans.extend(self._quoted_terms(text))
        spans.extend(self._capital_phrases(text))
        return self._dedupe(spans)

    def _capital_phrases(self, text: str) -> list[str]:
        spans = [
            normalize_text(match.group(0))
            for match in self._CAPITAL_PHRASE_RE.finditer(text)
            if self._is_informative(match.group(0))
        ]
        return self._dedupe(spans)

    def _regex_spans(self, pattern: re.Pattern[str], text: str) -> list[str]:
        return self._dedupe(
            normalize_text(match.group(0))
            for match in pattern.finditer(text)
            if normalize_text(match.group(0))
        )

    def _quoted_terms(self, text: str) -> list[str]:
        return self._dedupe(
            normalize_text(match.group(1) or match.group(2))
            for match in self._QUOTED_RE.finditer(text)
            if normalize_text(match.group(1) or match.group(2))
        )

    def _intent_terms(self, intent_plan: Any | None) -> list[str]:
        if intent_plan is None:
            return []
        terms: list[str] = []
        target = normalize_text(getattr(intent_plan, "target", "") or "")
        if target:
            terms.append(target)
        terms.extend(list(getattr(intent_plan, "must_include", []) or []))
        terms.extend(
            term
            for term in list(getattr(intent_plan, "completed_terms", []) or [])
            if not str(term).startswith("answer_candidate:")
        )
        return self._dedupe(str(term) for term in terms)

    def _answer_role(
        self,
        question: str,
        *,
        intent_plan: Any | None,
        answer_role: str,
    ) -> str:
        role = normalize_text(answer_role).casefold()
        if not role or role == "unknown":
            role = normalize_text(getattr(intent_plan, "answer_role", "") if intent_plan else "").casefold()
        if role:
            if role == "count":
                return "number"
            return role
        lowered = normalize_text(question).casefold()
        if re.search(r"\bwhat\s+does\b.+\bstand\s+for\b", lowered):
            return "text_span"
        if re.search(r"\b(?:what|which)\s+writer\b|\bquoted\s+by\b|\bfirst\s+name\b|\blast\s+name\b|\bwho\b", lowered):
            return "person"
        if "m^3" in lowered or "m3" in lowered or "cubic meter" in lowered or "volume" in lowered:
            return "volume"
        if any(term in lowered for term in ("hour", "minute", "second", "how long")):
            return "duration"
        if any(term in lowered for term in ("distance", "kilometer", "kilometre", "mile")):
            return "distance"
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", lowered):
            return "number"
        if re.search(r"\bwhen\b|\bwhat date\b|\bwhat year\b|\bwhich year\b", lowered):
            return "date"
        if re.search(r"\bwhere\b|\bwhich country\b|\bwhich city\b", lowered):
            return "location"
        if re.search(r"\btitle\b|\bname of\b|\bcalled\b", lowered):
            return "title"
        return "unknown"

    def _contains_term(self, text: str, term: str) -> bool:
        cleaned_text = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold())
        cleaned_term = re.sub(r"[^a-z0-9]+", " ", normalize_text(term).casefold()).strip()
        return bool(cleaned_term and f" {cleaned_term} " in f" {cleaned_text} ")

    def _is_year_like(self, value: str) -> bool:
        cleaned = normalize_text(value).replace(",", "").rstrip("%")
        return bool(re.fullmatch(r"(?:18|19|20)\d{2}", cleaned))

    def _is_informative(self, value: str) -> bool:
        cleaned = normalize_text(value).strip(" ,.;:!?()[]{}'\"")
        key = cleaned.casefold()
        if not cleaned or key in self._STOPWORDS:
            return False
        if len(cleaned) < 3:
            return False
        words = [
            word.casefold()
            for word in self._WORD_RE.findall(cleaned)
            if word.casefold() not in self._STOPWORDS
        ]
        if not words:
            return False
        if len(words) == 1 and len(words[0]) < 4 and not any(char.isdigit() for char in words[0]):
            return False
        return True

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).casefold()).strip()

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(str(value or ""))
            key = self._normalize_key(text)
            if text and key and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    def _load_spacy(self):
        global _SPACY_LOAD_ATTEMPTED, _SPACY_NLP
        if _SPACY_NLP is not None:
            return _SPACY_NLP
        if _SPACY_LOAD_ATTEMPTED:
            return None
        _SPACY_LOAD_ATTEMPTED = True
        try:
            import spacy

            for model_name in ("en_core_web_md", "en_core_web_sm"):
                try:
                    _SPACY_NLP = spacy.load(
                        model_name,
                        disable=["parser", "tagger", "lemmatizer", "textcat"],
                    )
                    return _SPACY_NLP
                except Exception:
                    continue
        except Exception:
            return None
        return None


__all__ = ["RecoveredSpans", "SpanRecovery"]
