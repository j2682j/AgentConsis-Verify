"""Signals that decide which sources earn a full-page fetch.

Search-engine rank alone is a poor fetch policy for benchmark questions: SEO
sites generate a page per long-tail question, so they rank at the top for
exactly the queries we care about and crowd out the authoritative source the
question names. These signals give the selector evidence-based reasons to
promote or demote a candidate. No model is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import unquote, urlparse

from utils.network_utils import normalize_text

from ...config import SearchSourceCandidate


FULL_MATCH = "full_match"
PARTIAL_MATCH = "partial_match"
NO_MATCH = "no_match"

# Domain labels that carry no identity, so they must not be used to claim a
# question named a source.
_GENERIC_LABELS = {
    "www", "com", "org", "net", "edu", "gov", "co", "uk", "ac", "de", "io",
    "info", "html", "index", "page", "pages", "search", "en", "m", "web",
    "site", "blog", "shop", "news", "app", "online", "home",
}
# Short brand names that are also ordinary words: matching them in the
# question is not enough on its own.
_AMBIGUOUS_BRANDS = {
    "nature", "science", "cell", "sun", "time", "post", "mirror", "independent",
    "observer", "spectator", "atlantic", "verge", "register", "conversation",
}

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{4,80})[\"“”']")
_IDENTIFIER_RE = re.compile(
    r"\b(?:doi|isbn|rfc|rule|article|episode|season|volume|no|number)\s*\.?\s*"
    r"([0-9]+(?:[./-][0-9]+)*)\b",
    re.IGNORECASE,
)
_MODEL_RE = re.compile(r"\b[A-Z]{2,}[-_ ]?\d{2,}\b")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(normalize_text(text).casefold())
        if len(token) > 2
    }


def _brand_key(text: str) -> str:
    """Collapse a brand or domain to a comparable key.

    ``Merriam-Webster``, ``merriam-webster.com`` and ``Merriam Webster`` all
    collapse to ``merriamwebster``.
    """
    return re.sub(r"[^a-z0-9]", "", normalize_text(text).casefold())


def _domain_labels(domain: str) -> list[str]:
    labels = [
        label
        for label in normalize_text(domain).casefold().split(".")
        if label and label not in _GENERIC_LABELS
    ]
    return labels


@dataclass(frozen=True)
class QuestionConstraints:
    """Directly comparable constraints extracted from the question."""

    years: frozenset[str] = frozenset()
    versions: frozenset[str] = frozenset()
    quoted_phrases: tuple[str, ...] = ()
    identifiers: frozenset[str] = frozenset()

    @property
    def total(self) -> int:
        return (
            len(self.years)
            + len(self.versions)
            + len(self.quoted_phrases)
            + len(self.identifiers)
        )


def extract_constraints(question: str, queries: list[str] | None = None) -> QuestionConstraints:
    """Pull year / version / quoted-phrase / identifier constraints."""
    haystack = " ".join(
        [normalize_text(question)] + [normalize_text(q) for q in queries or []]
    )
    quoted = tuple(
        dict.fromkeys(
            normalize_text(match.group(1)).casefold()
            for match in _QUOTED_RE.finditer(haystack)
        )
    )
    identifiers = {
        normalize_text(match.group(1)).casefold()
        for match in _IDENTIFIER_RE.finditer(haystack)
    }
    identifiers.update(
        normalize_text(match.group(0)).casefold()
        for match in _MODEL_RE.finditer(haystack)
    )
    return QuestionConstraints(
        years=frozenset(_YEAR_RE.findall(haystack)),
        versions=frozenset(
            normalize_text(match.group(0)).casefold()
            for match in _VERSION_RE.finditer(haystack)
        ),
        quoted_phrases=quoted,
        identifiers=frozenset(identifiers),
    )


class SourceSelectionSignalBuilder:
    """Populate the fetch-selection signals on a list of candidates."""

    def __init__(self, *, url_echo_overlap: float = 0.55) -> None:
        self.url_echo_overlap = url_echo_overlap

    def build(
        self,
        sources: list[SearchSourceCandidate],
        *,
        question: str = "",
        query_text_by_id: dict[str, str] | None = None,
    ) -> None:
        """Annotate every candidate in place."""
        queries = list((query_text_by_id or {}).values())
        constraints = extract_constraints(question, queries)
        named_keys = self._named_source_keys(question, sources)
        question_tokens = _tokens(question)

        for source in sources:
            reasons: list[str] = []
            matched = self._named_terms(source, named_keys)
            source.named_source_match = bool(matched)
            source.named_source_terms = matched
            if matched:
                reasons.append(f"named_source_match:{','.join(matched)}")

            source.url_echo = self._is_url_echo(source.url, question_tokens)
            if source.url_echo:
                reasons.append("url_echo")

            source.constraint_match_level = self._constraint_level(source, constraints)
            if source.constraint_match_level != NO_MATCH:
                reasons.append(f"constraint_{source.constraint_match_level}")

            if source.query_hit_count > 1:
                reasons.append(f"cross_query_consensus:{source.query_hit_count}")
            source.fetch_priority_reasons = reasons

    def _named_source_keys(
        self,
        question: str,
        sources: list[SearchSourceCandidate],
    ) -> set[str]:
        """Brand keys the question plausibly names.

        Built from the candidates' own domains so no external brand list is
        needed: a domain label counts only when the question actually
        contains it.
        """
        question_key = _brand_key(question)
        question_tokens = _tokens(question)
        keys: set[str] = set()
        for source in sources:
            hint = normalize_text(source.source_hint)
            if hint and _brand_key(hint) and _brand_key(hint) in question_key:
                keys.add(_brand_key(hint))
            for label in _domain_labels(source.domain or urlparse(source.url).netloc):
                key = _brand_key(label)
                if not key or len(key) < 4 or key not in question_key:
                    continue
                # An ambiguous single word must also be corroborated by the
                # title, otherwise "nature" in a sentence claims nature.com.
                if key in _AMBIGUOUS_BRANDS and key not in _tokens(source.title):
                    continue
                if key in _AMBIGUOUS_BRANDS and key not in question_tokens:
                    continue
                keys.add(key)
        return keys

    def _named_terms(
        self,
        source: SearchSourceCandidate,
        named_keys: set[str],
    ) -> list[str]:
        if not named_keys:
            return []
        domain_key = _brand_key(source.domain or urlparse(source.url).netloc)
        hits = [key for key in sorted(named_keys) if key and key in domain_key]
        return hits

    def _is_url_echo(self, url: str, question_tokens: set[str]) -> bool:
        """True when the URL path replays the question wording.

        Only the path is inspected; a domain that happens to share a word with
        the question is not an echo.
        """
        if not question_tokens:
            return False
        parsed = urlparse(normalize_text(url))
        path_tokens = _tokens(unquote(parsed.path).replace("-", " ").replace("_", " "))
        if len(path_tokens) < 5:
            return False
        overlap = len(path_tokens & question_tokens) / max(1, len(question_tokens))
        novelty = len(path_tokens - question_tokens) / max(1, len(path_tokens))
        return overlap >= self.url_echo_overlap and novelty <= 0.5

    def _constraint_level(
        self,
        source: SearchSourceCandidate,
        constraints: QuestionConstraints,
    ) -> str:
        if not constraints.total:
            return NO_MATCH
        haystack = normalize_text(
            " ".join([source.title, source.url, source.snippet])
        ).casefold()
        hit = 0
        total = 0
        for group in (
            constraints.years,
            constraints.versions,
            constraints.identifiers,
        ):
            for value in group:
                total += 1
                if value and value in haystack:
                    hit += 1
        for phrase in constraints.quoted_phrases:
            total += 1
            if phrase and phrase in haystack:
                hit += 1
        if not total:
            return NO_MATCH
        if hit == total:
            return FULL_MATCH
        return PARTIAL_MATCH if hit else NO_MATCH


__all__ = [
    "FULL_MATCH",
    "NO_MATCH",
    "PARTIAL_MATCH",
    "QuestionConstraints",
    "SourceSelectionSignalBuilder",
    "extract_constraints",
]
