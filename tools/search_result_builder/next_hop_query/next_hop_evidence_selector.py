from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..query.search_intent_plan import SearchIntentPlan


@dataclass(frozen=True)
class RejectedNextHopSpan:
    """
    Store a rejected next-hop evidence span for diagnostics.

    Args:
        - document_id: Source document ID.
        - span: Rejected bridge span.
        - reason: Rejection reason.

    Returns:
        - RejectedNextHopSpan: Debug record for next-hop evidence selection.
    """

    document_id: str
    span: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NextHopEvidenceSelection:
    """
    Store bridge spans selected for next-hop query composition.

    Args:
        - bridge_spans: Clean bridge spans allowed into the composer.
        - selected_document_ids: Documents contributing selected bridge spans.
        - rejected: Rejected span diagnostics.
        - metadata: Aggregate selection metadata.

    Returns:
        - NextHopEvidenceSelection: Strict next-hop evidence input.
    """

    bridge_spans: list[str] = field(default_factory=list)
    selected_document_ids: list[str] = field(default_factory=list)
    rejected: list[RejectedNextHopSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_spans": list(self.bridge_spans),
            "selected_document_ids": list(self.selected_document_ids),
            "rejected": [item.to_dict() for item in self.rejected],
            "metadata": dict(self.metadata),
        }


class NextHopEvidenceSelector:
    """
    Select strict evidence-side bridge spans for next-hop query composition.

    Args:
        - max_bridge_spans: Maximum bridge spans exposed to the composer.
        - min_span_chars: Minimum span length unless the span contains letters and digits.
        - max_span_chars: Maximum span length.

    Returns:
        - NextHopEvidenceSelector: Document-to-bridge-span selector.
    """

    NOISE_TERMS = {
        "about",
        "advertisement",
        "abstract",
        "captcha",
        "cloudflare",
        "community",
        "content",
        "copyright",
        "attribution",
        "creative commons",
        "current community",
        "e-mail",
        "email",
        "external links",
        "headings",
        "home",
        "image alt",
        "introduction",
        "lists",
        "login",
        "metadata",
        "navigation",
        "password",
        "privacy",
        "references",
        "related articles",
        "script",
        "sign in",
        "source",
        "structured data",
        "suggested searches",
        "table",
        "terms",
        "title",
        "user name",
        "verification",
        "wordplays",
    }
    PAGE_FEATURE_TOKENS = {
        "best",
        "cam",
        "camera",
        "cameras",
        "challenge",
        "cheat",
        "cloud",
        "color",
        "crossword",
        "cryptogram",
        "dictionary",
        "discover",
        "download",
        "feed",
        "feeder",
        "feeders",
        "game",
        "games",
        "helper",
        "live",
        "login",
        "menu",
        "night",
        "polygon",
        "powered",
        "pros",
        "reviews",
        "scrabble",
        "shop",
        "smart",
        "solar",
        "stream",
        "sudoku",
        "tools",
        "translator",
        "unlimited",
        "vision",
        "wildbird",
        "wireless",
        "word",
        "wordle",
        "words",
    }
    SECTION_HEADING_TOKENS = {
        "abstract",
        "appearance",
        "assumption",
        "assumptions",
        "background",
        "bibliography",
        "contents",
        "discussion",
        "examples",
        "external",
        "further",
        "heading",
        "headings",
        "historical",
        "introduction",
        "methods",
        "notes",
        "overview",
        "reading",
        "references",
        "related",
        "results",
        "section",
        "sections",
        "several",
        "terminology",
    }
    SOURCE_TERMS = {
        "britannica",
        "facebook",
        "fandom",
        "github",
        "google",
        "instagram",
        "linkedin",
        "nasa science",
        "researchgate",
        "stack exchange",
        "twitter",
        "wikipedia",
        "wikimedia commons",
        "wikia",
        "youtube",
    }
    GENERIC_SINGLE_TOKENS = {
        "abstract",
        "article",
        "caption",
        "camera",
        "cameras",
        "click",
        "content",
        "download",
        "episode",
        "figure",
        "home",
        "issue",
        "journal",
        "list",
        "metadata",
        "page",
        "paper",
        "publication",
        "season",
        "source",
        "table",
        "title",
        "video",
    }
    PAGE_CHROME_CONTEXT_TERMS = {
        "advertisement",
        "all rights reserved",
        "browse",
        "category",
        "categories",
        "cookie",
        "creative commons",
        "external links",
        "follow us",
        "footer",
        "home page",
        "language",
        "licenses",
        "log in",
        "menu",
        "navigation",
        "privacy policy",
        "related articles",
        "search results",
        "sign in",
        "site map",
        "subscribe",
        "table of contents",
        "terms of use",
        "word game",
        "word games",
    }

    def __init__(
        self,
        *,
        max_bridge_spans: int = 3,
        min_span_chars: int = 3,
        max_span_chars: int = 80,
    ) -> None:
        self.max_bridge_spans = max(1, max_bridge_spans)
        self.min_span_chars = max(1, min_span_chars)
        self.max_span_chars = max(self.min_span_chars, max_span_chars)

    def select(
        self,
        *,
        documents: list[Any],
        question: str,
        intent_plan: SearchIntentPlan | None = None,
    ) -> NextHopEvidenceSelection:
        """
        Select bridge spans from valid next-hop documents only.

        Args:
            - documents: RetrievedDocumentTrace-like objects.
            - question: Original task question.
            - intent_plan: Planner state, kept only for diagnostics.

        Returns:
            - NextHopEvidenceSelection: Clean bridge spans and rejected diagnostics.
        """
        question_key = self._match_key(question)
        selected: list[str] = []
        selected_document_ids: list[str] = []
        rejected: list[RejectedNextHopSpan] = []
        seen: set[str] = set()
        candidate_count = 0
        candidates: list[tuple[tuple[int, int, int, int, int, float], str, str, str]] = []

        for document in sorted(
            documents,
            key=lambda item: float(getattr(item, "retrieval_score", 0.0) or 0.0),
            reverse=True,
        ):
            document_id = normalize_text(str(getattr(document, "document_id", "") or ""))
            document_title = normalize_text(str(getattr(document, "title", "") or ""))
            document_text = normalize_text(str(getattr(document, "text", "") or ""))
            document_context = normalize_text(
                str(getattr(document, "utility_context", "") or "")
            ) or document_text
            if not bool(getattr(document, "valid_for_next_hop", False)):
                rejected.append(self._reject(document_id, "", "document_not_valid_for_next_hop"))
                continue
            if normalize_text(str(getattr(document, "support_level", "") or "")) != "bridge":
                rejected.append(self._reject(document_id, "", "document_not_bridge_support"))
                continue

            spans = sorted(
                list(getattr(document, "bridge_spans", []) or []),
                key=self._span_priority,
                reverse=True,
            )
            if not spans:
                rejected.append(self._reject(document_id, "", "document_without_bridge_spans"))
                continue

            for span in spans:
                candidate_count += 1
                cleaned, reason = self._clean_span(
                    span,
                    question_key=question_key,
                    document_title=document_title,
                    document_text=document_text,
                    document_context=document_context,
                    intent_plan=intent_plan,
                )
                if reason:
                    rejected.append(self._reject(document_id, str(span or ""), reason))
                    continue
                context = self._context_window(cleaned, document_context or document_text)
                if self._is_page_chrome_context(context):
                    rejected.append(self._reject(document_id, str(span or ""), "page_chrome_context"))
                    continue
                candidates.append(
                    (
                        self._candidate_priority(
                            cleaned,
                            question_key=question_key,
                            context=context,
                            sequence_tag=normalize_text(str(getattr(document, "sequence_tag", "") or "")),
                            retrieval_score=float(getattr(document, "retrieval_score", 0.0) or 0.0),
                        ),
                        cleaned,
                        document_id,
                        context,
                    )
                )

        for _priority, cleaned, document_id, _context in sorted(candidates, reverse=True):
            key = self._match_key(cleaned)
            if not key:
                rejected.append(self._reject(document_id, cleaned, "empty_match_key"))
                continue
            if key in seen or self._contained_by_existing(key, seen):
                rejected.append(self._reject(document_id, cleaned, "duplicate_or_contained_span"))
                continue
            selected.append(cleaned)
            selected_document_ids.append(document_id)
            seen.add(key)
            if len(selected) >= self.max_bridge_spans:
                return self._result(
                    selected=selected,
                    selected_document_ids=selected_document_ids,
                    rejected=rejected,
                    candidate_count=candidate_count,
                    intent_plan=intent_plan,
                )

        return self._result(
            selected=selected,
            selected_document_ids=selected_document_ids,
            rejected=rejected,
            candidate_count=candidate_count,
            intent_plan=intent_plan,
        )

    def _clean_span(
        self,
        span: object,
        *,
        question_key: str,
        document_title: str = "",
        document_text: str = "",
        document_context: str = "",
        intent_plan: SearchIntentPlan | None = None,
    ) -> tuple[str, str]:
        text = normalize_text(self._repair_escaped_unicode(str(span or ""))).strip(" \"'`.,;:")
        if not text:
            return "", "empty_span"
        if len(text) < self.min_span_chars:
            return "", "span_too_short"
        if len(text) > self.max_span_chars:
            return "", "span_too_long"
        if re.fullmatch(r"[\W_]+", text):
            return "", "punctuation_span"
        if re.fullmatch(r"\d+(?:\.\d+)?", text) or re.fullmatch(r"(?:18|19|20)\d{2}", text):
            return "", "pure_number_or_year"
        if self._is_partial_number_fragment(text):
            return "", "partial_number_fragment"

        key = self._match_key(text)
        if not key:
            return "", "empty_match_key"
        if not self._supported_by_body_context(
            text,
            document_text=document_text,
            document_context=document_context,
        ):
            return "", "span_not_in_body_context"
        if self._has_source_term(text):
            return "", "source_name_span"
        if self._is_clock_time_fragment(text):
            return "", "clock_time_fragment"
        if self._is_page_chrome_span(text):
            return "", "page_chrome_span"
        if self._is_category_or_navigation_span(text):
            return "", "category_or_navigation_span"
        if self._is_license_span(text):
            return "", "license_or_attribution_span"
        if self._is_page_feature_listing_span(text):
            return "", "page_feature_listing_span"
        if self._is_section_heading_span(text):
            return "", "section_heading_span"
        if self._is_title_glue_span(text):
            return "", "title_glue_span"
        if self._is_possessive_fragment(text):
            return "", "possessive_fragment"
        if self._has_bad_phrase_boundary(text):
            return "", "bad_phrase_boundary"
        if self._is_generic_single_token(text, intent_plan=intent_plan):
            return "", "generic_single_token"
        if (
            self._match_key(document_title)
            and key in self._match_key(document_title)
            and key not in self._match_key(document_text)
            and not self._has_value_signal(text)
            and not self._looks_like_named_entity(text)
        ):
            return "", "title_only_span"
        if question_key and key in question_key:
            return "", "question_echo_span"
        if self._noise_ratio(text) >= 0.5:
            return "", "noise_span"
        return text, ""

    def _result(
        self,
        *,
        selected: list[str],
        selected_document_ids: list[str],
        rejected: list[RejectedNextHopSpan],
        candidate_count: int,
        intent_plan: SearchIntentPlan | None,
    ) -> NextHopEvidenceSelection:
        rejected_reason_counts: dict[str, int] = {}
        for item in rejected:
            rejected_reason_counts[item.reason] = rejected_reason_counts.get(item.reason, 0) + 1
        return NextHopEvidenceSelection(
            bridge_spans=list(selected),
            selected_document_ids=list(dict.fromkeys(selected_document_ids)),
            rejected=list(rejected),
            metadata={
                "method": "global_ranked_strict_valid_bridge_span_selection",
                "candidate_count": candidate_count,
                "selected_count": len(selected),
                "rejected_count": len(rejected),
                "rejected_reason_counts": rejected_reason_counts,
                "answer_role": normalize_text(
                    str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
                ),
            },
        )

    def _reject(self, document_id: str, span: str, reason: str) -> RejectedNextHopSpan:
        return RejectedNextHopSpan(
            document_id=document_id,
            span=normalize_text(span),
            reason=reason,
        )

    def _contained_by_existing(self, key: str, seen: set[str]) -> bool:
        return any(key in existing or existing in key for existing in seen)

    def _noise_ratio(self, text: str) -> float:
        tokens = self._keywords(text)
        if not tokens:
            return 1.0
        noisy = sum(1 for token in tokens if token in self.NOISE_TERMS)
        return noisy / len(tokens)

    def _source_ratio(self, text: str) -> float:
        key = self._match_key(text)
        if not key:
            return 1.0
        hits = sum(1 for term in self.SOURCE_TERMS if term in key)
        return hits / max(1, len(key.split()))

    def _has_source_term(self, text: str) -> bool:
        key = self._match_key(text)
        return any(term in key for term in self.SOURCE_TERMS)

    def _span_priority(self, span: object) -> tuple[int, int]:
        text = normalize_text(str(span or ""))
        if self._has_value_signal(text):
            return (5, len(text))
        if self._has_source_term(text) or self._noise_ratio(text) >= 0.5:
            return (-2, -len(text))
        if self._looks_like_named_entity(text):
            return (4, len(text))
        words = self._keywords(text)
        if len(words) >= 2:
            return (3, len(text))
        return (1, len(text))

    def _candidate_priority(
        self,
        span: str,
        *,
        question_key: str,
        context: str,
        sequence_tag: str,
        retrieval_score: float,
    ) -> tuple[int, int, int, int, int, float]:
        span_score, span_length = self._span_priority(span)
        new_information_score = 1 if self._match_key(span) not in question_key else 0
        context_score = 1 if context else 0
        tag_score = {
            "<CONTINUE>": 2,
            "<FINISH>": 1,
            "<TERMINATE>": 0,
        }.get(normalize_text(sequence_tag), 0)
        return (span_score, new_information_score, tag_score, context_score, span_length, retrieval_score)

    def _has_value_signal(self, text: str) -> bool:
        return bool(
            re.search(
                r"(?<![A-Za-z0-9])[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|miles?|meters?|metres?|hours?|hrs?|minutes?|mins?|seconds?|secs?|m\^?3|m3|cubic\s+met(?:er|re)s?)\b",
                text,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(?<![A-Za-z0-9])\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?![A-Za-z0-9])",
                text,
            )
        )

    def _has_bad_phrase_boundary(self, text: str) -> bool:
        tokens = self._match_key(text).split()
        return bool(tokens and tokens[-1] in {"and", "for", "in", "of", "on", "the", "to"})

    def _is_page_chrome_span(self, text: str) -> bool:
        key = self._match_key(text)
        if not key:
            return True
        tokens = key.split()
        if any(
            term in key
            for term in {
                "accessibility help",
                "business technology list",
                "dictionary words",
                "home tv",
                "movies tv shows",
                "performance and security",
                "subslikescript",
                "text twist",
                "word checker word games",
                "word game",
                "wordle",
            }
        ):
            return True
        if tokens[-1:] == ["list"] and len(tokens) >= 2:
            return True
        if sum(1 for token in tokens if token in {"bahasa", "deutsch", "english", "espanol", "francais", "indonesia", "italiano", "language"}) >= 2:
            return True
        hits = sum(1 for token in tokens if token in self.NOISE_TERMS)
        return bool(tokens and hits / len(tokens) >= 0.34)

    def _is_license_span(self, text: str) -> bool:
        key = self._match_key(text)
        if not key:
            return False
        if any(
            phrase in key
            for phrase in {
                "attribution",
                "creative commons",
                "cc by",
                "international license",
                "licensed under",
                "reuse",
                "terms of use",
            }
        ):
            return True
        tokens = key.split()
        return "international" in tokens and any(
            token in tokens for token in {"attribution", "license", "commons"}
        )

    def _is_page_feature_listing_span(self, text: str) -> bool:
        key = self._match_key(text)
        if not key:
            return False
        tokens = key.split()
        if not tokens:
            return False
        hits = sum(1 for token in tokens if token in self.PAGE_FEATURE_TOKENS)
        if hits >= 2:
            return True
        if "bird" in tokens and any(token in tokens for token in {"feeder", "feeders"}):
            return True
        if any(token in tokens for token in {"camera", "cameras", "cam"}) and any(
            token in tokens
            for token in {"feed", "live", "night", "powered", "solar", "stream", "vision", "wireless"}
        ):
            return True
        if hits == 1 and len(tokens) <= 2 and not self._looks_like_named_entity(text):
            return True
        return False

    def _is_section_heading_span(self, text: str) -> bool:
        tokens = self._keywords(text)
        if not tokens:
            return True
        hits = sum(1 for token in tokens if token in self.SECTION_HEADING_TOKENS)
        if hits == len(tokens) and len(tokens) <= 4:
            return True
        if hits >= 2 and len(tokens) <= 4 and not self._has_value_signal(text):
            return True
        return False

    def _is_category_or_navigation_span(self, text: str) -> bool:
        key = self._match_key(text)
        if not key:
            return True
        if key.startswith(("category ", "categories ", "list of ", "template ")):
            return True
        if key.endswith((" category", " categories", " list", " navigation", " menu")):
            return True
        if any(
            phrase in key
            for phrase in {
                "category page",
                "contents page",
                "disambiguation page",
                "help page",
                "main page",
                "navigation menu",
                "related pages",
                "search results",
                "special page",
            }
        ):
            return True
        return False

    def _is_title_glue_span(self, text: str) -> bool:
        cleaned = normalize_text(text)
        if re.search(r"[a-z][A-Z][a-z]", cleaned):
            return True
        words = re.findall(r"[A-Za-z][A-Za-z']*", cleaned)
        if len(words) >= 2:
            glued = [word for word in words if re.search(r"[a-z][A-Z]", word)]
            if glued:
                return True
        return False

    def _is_possessive_fragment(self, text: str) -> bool:
        key = self._match_key(text)
        tokens = key.split()
        if len(tokens) <= 2 and normalize_text(text).endswith("'s"):
            return True
        if tokens and tokens[0] in {"a", "an", "the", "this", "that"} and len(tokens) <= 2:
            return True
        return False

    def _is_partial_number_fragment(self, text: str) -> bool:
        cleaned = normalize_text(text)
        return bool(re.match(r"^0{2,}\s+[A-Za-z]", cleaned))

    def _is_clock_time_fragment(self, text: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}:\d{2}", normalize_text(text)))

    def _is_generic_single_token(
        self,
        text: str,
        *,
        intent_plan: SearchIntentPlan | None = None,
    ) -> bool:
        tokens = self._keywords(text)
        if len(tokens) != 1:
            return False
        token = tokens[0]
        if token in self.GENERIC_SINGLE_TOKENS:
            return True
        intent_terms = self._intent_terms(intent_plan)
        if any(token in self._match_key(term).split() for term in intent_terms):
            return False
        return len(token) < 4

    def _looks_like_named_entity(self, text: str) -> bool:
        cleaned = normalize_text(text).strip()
        words = re.findall(r"[A-Za-z][A-Za-z'&.-]*", cleaned)
        if len(words) < 2:
            return False
        if any(word.casefold() in self.NOISE_TERMS for word in words):
            return False
        capitalized = sum(1 for word in words if word[:1].isupper())
        return capitalized >= max(1, len(words) - 1)

    def _intent_terms(self, intent_plan: SearchIntentPlan | None) -> list[str]:
        if intent_plan is None:
            return []
        terms: list[str] = []
        for name in ("target",):
            value = normalize_text(str(getattr(intent_plan, name, "") or ""))
            if value:
                terms.append(value)
        for name in ("must_include", "missing_terms", "completed_terms"):
            for value in getattr(intent_plan, name, []) or []:
                text = normalize_text(str(value or ""))
                if text and not self._is_internal_requirement(text):
                    terms.append(text)
        return terms

    def _is_internal_requirement(self, value: str) -> bool:
        text = str(value or "").strip().casefold()
        return text.startswith(
            (
                "answer_support:",
                "answer_candidate:",
                "preferred_domain:",
            )
        )

    def _supported_by_body_context(
        self,
        span: str,
        *,
        document_text: str,
        document_context: str,
    ) -> bool:
        body = normalize_text(" ".join(part for part in (document_context, document_text) if part))
        if not body:
            return False
        if self._contains_span(body, span):
            return True
        span_tokens = [
            token
            for token in self._keywords(span)
            if token not in self.NOISE_TERMS and token not in self.GENERIC_SINGLE_TOKENS
        ]
        if not span_tokens:
            return False
        body_tokens = set(self._keywords(body))
        required = max(1, min(len(span_tokens), int(round(len(span_tokens) * 0.8))))
        return sum(1 for token in span_tokens if token in body_tokens) >= required

    def _context_window(self, span: str, body: str, *, window_chars: int = 180) -> str:
        text = normalize_text(body)
        if not text:
            return ""
        needle = normalize_text(span)
        lower_text = text.casefold()
        lower_needle = needle.casefold()
        index = lower_text.find(lower_needle)
        if index < 0:
            tokens = self._keywords(needle)
            token = next((item for item in tokens if len(item) > 3), "")
            index = lower_text.find(token) if token else -1
        if index < 0:
            return ""
        start = max(0, index - window_chars)
        end = min(len(text), index + len(needle) + window_chars)
        return text[start:end].strip()

    def _is_page_chrome_context(self, context: str) -> bool:
        key = self._match_key(context)
        if not key:
            return True
        hits = sum(1 for term in self.PAGE_CHROME_CONTEXT_TERMS if term in key)
        if hits >= 2:
            return True
        tokens = key.split()
        if tokens:
            noise_hits = sum(1 for token in tokens if token in self.NOISE_TERMS)
            if noise_hits / len(tokens) >= 0.25:
                return True
        return False

    def _contains_span(self, text: str, span: str) -> bool:
        haystack = normalize_text(text).casefold()
        needle = normalize_text(span).casefold()
        if not haystack or not needle:
            return False
        if needle in haystack:
            return True
        key = self._match_key(span)
        return bool(key and key in self._match_key(text))

    def _repair_escaped_unicode(self, text: str) -> str:
        return (
            str(text or "")
            .replace("\\u2019", "'")
            .replace("\\u2018", "'")
            .replace("\\u201c", '"')
            .replace("\\u201d", '"')
            .replace("\\u2013", "-")
            .replace("\\u2014", "-")
        )

    def _keywords(self, text: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]{1,}", normalize_text(text).casefold()):
            token = token.strip("'_.-")
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _match_key(self, text: str) -> str:
        return " ".join(self._keywords(text))


__all__ = [
    "NextHopEvidenceSelection",
    "NextHopEvidenceSelector",
    "RejectedNextHopSpan",
]
