from __future__ import annotations

"""Hard source filtering without hand-crafted ranking scores."""

import re
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from utils.network_utils import normalize_text

from ...config import SearchSourceCandidate


class SourceFilter:
    """
    對搜尋來源做硬性過濾，不計算手刻來源分數。

    Args:
        - None.

    Returns:
        - SourceFilter: 可重複使用的 hard source filter。
    """

    BLOCKED_DOMAIN_MARKERS = (
        "quora.com",
        "pinterest.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "linkedin.com",
    )
    BENCHMARK_LEAK_MARKERS = (
        "assistants/gaia",
        "inspect_evals",
        "gaia benchmark",
        "gaia-benchmark",
        "gaia_subset",
        "gaia-subset",
        "webvoyager/data/gaia",
        "harbor-datasets",
        "open deep researcher",
        "final answer",
        "expected answer",
        "answer key",
        "leaderboard",
        "evaluation",
    )
    NO_RESULT_MARKERS = (
        "couldn't find a match",
        "could not find a match",
        "no results found",
        "did not match any documents",
        "missing:",
        "login required",
    )
    GENERIC_PAGE_MARKERS = (
        "/search",
        "/tag/",
        "/tags/",
        "/category/",
        "/categories/",
        "/topics/",
        "/archive",
        "/login",
        "/signin",
        "/signup",
    )

    def filter_sources(
        self,
        sources: list[SearchSourceCandidate],
        *,
        question: str = "",
        query_text_by_id: dict[str, str] | None = None,
        fetch_limit: int = 6,
    ) -> list[SearchSourceCandidate]:
        """
        移除不可用 source，並依原始 rank 標記需要抓全文的來源。

        Args:
            - sources: search tool 回傳的 source candidates。
            - question: 原始問題，目前只用於 question echo hard check。
            - query_text_by_id: query id 到 query 文字的對應，目前 filter 不使用。
            - fetch_limit: 最多標記多少來源抓全文。

        Returns:
            - list[SearchSourceCandidate]: 通過 hard filter 的來源。
        """
        del query_text_by_id
        filtered: list[SearchSourceCandidate] = []
        seen_urls: set[str] = set()
        seen_fingerprints: list[str] = []

        for source in sources:
            self._reset_source_marks(source)
            canonical_url = self._canonical_url(source.url)
            if canonical_url and canonical_url in seen_urls:
                self._mark_blocked(source, "duplicate_url")
                continue
            if canonical_url:
                seen_urls.add(canonical_url)

            fingerprint = self._text_fingerprint(source)
            if fingerprint and self._is_duplicate_text(fingerprint, seen_fingerprints):
                self._mark_blocked(source, "duplicate_text")
                continue
            if fingerprint:
                seen_fingerprints.append(fingerprint)

            block_reason = self._block_reason(source)
            if block_reason:
                self._mark_blocked(source, block_reason)
                continue

            if self._is_question_echo_only(source, question):
                self._mark_blocked(source, "question_echo_only")
                continue

            filtered.append(source)

        filtered.sort(key=lambda item: (item.query_id, item.rank, item.source_id))
        self._mark_fetch_candidates(filtered, fetch_limit=fetch_limit)
        return filtered

    def _reset_source_marks(self, source: SearchSourceCandidate) -> None:
        source.blocked = False
        source.block_reason = ""
        source.filter_reasons = []
        source.should_fetch_full_page = False

    def _mark_blocked(self, source: SearchSourceCandidate, reason: str) -> None:
        source.blocked = True
        source.block_reason = reason
        source.filter_reasons.append(reason)

    def _block_reason(self, source: SearchSourceCandidate) -> str:
        domain = source.domain.lower()
        haystack = self._haystack(source).lower()

        if any(marker in domain for marker in self.BLOCKED_DOMAIN_MARKERS):
            return "blocked_domain"
        if any(marker in haystack for marker in self.BENCHMARK_LEAK_MARKERS):
            return "benchmark_or_answer_leak"
        if any(marker in haystack for marker in self.NO_RESULT_MARKERS):
            return "no_result_or_login_page"
        if "github.com" in domain and "gaia" in haystack:
            return "gaia_repository_source"
        if any(marker in source.url.lower() for marker in self.GENERIC_PAGE_MARKERS):
            source.filter_reasons.append("generic_page")
        return ""

    def _mark_fetch_candidates(
        self,
        sources: list[SearchSourceCandidate],
        *,
        fetch_limit: int,
    ) -> None:
        marked = 0
        for source in sources:
            if marked >= fetch_limit:
                break
            if source.blocked or source.fetched or self._has_full_page_content(source):
                continue
            if not source.url:
                continue
            source.should_fetch_full_page = True
            source.filter_reasons.append("fetch_candidate")
            marked += 1

    def _has_full_page_content(self, source: SearchSourceCandidate) -> bool:
        raw = normalize_text(source.raw_content)
        snippet = normalize_text(source.snippet)
        return bool(raw and raw != snippet and len(raw) > len(snippet) + 120)

    def _is_question_echo_only(self, source: SearchSourceCandidate, question: str) -> bool:
        normalized_question = normalize_text(question).lower()
        haystack = self._haystack(source).lower()
        if not normalized_question or not haystack:
            return False
        if normalized_question not in haystack:
            return False
        question_terms = self._keywords(normalized_question)
        text_terms = self._keywords(haystack)
        novelty_terms = text_terms - question_terms
        if len(novelty_terms) <= 3:
            return True
        return False

    def _keywords(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower())
        stopwords = {"the", "and", "for", "with", "from", "what", "which", "who", "when", "where", "why", "how"}
        return {token for token in tokens if token not in stopwords and len(token) > 2}

    def _haystack(self, source: SearchSourceCandidate) -> str:
        return normalize_text(" ".join([source.title, source.url, source.snippet, source.raw_content[:1200]]))

    def _text_fingerprint(self, source: SearchSourceCandidate) -> str:
        text = normalize_text(" ".join([source.title, source.snippet])).lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        tokens = [token for token in text.split() if len(token) > 2]
        return " ".join(tokens[:40])

    def _is_duplicate_text(self, fingerprint: str, seen_fingerprints: list[str]) -> bool:
        if not fingerprint:
            return False
        for seen in seen_fingerprints:
            if fingerprint == seen:
                return True
            if SequenceMatcher(None, fingerprint, seen).ratio() >= 0.9:
                return True
        return False

    def _canonical_url(self, url: str) -> str:
        parsed = urlparse(normalize_text(url))
        if not parsed.netloc:
            return normalize_text(url).lower()
        keep_params = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            if key.lower().startswith("utm_") or key.lower() in {"fbclid", "gclid"}:
                continue
            keep_params.append((key, value))
        return urlunparse(
            (
                parsed.scheme.lower() or "https",
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                urlencode(keep_params),
                "",
            )
        )


__all__ = ["SourceFilter"]
