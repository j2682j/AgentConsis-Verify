from __future__ import annotations

"""Hard source filtering without hand-crafted ranking scores."""

import os
import re
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from utils.network_utils import normalize_text, semantic_similarity_score

from ...config import SearchSourceCandidate
from .fetch_candidate_selector import FetchCandidateSelector, FetchSelectionResult
from .source_selection_signals import SourceSelectionSignalBuilder


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
    ACADEMIC_DOMAIN_MARKERS = (
        "scholar.google.com",
        "doi.org",
        "sciencedirect.com",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "tandfonline.com",
        "journals.plos.org",
        "researchgate.net",
        "ncbi.nlm.nih.gov",
        "semanticscholar.org",
        "jstor.org",
        "arxiv.org",
        "biorxiv.org",
        "core.ac.uk",
    )
    BENCHMARK_LEAK_MARKERS = (
        "assistants/gaia",
        "agentscope",
        "albertvillanova/answers",
        "huggingface.co/datasets",
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
    # SEO sites that mass-generate pages for long-tail question queries, plus
    # storefront pages: demoted in fetch priority, never hard-blocked.
    DEMOTED_DOMAIN_MARKERS = (
        "wordplays.com",
        "spellingcenter.com",
        "crossword",
        "etsy.com",
        "ebay.",
        "amazon.",
    )
    PRODUCT_PAGE_MARKERS = (
        "/market/",
        "shop_product",
        "/product/",
        "/dp/",
        "/itm/",
    )
    GENERIC_DOMAIN_LABELS = {
        "www", "com", "org", "net", "edu", "gov", "html", "co", "uk", "ac",
        "de", "info", "io", "blog", "shop", "web", "site", "index", "pages",
        "en", "m",
    }
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
    TASK_ID_RE = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )
    BENCHMARK_PATH_MARKERS = (
        "assistants/gaia",
        "agentscope",
        "huggingface.co/datasets",
        "webvoyager/data/gaia",
        "gaia-benchmark",
        "gaia_subset",
        "gaia-subset",
        "inspect_evals",
        "harbor-datasets",
        "albertvillanova/answers",
        "open deep researcher",
    )
    TASK_TRACE_MARKERS = (
        '"task_id"',
        "'task_id'",
        "task_id:",
        '"final_answer"',
        "'final_answer'",
        "final_answer:",
        '"expected_answer"',
        "'expected_answer'",
        "expected_answer:",
        "ground truth",
    )
    DIALOGUE_TRACE_MARKERS = (
        "role: user",
        "role: assistant",
        '"role": "user"',
        '"role":"user"',
        '"role": "assistant"',
        '"role":"assistant"',
        "initial plan",
        "we need answer",
        "final answer",
    )

    def __init__(
        self,
        *,
        max_urls_per_domain: int = 3,
        min_sources: int = 5,
        semantic_echo_threshold: float | None = None,
        lexical_echo_threshold: float | None = None,
        max_new_information_ratio: float | None = None,
        signal_builder: "SourceSelectionSignalBuilder | None" = None,
        fetch_selector: "FetchCandidateSelector | None" = None,
    ) -> None:
        self.signal_builder = signal_builder or SourceSelectionSignalBuilder()
        self.fetch_selector = fetch_selector or FetchCandidateSelector(
            demoted_domain_markers=self.DEMOTED_DOMAIN_MARKERS,
            product_page_markers=self.PRODUCT_PAGE_MARKERS,
        )
        self.last_fetch_selection = FetchSelectionResult()
        self.max_urls_per_domain = max(1, max_urls_per_domain)
        self.min_sources = max(0, min_sources)
        self.semantic_echo_threshold = (
            semantic_echo_threshold
            if semantic_echo_threshold is not None
            else float(os.getenv("SEARCH_SEMANTIC_ECHO_THRESHOLD", "0.90"))
        )
        self.lexical_echo_threshold = (
            lexical_echo_threshold
            if lexical_echo_threshold is not None
            else float(os.getenv("SEARCH_LEXICAL_ECHO_THRESHOLD", "0.25"))
        )
        self.max_new_information_ratio = (
            max_new_information_ratio
            if max_new_information_ratio is not None
            else float(os.getenv("SEARCH_MAX_ECHO_NEW_INFORMATION_RATIO", "0.65"))
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
            - question: 原始問題，用於 question echo 檢查與指名來源比對。
            - query_text_by_id: query id 到 query 文字的對應，用於跨 query 共識計數。
            - fetch_limit: 最多標記多少來源抓全文。

        Returns:
            - list[SearchSourceCandidate]: 通過 hard filter 的來源。
        """
        filtered: list[SearchSourceCandidate] = []
        soft_blocked: list[SearchSourceCandidate] = []
        seen_urls: set[str] = set()
        seen_document_ids: set[str] = set()
        seen_fingerprints: list[str] = []
        domain_counts: dict[str, int] = {}

        primary_by_url: dict[str, SearchSourceCandidate] = {}
        normalized_queries = {
            key: normalize_text(value).casefold()
            for key, value in (query_text_by_id or {}).items()
        }

        def register_query(target: SearchSourceCandidate, query_id: str) -> None:
            """Record that ``query_id`` also found this source.

            Consensus counts distinct queries, not repeated hits: the same
            query returning a URL twice says nothing new, whereas two
            different queries agreeing is real evidence the page is central.
            """
            if query_id and query_id not in target.matched_query_ids:
                target.matched_query_ids.append(query_id)
            distinct = {
                normalized_queries.get(item, item)
                for item in target.matched_query_ids
                if item
            }
            target.query_hit_count = len(distinct)

        for source in sources:
            self._reset_source_marks(source)
            canonical_url = self._canonical_url(source.url)
            domain = self._source_domain(source, canonical_url)
            source.domain = domain
            source.canonical_url = canonical_url
            register_query(source, source.query_id)
            if canonical_url and canonical_url in seen_urls:
                primary = primary_by_url.get(canonical_url)
                if primary is not None:
                    register_query(primary, source.query_id)
                    # Keep whichever copy carries more usable metadata.
                    if len(normalize_text(source.snippet)) > len(
                        normalize_text(primary.snippet)
                    ):
                        primary.snippet = source.snippet
                    if not normalize_text(primary.title):
                        primary.title = source.title
                    if not normalize_text(primary.source_hint):
                        primary.source_hint = source.source_hint
                self._mark_blocked(source, "duplicate_url")
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
                primary_by_url[canonical_url] = source
            document_id = self._canonical_document_identity(source, canonical_url)
            if document_id and document_id in seen_document_ids:
                self._mark_blocked(source, "duplicate_document")
                continue
            if document_id:
                seen_document_ids.add(document_id)

            safety_reason = self._source_safety_block_reason(
                source,
                question=question,
                include_raw_content=False,
            )
            if safety_reason:
                self._mark_blocked(source, safety_reason)
                continue

            fingerprint = self._text_fingerprint(source)
            if fingerprint and self._is_duplicate_text(fingerprint, seen_fingerprints):
                self._mark_blocked(source, "duplicate_text")
                if self._is_soft_block(source.block_reason):
                    soft_blocked.append(source)
                continue
            if fingerprint:
                seen_fingerprints.append(fingerprint)

            block_reason = self._block_reason(source)
            if block_reason:
                self._mark_blocked(source, block_reason)
                continue

            if self._is_question_echo_only(source, question):
                self._append_reason(source, "question_echo_only")

            if self._is_question_semantic_echo(source, question):
                self._append_reason(source, "question_semantic_echo")

            if source.source_kind == "web" and self._is_academic_domain(domain):
                source.source_kind = "academic"
                self._append_reason(source, "domain_reclassified:academic")

            domain_limit = self._domain_limit(source)
            if domain and domain_counts.get(domain, 0) >= domain_limit:
                self._mark_blocked(source, "domain_result_limit")
                if self._is_soft_block(source.block_reason):
                    soft_blocked.append(source)
                continue

            filtered.append(source)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if len(filtered) < self.min_sources:
            self._rescue_soft_blocked_sources(
                filtered=filtered,
                soft_blocked=soft_blocked,
                domain_counts=domain_counts,
            )

        filtered.sort(key=lambda item: (item.query_id, item.rank, item.source_id))
        # Signals first, then selection: the filter decides usability, the
        # selector decides which usable sources are worth a full-page fetch.
        self.signal_builder.build(
            filtered,
            question=question,
            query_text_by_id=query_text_by_id,
        )
        self.last_fetch_selection = self.fetch_selector.select(
            filtered,
            fetch_limit=fetch_limit,
        )
        return filtered

    def apply_post_fetch_safety(
        self,
        sources: list[SearchSourceCandidate],
        *,
        question: str = "",
    ) -> list[SearchSourceCandidate]:
        """
        在 full-page fetch 後檢查全文內容是否包含 benchmark leak 或答案爬取痕跡。

        Args:
            - sources: 已經通過 pre-fetch filter 的來源。
            - question: 原始問題，用於記錄 echo / new-information 訊號。

        Returns:
            - list[SearchSourceCandidate]: 移除不安全來源後的來源。
        """
        kept: list[SearchSourceCandidate] = []
        for source in sources:
            if source.blocked:
                continue
            safety_reason = self._source_safety_block_reason(
                source,
                question=question,
                include_raw_content=True,
            )
            if safety_reason:
                self._mark_blocked(source, safety_reason)
                continue
            kept.append(source)
        return kept

    def canonical_url(self, url: str) -> str:
        """
        將 URL 正規化，供跨搜尋輪次共用去重鍵。

        Args:
            - url: 原始 URL。

        Returns:
            - str: 移除 fragment 與追蹤參數後的 canonical URL。
        """
        return self._canonical_url(url)

    def canonical_document_identity(self, source: SearchSourceCandidate) -> str:
        """Return an identity shared by alternate URLs for the same document."""

        return self._canonical_document_identity(
            source,
            self._canonical_url(source.url),
        )

    def _reset_source_marks(self, source: SearchSourceCandidate) -> None:
        source.blocked = False
        source.block_reason = ""
        source.filter_reasons = []
        source.should_fetch_full_page = False

    def _mark_blocked(self, source: SearchSourceCandidate, reason: str) -> None:
        source.blocked = True
        source.block_reason = reason
        self._append_reason(source, reason)

    def _append_reason(self, source: SearchSourceCandidate, reason: str) -> None:
        if reason not in source.filter_reasons:
            source.filter_reasons.append(reason)

    def _unblock_rescued(self, source: SearchSourceCandidate) -> None:
        original_reason = source.block_reason
        source.blocked = False
        source.block_reason = ""
        source.filter_reasons.append(f"rescued_soft_block:{original_reason}")

    def _is_soft_block(self, reason: str) -> bool:
        return reason in {
            "duplicate_text",
            "question_echo_only",
            "question_semantic_echo",
            "domain_result_limit",
        }

    def _rescue_soft_blocked_sources(
        self,
        *,
        filtered: list[SearchSourceCandidate],
        soft_blocked: list[SearchSourceCandidate],
        domain_counts: dict[str, int],
    ) -> None:
        needed = max(0, self.min_sources - len(filtered))
        if not needed:
            return
        for source in sorted(soft_blocked, key=lambda item: (item.query_id, item.rank, item.source_id)):
            if needed <= 0:
                break
            if not source.url or not self._is_soft_block(source.block_reason):
                continue
            domain = source.domain or self._source_domain(source, self._canonical_url(source.url))
            if domain and domain_counts.get(domain, 0) >= self.max_urls_per_domain + 1:
                continue
            self._unblock_rescued(source)
            filtered.append(source)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            needed -= 1

    def _block_reason(self, source: SearchSourceCandidate) -> str:
        domain = source.domain.lower()
        haystack = self._haystack(source).lower()

        if any(marker in domain for marker in self.BLOCKED_DOMAIN_MARKERS):
            return "blocked_domain"
        if any(marker in haystack for marker in self.BENCHMARK_PATH_MARKERS):
            return "benchmark_or_answer_leak"
        if any(marker in haystack for marker in self.NO_RESULT_MARKERS):
            return "no_result_or_login_page"
        if "github.com" in domain and "gaia" in haystack:
            return "gaia_repository_source"
        if any(marker in source.url.lower() for marker in self.GENERIC_PAGE_MARKERS):
            source.filter_reasons.append("generic_page")
        return ""

    def _source_safety_block_reason(
        self,
        source: SearchSourceCandidate,
        *,
        question: str,
        include_raw_content: bool,
    ) -> str:
        text = self._safety_text(source, include_raw_content=include_raw_content)
        lowered = text.lower()
        phase = "post_fetch" if include_raw_content else "pre_fetch"

        self._record_question_overlap_signals(source, question, text, phase=phase)

        if self.TASK_ID_RE.search(text):
            self._append_reason(source, f"{phase}:task_id_like_uuid")
            return "benchmark_task_id_leak"
        if any(marker in lowered for marker in self.BENCHMARK_PATH_MARKERS):
            self._append_reason(source, f"{phase}:benchmark_path_marker")
            return "benchmark_or_answer_leak"
        if self._has_task_trace(lowered):
            self._append_reason(source, f"{phase}:task_trace_marker")
            return "benchmark_task_trace_leak"
        if self._has_dialogue_trace(lowered):
            self._append_reason(source, f"{phase}:dialogue_trace_marker")
            return "benchmark_dialogue_trace_leak"
        return ""

    def _safety_text(
        self,
        source: SearchSourceCandidate,
        *,
        include_raw_content: bool,
    ) -> str:
        parts = [source.title, source.url, source.domain, source.snippet]
        if include_raw_content:
            parts.append(source.raw_content[:12000])
        return normalize_text(" ".join(str(part or "") for part in parts))

    def _has_task_trace(self, lowered: str) -> bool:
        trace_hits = sum(1 for marker in self.TASK_TRACE_MARKERS if marker in lowered)
        json_like = "{" in lowered and "}" in lowered and (":" in lowered or "," in lowered)
        gaia_like = "gaia" in lowered or "task_id" in lowered
        return trace_hits >= 2 and (json_like or gaia_like)

    def _has_dialogue_trace(self, lowered: str) -> bool:
        role_trace = (
            ("role: user" in lowered or '"role": "user"' in lowered or '"role":"user"' in lowered)
            and (
                "role: assistant" in lowered
                or '"role": "assistant"' in lowered
                or '"role":"assistant"' in lowered
            )
        )
        reasoning_trace = (
            ("initial plan" in lowered or "we need answer" in lowered)
            and ("final answer" in lowered or "expected answer" in lowered)
        )
        return role_trace or reasoning_trace

    def _record_question_overlap_signals(
        self,
        source: SearchSourceCandidate,
        question: str,
        source_text: str,
        *,
        phase: str,
    ) -> None:
        question_terms = self._keywords(question)
        source_terms = self._keywords(source_text)
        if not question_terms or not source_terms:
            return
        lexical_overlap = len(question_terms & source_terms) / max(1, len(question_terms))
        new_information_ratio = len(source_terms - question_terms) / max(1, len(source_terms))
        if lexical_overlap >= self.lexical_echo_threshold:
            self._append_reason(source, f"{phase}:question_overlap={lexical_overlap:.3f}")
        if new_information_ratio <= self.max_new_information_ratio:
            self._append_reason(source, f"{phase}:low_new_information={new_information_ratio:.3f}")

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

    def _is_question_semantic_echo(self, source: SearchSourceCandidate, question: str) -> bool:
        question_text = normalize_text(question)
        source_text = self._content_text(source)
        if not question_text or not source_text:
            return False

        question_terms = self._keywords(question_text)
        source_terms = self._keywords(source_text)
        if not question_terms or not source_terms:
            return False

        lexical_overlap = len(question_terms & source_terms) / max(1, len(question_terms))
        new_information_ratio = len(source_terms - question_terms) / max(1, len(source_terms))
        semantic_similarity = self._semantic_similarity(question_text, source_text)

        source.filter_reasons.append(f"semantic_echo={semantic_similarity:.3f}")
        source.filter_reasons.append(f"lexical_overlap={lexical_overlap:.3f}")
        source.filter_reasons.append(f"new_information_ratio={new_information_ratio:.3f}")

        semantic_echo = semantic_similarity >= self.semantic_echo_threshold
        lexical_echo = lexical_overlap >= self.lexical_echo_threshold
        low_new_information = new_information_ratio <= self.max_new_information_ratio

        # A result is treated as question echo only when it is both very close
        # to the question and contributes little new lexical information.
        if semantic_echo and lexical_echo and low_new_information:
            return True

        # Some benchmark leaks include small wrappers around the copied
        # question. This fallback catches near-exact lexical copies even when
        # the embedding model is unavailable or conservative.
        if (
            lexical_overlap >= self.lexical_echo_threshold
            and new_information_ratio <= self.max_new_information_ratio
        ):
            return True

        return False

    def _semantic_similarity(self, question: str, source_text: str) -> float:
        try:
            score = semantic_similarity_score(question, source_text[:1600])
        except Exception:
            return 0.0
        if score is None:
            return 0.0
        return float(score)

    def _keywords(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower())
        stopwords = {"the", "and", "for", "with", "from", "what", "which", "who", "when", "where", "why", "how"}
        return {token for token in tokens if token not in stopwords and len(token) > 2}

    def _haystack(self, source: SearchSourceCandidate) -> str:
        return normalize_text(" ".join([source.title, source.url, source.snippet, source.raw_content[:1200]]))

    def _content_text(self, source: SearchSourceCandidate) -> str:
        return normalize_text(" ".join([source.title, source.snippet, source.raw_content[:1200]]))

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

    def _canonical_document_identity(
        self,
        source: SearchSourceCandidate,
        canonical_url: str,
    ) -> str:
        parsed = urlparse(canonical_url or source.url)
        domain = parsed.netloc.casefold().removeprefix("www.")
        path = parsed.path.casefold().rstrip("/")
        if "arxiv.org" in domain:
            match = re.search(r"/(?:abs|pdf)/([^/?]+?)(?:\.pdf)?$", path)
            if match:
                return f"arxiv:{match.group(1)}"
        doi_match = re.search(r"10\.\d{4,9}/[^\s?#]+", canonical_url, re.IGNORECASE)
        if doi_match:
            return "doi:" + doi_match.group(0).rstrip("./").casefold()
        return canonical_url.casefold()

    def _is_academic_domain(self, domain: str) -> bool:
        if not domain:
            return False
        return any(marker in domain for marker in self.ACADEMIC_DOMAIN_MARKERS)

    def _domain_limit(self, source: SearchSourceCandidate) -> int:
        if source.source_kind in {"academic", "collection"}:
            return self.max_urls_per_domain + 3
        if source.required_content in {
            "pdf_text",
            "pdf_figure",
            "collection_records",
        }:
            return self.max_urls_per_domain + 3
        return self.max_urls_per_domain

    def _source_domain(
        self,
        source: SearchSourceCandidate,
        canonical_url: str,
    ) -> str:
        domain = normalize_text(source.domain).lower()
        if domain:
            return domain
        parsed = urlparse(canonical_url or source.url)
        return parsed.netloc.lower()


__all__ = ["SourceFilter"]
