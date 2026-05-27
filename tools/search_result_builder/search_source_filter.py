from __future__ import annotations

"""Filter search sources before evidence extraction."""

from .config import SearchSourceCandidate


class SourceFilter:
    """
    Filter low-value, duplicated, or benchmark-leak search sources.

    Args:
        - None.

    Returns:
        - SourceFilter: Reusable source filtering service.
    """

    BLOCKED_DOMAIN_MARKERS = (
        "huggingface.co",
        "github.com",
        "gitlab.com",
        "paperswithcode.com",
    )
    BLOCKED_TEXT_MARKERS = (
        "gaia",
        "benchmark",
        "dataset",
        "agentquest",
        "openresearcher",
        "web-bench",
        "final answer",
    )
    LOW_TRUST_DOMAIN_MARKERS = (
        "quora.com",
        "pinterest.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
    )

    def filter_sources(self, sources: list[SearchSourceCandidate]) -> list[SearchSourceCandidate]:
        """
        Return usable sources and mark blocked sources in-place.

        Args:
            - sources: Search source candidates created from structured search results.

        Returns:
            - list[SearchSourceCandidate]: Sources that passed filtering.
        """
        filtered: list[SearchSourceCandidate] = []
        seen_urls: set[str] = set()

        for source in sources:
            url_key = source.url.strip().lower()
            if url_key and url_key in seen_urls:
                source.blocked = True
                source.block_reason = "duplicate_url"
                continue
            if url_key:
                seen_urls.add(url_key)

            block_reason = self._block_reason(source)
            if block_reason:
                source.blocked = True
                source.block_reason = block_reason
                continue

            filtered.append(source)

        return filtered

    def _block_reason(self, source: SearchSourceCandidate) -> str:
        domain = source.domain.lower()
        haystack = " ".join(
            [source.title, source.url, source.snippet, source.raw_content[:500]]
        ).lower()

        if any(marker in domain for marker in self.LOW_TRUST_DOMAIN_MARKERS):
            return "low_trust_domain"

        if any(marker in domain for marker in self.BLOCKED_DOMAIN_MARKERS) and any(
            marker in haystack for marker in self.BLOCKED_TEXT_MARKERS
        ):
            return "benchmark_or_dataset_source"

        if "github.com" in domain and "gaia" in haystack:
            return "gaia_repository_source"

        return ""


__all__ = ["SourceFilter"]
