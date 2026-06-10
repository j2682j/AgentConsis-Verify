from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.network_utils import normalize_text

from ..config import EvidenceItem


@dataclass
class RAGFilterResult:
    """
    儲存 EfficientRAG filter 產生 next-hop query 的結果。

    Args:
        - query: 產生出的 next-hop query。
        - kept_question_tokens: 從問題保留的 tokens。
        - kept_evidence_tokens: 從 useful evidence 保留的 tokens。
        - fallback_used: 是否使用 fallback。
        - metadata: 額外診斷資訊。

    Returns:
        - RAGFilterResult: next-hop query filter 結果。
    """

    query: str
    kept_question_tokens: list[str] = field(default_factory=list)
    kept_evidence_tokens: list[str] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class EfficientRAGFilterAdapter:
    """
    EfficientRAG filter adapter，第一版用 useful evidence tokens 組合 next-hop query。

    Args:
        - max_question_tokens: query 中最多保留多少 question tokens。
        - max_evidence_tokens: query 中最多保留多少 evidence tokens。

    Returns:
        - EfficientRAGFilterAdapter: next-hop query filter。
    """

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "answer",
        "question",
    }

    def __init__(self, *, max_question_tokens: int = 6, max_evidence_tokens: int = 8) -> None:
        self.max_question_tokens = max_question_tokens
        self.max_evidence_tokens = max_evidence_tokens

    def build_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> RAGFilterResult:
        """
        用問題 tokens 與 useful evidence tokens 產生 next-hop query。

        Args:
            - question: 原始問題。
            - evidence_items: useful evidence items。

        Returns:
            - RAGFilterResult: next-hop query 結果。
        """
        question_tokens = self._ordered_keywords(question)[: self.max_question_tokens]
        evidence_tokens: list[str] = []
        for item in evidence_items:
            evidence_tokens.extend(item.matched_terms)
            evidence_tokens.extend(self._ordered_keywords(item.text)[:4])
            evidence_tokens = self._dedupe(evidence_tokens)
            if len(evidence_tokens) >= self.max_evidence_tokens:
                break
        evidence_tokens = evidence_tokens[: self.max_evidence_tokens]
        parts = self._dedupe(evidence_tokens + question_tokens)
        query = normalize_text(" ".join(parts))[:300]
        fallback_used = False
        if not query:
            query = normalize_text(question)
            fallback_used = True
        return RAGFilterResult(
            query=query,
            kept_question_tokens=question_tokens,
            kept_evidence_tokens=evidence_tokens,
            fallback_used=fallback_used,
            metadata={"method": "efficientrag_filter_fallback"},
        )

    def _ordered_keywords(self, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower()):
            if token in self.STOPWORDS or len(token) <= 2 or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = normalize_text(value).lower()
            if key and key not in seen:
                result.append(value)
                seen.add(key)
        return result


__all__ = ["EfficientRAGFilterAdapter", "RAGFilterResult"]
