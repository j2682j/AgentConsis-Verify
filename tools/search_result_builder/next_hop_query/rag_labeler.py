from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.network_utils import normalize_text


@dataclass
class RAGLabelResult:
    """
    儲存 EfficientRAG labeler 對一段文字的 useful / useless 標註。

    Args:
        - label: useful 或 useless。
        - kept_tokens: 被保留的 tokens。
        - dropped_tokens: 被捨棄的 tokens。
        - metadata: 額外診斷資訊。

    Returns:
        - RAGLabelResult: labeler 標註結果。
    """

    label: str
    kept_tokens: list[str] = field(default_factory=list)
    dropped_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class EfficientRAGLabelerAdapter:
    """
    EfficientRAG labeler adapter，第一版使用 deterministic fallback。

    Args:
        - None.

    Returns:
        - EfficientRAGLabelerAdapter: useful / useless token labeler。
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
        "this",
        "that",
        "are",
        "was",
        "were",
    }

    def label_text(
        self,
        *,
        question: str,
        text: str,
        useful_probability: float,
        threshold: float,
    ) -> RAGLabelResult:
        """
        標註一段 evidence text 是否 useful。

        Args:
            - question: 原始問題。
            - text: evidence chunk。
            - useful_probability: Helpfulness Expert 分數。
            - threshold: useful threshold。

        Returns:
            - RAGLabelResult: useful / useless label result。
        """
        question_terms = set(self._ordered_keywords(question))
        text_terms = self._ordered_keywords(text)
        kept_tokens = [token for token in text_terms if token in question_terms][:12]
        if not kept_tokens and useful_probability >= threshold:
            kept_tokens = text_terms[:12]
        dropped_tokens = [token for token in text_terms if token not in kept_tokens][:20]
        label = "useful" if useful_probability >= threshold and kept_tokens else "useless"
        return RAGLabelResult(
            label=label,
            kept_tokens=kept_tokens,
            dropped_tokens=dropped_tokens,
            metadata={
                "method": "efficientrag_labeler_fallback",
                "useful_probability": useful_probability,
                "threshold": threshold,
            },
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


__all__ = ["EfficientRAGLabelerAdapter", "RAGLabelResult"]
