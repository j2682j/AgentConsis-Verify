"""
Deduplication via n-gram similarity for SEER.
    - n-gram Tokenization：
        將文字拆成 n-gram，例如 2-gram 會以連續 2 個字元或 token 作為比對單位。
    - Similarity Calculation:
        透過 n-gram overlap / Jaccard similarity 判斷兩段 evidence 是否高度重複。
"""

from __future__ import annotations


from utils.network_utils import normalize_text


class NgramDeduplicator:
    """
    N-gram Deduplicator
    - n-gram tokenization
    - similarity calculation
    """
    def __init__(self, n=2):
        self.n = max(1, int(n))

    # 1. n-gram Tokenization
    def ngram_tokenize(self, text):
        """
        將文字切成 n-gram。

        Args:
            - text (str): 輸入文字。

        Returns:
            - list[str]: n-gram 清單。
        """
        text = normalize_text(text).lower()
        if not text:
            return []
        if len(text) <= self.n:
            return [text]
        return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]

    # 2. Similarity Calculation
    def calculate_similarity(self, text1, text2):
        """
        使用 n-gram Jaccard similarity 計算兩段文字的相似度。

        Args:
            - text1 (str): 文字 1。
            - text2 (str): 文字 2。

        Returns:
            - float: 相似度分數，範圍為 0 到 1。

        """
        grams1 = set(self.ngram_tokenize(text1))
        grams2 = set(self.ngram_tokenize(text2))
        if not grams1 and not grams2:
            return 1.0
        if not grams1 or not grams2:
            return 0.0
        return len(grams1 & grams2) / len(grams1 | grams2)

    def is_duplicate(self, text1: str, text2: str, *, threshold: float = 0.82) -> bool:
        """
        判斷兩段文字是否為近似重複 evidence。

        Args:
            - text1: 第一段 evidence 文字。
            - text2: 第二段 evidence 文字。
            - threshold: 判定為重複的 similarity 閾值。

        Returns:
            - bool: True 表示兩段文字高度相似。
        """
        return self.calculate_similarity(text1, text2) >= threshold