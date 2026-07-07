from __future__ import annotations

import re


class DocumentChunker:
    """
    將清理後正文切成適合 dense retrieval 的固定大小文字片段。

    Args:
        - max_chars: 每個 chunk 的最大字元數。
        - overlap_chars: 相鄰 chunk 保留的尾端重疊字元數。
        - min_chars: 匯出 chunk 的最小字元數。

    Returns:
        - DocumentChunker: 保留段落邊界並支援長段落切分的 chunker。
    """

    SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？.!?])\s+")

    def __init__(
        self,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 120,
        min_chars: int = 80,
    ) -> None:
        self.max_chars = max(100, max_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.max_chars // 3))
        self.min_chars = max(1, min(min_chars, self.max_chars))

    def chunk(self, text: str) -> list[str]:
        """
        切分單份正文。

        Args:
            - text: 已清理且保留換行的正文。

        Returns:
            - list[str]: 按原文順序排列的 chunks。
        """
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n|\n", str(text or ""))
            if paragraph.strip()
        ]
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(self._split_long_unit(paragraph))

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n{unit}".strip() if current else unit
            if len(candidate) <= self.max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            prefix = self._overlap_tail(current)
            candidate = f"{prefix}\n{unit}".strip() if prefix else unit
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                chunks.extend(self._fixed_windows(candidate)[:-1])
                current = self._fixed_windows(candidate)[-1]

        if current:
            chunks.append(current)
        return self._merge_short_tail(chunks)

    def _split_long_unit(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        sentences = [
            sentence.strip()
            for sentence in self.SENTENCE_BOUNDARY_RE.split(text)
            if sentence.strip()
        ]
        if len(sentences) <= 1:
            return self._fixed_windows(text)

        units: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= self.max_chars:
                current = candidate
                continue
            if current:
                units.append(current)
            if len(sentence) > self.max_chars:
                windows = self._fixed_windows(sentence)
                units.extend(windows[:-1])
                current = windows[-1]
            else:
                current = sentence
        if current:
            units.append(current)
        return units

    def _fixed_windows(self, text: str) -> list[str]:
        step = max(1, self.max_chars - self.overlap_chars)
        windows = [
            text[start : start + self.max_chars].strip()
            for start in range(0, len(text), step)
        ]
        return [window for window in windows if window]

    def _overlap_tail(self, text: str) -> str:
        if not text or self.overlap_chars <= 0:
            return ""
        return text[-self.overlap_chars :].lstrip()

    def _merge_short_tail(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2 or len(chunks[-1]) >= self.min_chars:
            return [chunk for chunk in chunks if len(chunk) >= self.min_chars]
        merged = f"{chunks[-2]}\n{chunks[-1]}".strip()
        if len(merged) <= self.max_chars:
            chunks[-2:] = [merged]
        return [chunk for chunk in chunks if len(chunk) >= self.min_chars]


__all__ = ["DocumentChunker"]
