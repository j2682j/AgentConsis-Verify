from __future__ import annotations

import re

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, SearchSourceCandidate


class EvidenceExtractor:
    """
    Extract compact, source-linked evidence chunks from search sources.

    Args:
        - None.

    Returns:
        - EvidenceExtractor: Rule-based evidence extraction service.
    """

    STOPWORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "and",
        "or",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "answer",
        "question",
    }

    def extract(
        self,
        *,
        question: str,
        sources: list[SearchSourceCandidate],
        max_items: int = 8,
        max_chars_per_item: int = 420,
    ) -> list[EvidenceItem]:
        """
        Extract and rank evidence items from unblocked sources.

        Args:
            - question: Original task question.
            - sources: Filtered search source candidates.
            - max_items: Maximum number of evidence items to return.
            - max_chars_per_item: Maximum characters per evidence item.

        Returns:
            - list[EvidenceItem]: Ranked evidence chunks with source IDs.
        """
        question_terms = self._keywords(question)
        scored: list[tuple[float, int, EvidenceItem]] = []

        for source_index, source in enumerate(sources):
            text = source.raw_content or source.snippet
            chunks = self._split_chunks(text)
            if not chunks and source.snippet:
                chunks = [source.snippet]

            for chunk_index, chunk in enumerate(chunks):
                cleaned = normalize_text(chunk)
                if len(cleaned) < 40:
                    continue

                matched_terms = sorted(term for term in question_terms if term in cleaned.lower())
                score = self._score_chunk(
                    chunk=cleaned,
                    title=source.title,
                    question_terms=question_terms,
                    matched_terms=matched_terms,
                    source=source,
                )
                if score <= 0:
                    continue

                clipped = cleaned[:max_chars_per_item].strip()
                if len(cleaned) > max_chars_per_item:
                    clipped += " ..."

                evidence = EvidenceItem(
                    evidence_id=f"E{len(scored) + 1}",
                    source_id=source.source_id,
                    query_id=source.query_id,
                    text=clipped,
                    title=source.title,
                    url=source.url,
                    relevance_score=score,
                    matched_terms=matched_terms[:12],
                    extracted_answer="",
                )
                scored.append((score, source_index * 1000 + chunk_index, evidence))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [item for _, _, item in scored[:max_items]]
        for index, item in enumerate(selected, start=1):
            item.evidence_id = f"E{index}"
        return selected

    def _split_chunks(self, text: str) -> list[str]:
        cleaned = str(text or "").replace("\\n", "\n")
        cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
        if paragraphs:
            return paragraphs

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            current.append(sentence)
            current_len += len(sentence)
            if current_len >= 360:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _score_chunk(
        self,
        *,
        chunk: str,
        title: str,
        question_terms: set[str],
        matched_terms: list[str],
        source: SearchSourceCandidate,
    ) -> float:
        lower = chunk.lower()
        title_lower = title.lower()
        score = float(len(set(matched_terms)) * 2)

        if len(set(matched_terms)) >= 2:
            score += 3
        if any(term in title_lower for term in question_terms):
            score += 1.5
        if re.search(r"\b(?:19|20)\d{2}\b", lower):
            score += 0.5
        if re.search(r"\b\d+(?:\.\d+)?\b", lower):
            score += 0.25
        if source.rerank_score:
            score += min(float(source.rerank_score), 5.0) * 0.1
        if "search result" in lower or "cookie" in lower:
            score -= 2

        return score

    def _keywords(self, question: str) -> set[str]:
        lowered = re.sub(r"[^\w\s]", " ", normalize_text(question).lower())
        return {
            token
            for token in lowered.split()
            if len(token) > 2 and token not in self.STOPWORDS
        }



__all__ = ["EvidenceExtractor"]
