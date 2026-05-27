from __future__ import annotations

import re

from utils.network_utils import normalize_text

from .config import CandidateAnswer, EvidenceItem, SearchSourceCandidate


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


class CandidateExtractor:
    """
    Extract answer candidates from evidence items and connect them to evidence IDs.

    Args:
        - None.

    Returns:
        - CandidateExtractor: Rule-based candidate answer extraction service.
    """

    STOP_CANDIDATES = {
        "Wikipedia",
        "YouTube",
        "Google",
        "Facebook",
        "Twitter",
        "LinkedIn",
        "Amazon",
        "Home",
        "Official",
        "Source",
        "These",
        "This",
        "That",
        "There",
        "Search",
        "Result",
        "Results",
    }

    STOPWORDS = EvidenceExtractor.STOPWORDS | {"about", "notes", "source"}

    def extract_candidates(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
        sources: list[SearchSourceCandidate],
        max_candidates: int = 5,
    ) -> list[CandidateAnswer]:
        """
        Extract likely final-answer candidates from evidence chunks.

        Args:
            - question: Original task question.
            - evidence_items: Evidence chunks selected from search sources.
            - sources: Search sources used to create evidence.
            - max_candidates: Maximum candidate answers to return.

        Returns:
            - list[CandidateAnswer]: Ranked candidates linked to evidence/source IDs.
        """
        question_type = self._question_type(question)
        question_keywords = self._keywords(question)
        source_by_id = {source.source_id: source for source in sources}
        grouped: dict[str, CandidateAnswer] = {}

        for evidence in evidence_items:
            for answer in self._extract_answers(evidence.text, question_type=question_type):
                answer = normalize_text(answer).strip(" .,;:-")
                if not self._valid_answer(answer):
                    continue

                key = self._answer_key(answer)
                source = source_by_id.get(evidence.source_id)
                keyword_hits = sum(
                    1
                    for keyword in question_keywords
                    if keyword in evidence.text.lower() or keyword in evidence.title.lower()
                )
                score = evidence.relevance_score + min(keyword_hits, 4)

                candidate = grouped.get(key)
                if candidate is None:
                    candidate = CandidateAnswer(
                        answer=answer,
                        answer_type=question_type,
                        support_count=0,
                        verification_score=0.0,
                        evidence_ids=[],
                        source_ids=[],
                        verified=False,
                    )
                    grouped[key] = candidate

                candidate.support_count += 1
                candidate.verification_score += score
                if evidence.evidence_id not in candidate.evidence_ids:
                    candidate.evidence_ids.append(evidence.evidence_id)
                if source and source.source_id not in candidate.source_ids:
                    candidate.source_ids.append(source.source_id)
                candidate.verified = bool(candidate.evidence_ids)

        candidates = list(grouped.values())
        candidates.sort(
            key=lambda item: (
                item.verification_score,
                item.support_count,
                len(item.evidence_ids),
            ),
            reverse=True,
        )
        return candidates[:max_candidates]

    def _question_type(self, question: str) -> str:
        lowered = normalize_text(question).lower()
        if "who" in lowered:
            return "person"
        if "where" in lowered:
            return "place"
        if "when" in lowered or "date" in lowered or "year" in lowered:
            return "date"
        if "title" in lowered or "book" in lowered or "paper" in lowered or "video" in lowered:
            return "title"
        if "website" in lowered or "url" in lowered:
            return "website"
        return "entity"

    def _extract_answers(self, text: str, *, question_type: str) -> list[str]:
        candidates: list[str] = []
        if question_type == "website":
            candidates.extend(re.findall(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+", text))
        if question_type == "date":
            candidates.extend(
                re.findall(
                    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b",
                    text,
                )
            )
        if question_type in {"person", "place", "entity"}:
            candidates.extend(
                match.strip()
                for match in re.findall(
                    r"\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,4}\b",
                    text,
                )
            )
        if question_type == "title":
            candidates.extend(re.findall(r'"([^"]{4,120})"', text))
            candidates.extend(
                match.strip()
                for match in re.findall(
                    r"\b[A-Z][A-Za-z0-9'&:-]+(?:\s+[A-Z][A-Za-z0-9'&:-]+){1,8}\b",
                    text,
                )
            )
        return self._dedupe(candidates)

    def _valid_answer(self, answer: str) -> bool:
        text = normalize_text(answer).strip(" .,;:-")
        if len(text) < 2 or len(text) > 140:
            return False
        if text in self.STOP_CANDIDATES:
            return False
        lowered = text.lower()
        if lowered.startswith(("http://", "https://")):
            return True
        if lowered in {"search results", "official website", "home page"}:
            return False
        if all(token.lower() in self.STOPWORDS for token in text.split()):
            return False
        return True

    def _keywords(self, question: str) -> set[str]:
        lowered = re.sub(r"[^\w\s]", " ", normalize_text(question).lower())
        return {
            token
            for token in lowered.split()
            if len(token) > 2 and token not in self.STOPWORDS
        }

    def _answer_key(self, answer: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(answer).lower()).strip(" .,;:-")

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = normalize_text(value).strip(" .,;:-")
            key = self._answer_key(cleaned)
            if key and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result


__all__ = ["CandidateExtractor", "EvidenceExtractor"]
