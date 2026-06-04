from __future__ import annotations

import re
from dataclasses import dataclass

from utils.network_utils import normalize_text

from ..config import CandidateAnswer, EvidenceItem, SearchQueryPlan, SearchSignals


@dataclass
class EvidenceDrivenQueryCandidate:
    query: str
    reason: str
    score: float = 0.0


class EvidenceDrivenQueryBuilder:
    """
    Build bounded follow-up search queries from first-hop evidence.

    Args:
        - None.

    Returns:
        - EvidenceDrivenQueryBuilder: Planner for evidence-driven follow-up queries.
    """

    GENERIC_TERMS = {
        "answer",
        "article",
        "book",
        "candidate",
        "character",
        "content",
        "data",
        "example",
        "find",
        "information",
        "issue",
        "label",
        "language",
        "main",
        "number",
        "paper",
        "question",
        "result",
        "search",
        "source",
        "string",
        "symbol",
        "title",
        "tool",
        "unknown",
        "version",
        "website",
    }
    STOP_ENTITY_HEADS = {
        "A",
        "An",
        "As",
        "At",
        "By",
        "For",
        "From",
        "How",
        "If",
        "In",
        "Of",
        "On",
        "Please",
        "The",
        "There",
        "This",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
    }
    QUERY_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }

    def build(
        self,
        *,
        question: str,
        search_signals: SearchSignals,
        initial_queries: list[SearchQueryPlan],
        evidence_items: list[EvidenceItem],
        candidates: list[CandidateAnswer],
        max_queries: int = 2,
    ) -> list[SearchQueryPlan]:
        """
        Build follow-up query plans that combine first-hop clues with the answer target.

        Args:
            - question: Original task question.
            - search_signals: Search signals produced by embedding salience.
            - initial_queries: Queries already executed in the first hop.
            - evidence_items: Evidence chunks extracted from first-hop sources.
            - candidates: First-hop answer candidates.
            - max_queries: Maximum follow-up queries to return.

        Returns:
            - list[SearchQueryPlan]: Search plans for the second evidence hop.
        """
        if max_queries <= 0:
            return []

        role_terms = self._answer_role_terms(question, search_signals)
        constraints = self._constraints(question, search_signals)
        source_hints = list(search_signals.source_hints)
        initial_keys = {self._query_key(plan.query) for plan in initial_queries}

        query_candidates: list[EvidenceDrivenQueryCandidate] = []
        for clue, clue_score in self._ranked_intermediate_clues(
            question=question,
            search_signals=search_signals,
            evidence_items=evidence_items,
            candidates=candidates,
        ):
            query = self._compose_query(
                clue=clue,
                role_terms=role_terms,
                constraints=constraints,
                source_hints=source_hints,
            )
            if not self._valid_query(query, initial_keys=initial_keys):
                continue
            query_candidates.append(
                EvidenceDrivenQueryCandidate(
                    query=query,
                    reason=f"first_hop_clue:{clue}",
                    score=clue_score + self._query_bonus(query, role_terms=role_terms),
                )
            )

        if not query_candidates and search_signals.target_terms:
            fallback = self._compose_query(
                clue=" ".join(search_signals.target_terms[:3]),
                role_terms=role_terms,
                constraints=constraints,
                source_hints=source_hints,
            )
            if self._valid_query(fallback, initial_keys=initial_keys):
                query_candidates.append(
                    EvidenceDrivenQueryCandidate(
                        query=fallback,
                        reason="target_terms_fallback",
                        score=0.1,
                    )
                )

        deduped = self._dedupe_candidates(query_candidates)
        deduped.sort(key=lambda item: item.score, reverse=True)
        plans: list[SearchQueryPlan] = []
        for index, item in enumerate(deduped[:max_queries], start=1):
            plans.append(
                SearchQueryPlan(
                    query_id=f"H{index}",
                    query=item.query,
                    purpose="evidence_driven_followup",
                    priority=80 - index,
                    source_hints=source_hints,
                    expected_answer_type=search_signals.answer_type,
                    requires_full_page=True,
                )
            )
        return plans

    def _ranked_intermediate_clues(
        self,
        *,
        question: str,
        search_signals: SearchSignals,
        evidence_items: list[EvidenceItem],
        candidates: list[CandidateAnswer],
    ) -> list[tuple[str, float]]:
        question_key = self._query_key(question)
        clues: dict[str, tuple[str, float]] = {}

        for index, clue in enumerate(self._extract_question_anchor_clues(question)):
            self._add_clue(
                clues,
                clue,
                score=5.0 - index * 0.05,
                question_key=question_key,
                allow_question_overlap=True,
            )

        for index, candidate in enumerate(candidates[:5]):
            self._add_clue(
                clues,
                candidate.answer,
                score=0.65 - index * 0.04,
                question_key=question_key,
                allow_question_overlap=False,
            )

        for index, term in enumerate(search_signals.target_terms[:8]):
            self._add_clue(
                clues,
                term,
                score=0.55 - index * 0.03,
                question_key=question_key,
                allow_question_overlap=True,
            )

        for index, evidence in enumerate(evidence_items[:8]):
            evidence_score = max(0.15, evidence.helpfulness_score, evidence.evidence_quality)
            for clue in self._extract_text_clues(evidence.text):
                self._add_clue(
                    clues,
                    clue,
                    score=evidence_score - index * 0.02,
                    question_key=question_key,
                    allow_question_overlap=False,
                )
            for clue in self._extract_text_clues(evidence.title):
                self._add_clue(
                    clues,
                    clue,
                    score=evidence_score * 0.35 - index * 0.02,
                    question_key=question_key,
                    allow_question_overlap=False,
                )

        values = list(clues.values())
        values.sort(key=lambda item: item[1], reverse=True)
        return values[:12]

    def _extract_question_anchor_clues(self, question: str) -> list[str]:
        text = normalize_text(question)
        clues: list[str] = []
        entity = r"[A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){0,3}"
        role = (
            r"prime minister|president|zip code|zip codes|ec number|ec numbers|"
            r"regression label|label|date|title|website|source"
        )
        for match in re.findall(
            rf"\b(?i:(?:{role}))\s+(?:of|for|in)\s+({entity})\b",
            text,
        ):
            self._append(clues, self._trim_entity(match))
        for match in re.findall(
            rf"\b(?:of|for|in|from)\s+({entity})\s+(?:in|on|during|before|after|between)\b",
            text,
        ):
            self._append(clues, self._trim_entity(match))
        return clues[:5]

    def _extract_text_clues(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        clues: list[str] = []

        for quoted in re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', normalized):
            value = next((part for part in quoted if part), "")
            self._append(clues, value)

        for match in re.findall(
            r"\bfrom\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})\s+to\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})\b",
            normalized,
        ):
            for value in match:
                self._append(clues, value)

        patterns = [
            r"\b[A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){0,5}\b",
            r"\b[A-Z][a-z]{2,}\s+[a-z][a-z-]{2,}\b",
            r"\b[A-Z]{2,}[A-Z0-9-]*\b",
            r"\b[a-z]+(?:\.[a-z0-9_]+){1,4}\b",
            r"\b[a-z][a-z-]{4,}\s+(?:virus|label|issue|repository|module|species)\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, normalized):
                self._append(clues, self._trim_entity(match))

        return clues[:20]

    def _answer_role_terms(self, question: str, search_signals: SearchSignals) -> list[str]:
        lowered = normalize_text(question).lower()
        roles: list[str] = []
        role_patterns = [
            ("prime minister", "Prime Minister"),
            ("president", "President"),
            ("capital", "capital"),
            ("zip code", "ZIP code"),
            ("zip codes", "ZIP code"),
            ("ec number", "EC number"),
            ("ec numbers", "EC number"),
            ("oldest closed issue", "oldest closed issue"),
            ("regression label", "Regression label"),
            ("label", "label"),
            ("date", "date"),
            ("when", "date"),
            ("distance", "distance"),
            ("main character", "main character"),
            ("symbol", "symbol"),
            ("word", "word"),
            ("title", "title"),
        ]
        for marker, value in role_patterns:
            if marker in lowered:
                self._append(roles, value)

        if not roles:
            fallback_by_type = {
                "code": "code",
                "date": "date",
                "number": "number",
                "person": "person",
                "place": "place",
                "symbol": "symbol",
                "title": "title",
                "website": "website",
                "word": "word",
            }
            fallback = fallback_by_type.get(search_signals.answer_type)
            if fallback:
                roles.append(fallback)
        return roles[:4]

    def _constraints(self, question: str, search_signals: SearchSignals) -> list[str]:
        constraints: list[str] = []
        for value in search_signals.constraints:
            self._append(constraints, value)
        for value in re.findall(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
            question,
            flags=re.IGNORECASE,
        ):
            self._append(constraints, value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", question):
            self._append(constraints, value)
        return constraints[:5]

    def _compose_query(
        self,
        *,
        clue: str,
        role_terms: list[str],
        constraints: list[str],
        source_hints: list[str],
    ) -> str:
        parts: list[str] = []
        if clue:
            parts.append(normalize_text(clue))
        parts.extend(normalize_text(term) for term in role_terms[:3])
        parts.extend(normalize_text(term) for term in constraints[:3])
        parts.extend(source_hints[:2])
        deduped_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            cleaned = normalize_text(part)
            key = self._query_key(cleaned)
            if not cleaned or not key or key in seen:
                continue
            deduped_parts.append(cleaned)
            seen.add(key)
        return normalize_text(" ".join(deduped_parts))[:300]

    def _valid_query(self, query: str, *, initial_keys: set[str]) -> bool:
        text = normalize_text(query)
        if not text:
            return False
        key = self._query_key(text)
        if key in initial_keys:
            return False
        tokens = re.findall(r"[A-Za-z0-9_.-]+", text)
        meaningful = [
            token
            for token in tokens
            if token.lower() not in self.QUERY_STOPWORDS and token.lower() not in self.GENERIC_TERMS
        ]
        if len(meaningful) < 2:
            return False
        if len(key) < 8 or key in self.GENERIC_TERMS:
            return False
        return True

    def _query_bonus(self, query: str, *, role_terms: list[str]) -> float:
        lowered = query.lower()
        bonus = 0.0
        for term in role_terms:
            if term.lower() in lowered:
                bonus += 0.08
        if re.search(r"\b(?:19|20)\d{2}\b", lowered):
            bonus += 0.06
        if '"' in query:
            bonus += 0.03
        return bonus

    def _add_clue(
        self,
        clues: dict[str, tuple[str, float]],
        value: str,
        *,
        score: float,
        question_key: str,
        allow_question_overlap: bool,
    ) -> None:
        cleaned = self._clean_clue(value)
        if not cleaned:
            return
        key = self._query_key(cleaned)
        if not allow_question_overlap and key and key == question_key:
            return
        if key in clues and clues[key][1] >= score:
            return
        clues[key] = (cleaned, score)

    def _clean_clue(self, value: str) -> str:
        cleaned = self._trim_entity(normalize_text(value)).strip(" .,;:-")
        if not cleaned or len(cleaned) > 80:
            return ""
        key = self._query_key(cleaned)
        if key in self.GENERIC_TERMS:
            return ""
        tokens = re.findall(r"[A-Za-z0-9_.-]+", cleaned)
        meaningful = [
            token
            for token in tokens
            if token.lower() not in self.QUERY_STOPWORDS and token.lower() not in self.GENERIC_TERMS
        ]
        if not meaningful:
            return ""
        if len(cleaned) <= 2 and not cleaned.isupper():
            return ""
        return cleaned

    def _dedupe_candidates(
        self,
        candidates: list[EvidenceDrivenQueryCandidate],
    ) -> list[EvidenceDrivenQueryCandidate]:
        deduped: dict[str, EvidenceDrivenQueryCandidate] = {}
        for candidate in candidates:
            key = self._query_key(candidate.query)
            existing = deduped.get(key)
            if existing is None or candidate.score > existing.score:
                deduped[key] = candidate
        return list(deduped.values())

    def _quote_if_phrase(self, value: str) -> str:
        cleaned = normalize_text(value).strip(" .,;:-")
        if not cleaned:
            return ""
        if " " in cleaned and not (cleaned.startswith('"') and cleaned.endswith('"')):
            return f'"{cleaned}"'
        return cleaned

    def _trim_entity(self, value: str) -> str:
        tokens = normalize_text(value).split()
        while tokens and tokens[0] in self.STOP_ENTITY_HEADS:
            tokens = tokens[1:]
        return " ".join(tokens).strip(" .,;:-")

    def _append(self, values: list[str], value: str) -> None:
        cleaned = normalize_text(value).strip(" .,;:-")
        if cleaned and self._query_key(cleaned) not in {self._query_key(item) for item in values}:
            values.append(cleaned)

    def _query_key(self, value: str) -> str:
        return re.sub(r"\s+", " ", normalize_text(value).lower()).strip(" \"'`.,;:-")


NextHopQueryGenerator = EvidenceDrivenQueryBuilder

__all__ = ["EvidenceDrivenQueryBuilder", "EvidenceDrivenQueryCandidate", "NextHopQueryGenerator"]
