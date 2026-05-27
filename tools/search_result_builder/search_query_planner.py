from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text


class SearchQueryPlanner:
    """SearchQueryPlanner 類別，封裝此模組的資料結構與服務邏輯。"""
    SEARCH_STOPWORDS = {
        "a",
        "an",
        "and",
        "answer",
        "as",
        "at",
        "between",
        "by",
        "can",
        "could",
        "distance",
        "do",
        "find",
        "for",
        "from",
        "give",
        "how",
        "i",
        "if",
        "in",
        "included",
        "is",
        "it",
        "latest",
        "many",
        "me",
        "must",
        "my",
        "nearest",
        "not",
        "of",
        "on",
        "or",
        "page",
        "please",
        "provide",
        "result",
        "round",
        "should",
        "take",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "use",
        "using",
        "version",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }

    SOURCE_MARKERS = {
        "wikipedia": "wikipedia",
        "official": "official",
        "latest": "latest",
        "english wikipedia": "english_wikipedia",
        "2022 version": "versioned_source",
    }

    INSTRUCTION_MARKERS = [
        "please use",
        "please provide your answer",
        "provide your answer",
        "round your",
        "do not use",
        "if necessary",
        "answer as",
        "you can use",
        "use the latest",
    ]

    LEADING_ENTITY_STOPWORDS = {
        "A",
        "An",
        "How",
        "If",
        "Please",
        "The",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
        "You",
    }

    def plan(self, question: str, max_queries: int = 5) -> dict[str, Any]:
        """plan 的主要實作。"""
        text = normalize_text(question)
        if not text:
            return {
                "queries": [],
                "core_query": "",
                "keyword_query": "",
                "source_query": "",
                "source_hints": [],
                "year_tokens": [],
                "precision_needed": False,
            }

        source_hints = self._detect_source_hints(text)
        year_tokens = self._extract_year_tokens(text)
        core_query = self._build_core_query(text)
        keyword_query = self._build_keyword_query(
            text,
            core_query=core_query,
            year_tokens=year_tokens,
        )
        source_query = self._build_source_query(
            text,
            core_query=core_query,
            keyword_query=keyword_query,
            source_hints=source_hints,
            year_tokens=year_tokens,
        )

        quoted_queries = self._build_quoted_entity_queries(
            text,
            core_query=core_query,
            year_tokens=year_tokens,
        )
        ordered_candidates = [core_query, *quoted_queries, keyword_query, source_query]
        queries: list[str] = []
        seen: set[str] = set()
        for candidate in ordered_candidates:
            normalized = self._normalize_query_key(candidate)
            if not normalized or normalized in seen:
                continue
            queries.append(candidate.strip())
            seen.add(normalized)
            if len(queries) >= max(1, max_queries):
                break

        precision_needed = bool(source_hints or year_tokens or re.search(r"\d", text))

        return {
            "queries": queries,
            "core_query": core_query,
            "keyword_query": keyword_query,
            "source_query": source_query,
            "quoted_queries": quoted_queries,
            "source_hints": source_hints,
            "year_tokens": year_tokens,
            "precision_needed": precision_needed,
        }

    def build_verification_query(self, *, question: str, candidate_answer: str) -> str:
        """build_verification_query 的主要實作。"""
        text = normalize_text(question)
        candidate = normalize_text(candidate_answer)
        if not text or not candidate:
            return text or candidate
        core_query = self._build_core_query(text)
        entities = self._extract_entity_phrases(text)
        years = self._extract_year_tokens(text)
        focus = " ".join([f'"{item}"' for item in entities[:2]])
        year_text = " ".join(years)
        return normalize_text(f'"{candidate}" {focus} {year_text} {core_query}')[:300]

    def _build_quoted_entity_queries(
        self,
        question: str,
        *,
        core_query: str,
        year_tokens: list[str],
    ) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        queries: list[str] = []
        entities = self._extract_entity_phrases(question)
        answer_terms = self._answer_target_terms(question)
        year_text = " ".join(year_tokens)
        for entity in entities[:3]:
            quoted = f'"{entity}"'
            suffix = " ".join(answer_terms[:2])
            query = normalize_text(f"{quoted} {suffix} {year_text}")
            if query and query not in queries:
                queries.append(query)
        if not queries and core_query:
            focus_tokens = self._extract_focus_tokens(core_query)
            if focus_tokens:
                query = normalize_text(f'"{" ".join(focus_tokens[:4])}" {" ".join(answer_terms[:2])} {year_text}')
                if query:
                    queries.append(query)
        return queries

    def _answer_target_terms(self, question: str) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        lowered = question.lower()
        terms: list[str] = []
        target_map = [
            ("who", ["author", "person", "name"]),
            ("where", ["location", "place"]),
            ("title", ["title"]),
            ("book", ["book", "title"]),
            ("paper", ["paper", "author", "title"]),
            ("website", ["website", "official"]),
            ("video", ["video", "title"]),
            ("company", ["company"]),
            ("institution", ["institution", "organization"]),
            ("date", ["date", "year"]),
            ("when", ["date", "year"]),
        ]
        for marker, values in target_map:
            if marker in lowered:
                terms.extend(value for value in values if value not in terms)
        return terms[:4]

    def _build_core_query(self, question: str) -> str:
        """_build_core_query 的內部輔助實作。"""
        clauses = self._split_into_clauses(question)
        kept_clauses = []
        for clause in clauses:
            lowered = clause.lower()
            trimmed = clause.strip()
            for marker in self.INSTRUCTION_MARKERS:
                marker_index = lowered.find(marker)
                if marker_index != -1:
                    trimmed = trimmed[:marker_index].rstrip(" ,;:-")
                    break
            if trimmed:
                kept_clauses.append(trimmed)

        if not kept_clauses:
            return question.strip()

        core = " ".join(kept_clauses).strip()
        core = re.sub(r"\s+", " ", core)
        return core

    def _build_keyword_query(self, question: str, *, core_query: str, year_tokens: list[str]) -> str:
        """_build_keyword_query 的內部輔助實作。"""
        entities = self._extract_entity_phrases(question)
        tokens = self._extract_focus_tokens(core_query)

        parts: list[str] = []
        part_tokens: set[str] = set()
        for entity in entities[:2]:
            parts.append(entity)
            part_tokens.update(entity.lower().split())

        for token in year_tokens:
            if token not in parts:
                parts.append(token)
                part_tokens.update(token.lower().split())

        for token in tokens:
            lowered = token.lower()
            if lowered not in part_tokens:
                parts.append(token)
                part_tokens.add(lowered)
            if len(parts) >= 8:
                break

        if parts:
            return " ".join(parts)

        return core_query

    def _build_source_query(
        self,
        question: str,
        *,
        core_query: str,
        keyword_query: str,
        source_hints: list[str],
        year_tokens: list[str],
    ) -> str:
        """_build_source_query 的內部輔助實作。"""
        base = keyword_query or core_query
        if not base:
            return ""

        lowered = question.lower()
        if "english wikipedia" in lowered or "wikipedia" in lowered:
            extra_years = [token for token in year_tokens if token not in base]
            query = f"{base} site:en.wikipedia.org"
            if extra_years:
                query = f"{query} {' '.join(extra_years)}"
            return query

        if "official" in lowered:
            return f"{base} official"

        if source_hints:
            return f"{base} {' '.join(source_hints)}".strip()

        return ""

    def _split_into_clauses(self, text: str) -> list[str]:
        """_split_into_clauses 的內部輔助實作。"""
        raw = re.split(r"[;；。.!?\n]+", text)
        clauses = [segment.strip() for segment in raw if segment.strip()]
        return clauses or [text.strip()]

    def _detect_source_hints(self, text: str) -> list[str]:
        """_detect_source_hints 的內部輔助實作。"""
        lowered = text.lower()
        hints = []
        for marker, hint in self.SOURCE_MARKERS.items():
            if marker in lowered and hint not in hints:
                hints.append(hint)
        return hints

    def _extract_year_tokens(self, text: str) -> list[str]:
        """_extract_year_tokens 的內部輔助實作。"""
        tokens: list[str] = []
        consumed_years: set[str] = set()

        between_match = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", text, flags=re.IGNORECASE)
        if between_match:
            tokens.append(f"{between_match.group(1)} {between_match.group(2)}")
            consumed_years.update([between_match.group(1), between_match.group(2)])

        from_match = re.search(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", text, flags=re.IGNORECASE)
        if from_match:
            range_token = f"{from_match.group(1)} {from_match.group(2)}"
            if range_token not in tokens:
                tokens.append(range_token)
            consumed_years.update([from_match.group(1), from_match.group(2)])

        for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
            if year in consumed_years:
                continue
            if year not in tokens:
                tokens.append(year)

        return tokens[:3]

    def _extract_entity_phrases(self, text: str) -> list[str]:
        """_extract_entity_phrases 的內部輔助實作。"""
        phrases: list[str] = []

        for quote in re.findall(r'"([^"]+)"|\'([^\']+)\'', text):
            candidate = next((part for part in quote if part), "").strip()
            if candidate and candidate not in phrases:
                phrases.append(candidate)

        capitalized_pattern = re.compile(
            r"\b[A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,4}\b"
        )
        for match in capitalized_pattern.findall(text):
            candidate = self._trim_leading_entity_stopwords(match.strip())
            if not candidate:
                continue
            if candidate.lower() in self.SEARCH_STOPWORDS:
                continue
            if candidate not in phrases:
                phrases.append(candidate)

        return phrases[:4]

    def _extract_focus_tokens(self, text: str) -> list[str]:
        """_extract_focus_tokens 的內部輔助實作。"""
        lowered = normalize_text(text).lower()
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        tokens = []
        for token in lowered.split():
            if token in self.SEARCH_STOPWORDS:
                continue
            if len(token) <= 2:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:8]

    def _normalize_query_key(self, query: str) -> str:
        """_normalize_query_key 的內部輔助實作。"""
        return re.sub(r"\s+", " ", normalize_text(query).lower()).strip()

    def _trim_leading_entity_stopwords(self, candidate: str) -> str:
        """_trim_leading_entity_stopwords 的內部輔助實作。"""
        tokens = candidate.split()
        while tokens and tokens[0] in self.LEADING_ENTITY_STOPWORDS:
            tokens = tokens[1:]
        return " ".join(tokens).strip()
