from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text

from .search_candidate_extractor import SearchCandidateExtractor


class SearchEvidenceBuilder:
    """SearchEvidenceBuilder 類別，封裝此模組的資料結構與服務邏輯。"""
    def __init__(self, tool_manager=None, runtime=None, *, search_query_planner=None):
        """初始化物件與必要狀態。"""
        self.tool_manager = tool_manager
        self.runtime = runtime
        self.search_query_planner = search_query_planner
        self.candidate_extractor = SearchCandidateExtractor()

    def summarize_search_output(
        self,
        search_output_text: Any,
        question: str,
        max_sections: int = 3,
        max_chars_per_section: int = 300,
        max_total_chars: int = 900,
    ) -> str:
        """summarize_search_output 的主要實作。"""
        cleaned = self._clean_search_text(search_output_text)
        if not cleaned:
            return ""

        sections = self._split_search_sections(cleaned)
        ranked_sections = self._rank_search_sections(sections, question)
        selected = ranked_sections[:max_sections]

        summarized_parts = []
        for idx, section in enumerate(selected, start=1):
            clipped = section[:max_chars_per_section].strip()
            if len(section) > max_chars_per_section:
                clipped += " ..."
            summarized_parts.append(f"[{idx}] {clipped}")

        summary = "\n".join(summarized_parts).strip()
        if len(summary) > max_total_chars:
            summary = summary[:max_total_chars].rstrip() + " ..."

        return summary

    def summarize_structured_search_result(
        self,
        search_result: dict[str, Any],
        question: str,
        max_results: int = 2,
        max_chars_per_result: int = 240,
        max_total_chars: int = 900,
    ) -> str:
        """summarize_structured_search_result 的主要實作。"""
        payload = search_result.get("raw_result")
        if not isinstance(payload, dict):
            return ""

        results = payload.get("results") or []
        if not results:
            return ""

        query_keywords = self._extract_query_keywords(question)
        parts: list[str] = []
        for idx, item in enumerate(results[:max_results], start=1):
            title = self._clean_search_text(item.get("title", "")) or item.get("url", "")
            url = self._clean_search_text(item.get("url", ""))
            body = self._clean_search_text(item.get("raw_content") or item.get("content") or "")
            if not body:
                continue

            clipped = body[:max_chars_per_result].strip()
            if len(body) > max_chars_per_result:
                clipped += " ..."

            bonus = ""
            rerank_score = item.get("rerank_score")
            if rerank_score is not None:
                bonus = f" (score={rerank_score})"

            keyword_hits = sum(1 for kw in query_keywords if kw in body.lower() or kw in title.lower())
            if keyword_hits == 0 and idx > 1:
                continue

            section = f"[{idx}] {title}{bonus}\n{clipped}"
            if url:
                section += f"\nSource: {url}"
            parts.append(section)

        summary = "\n".join(parts).strip()
        if len(summary) > max_total_chars:
            summary = summary[:max_total_chars].rstrip() + " ..."
        return summary

    def build_search_evidence_block(
        self,
        search_result: dict[str, Any],
        question: str,
        max_sections: int = 3,
        max_chars_per_section: int = 300,
        max_total_chars: int = 900,
    ) -> str:
        """build_search_evidence_block 的主要實作。"""
        if not search_result or not search_result.get("ok"):
            return ""

        summary = self.summarize_structured_search_result(
            search_result=search_result,
            question=question,
            max_results=max_sections,
            max_chars_per_result=max_chars_per_section,
            max_total_chars=max_total_chars,
        ) or self.summarize_search_output(
            search_output_text=search_result.get("output_text", ""),
            question=question,
            max_sections=max_sections,
            max_chars_per_section=max_chars_per_section,
            max_total_chars=max_total_chars,
        )

        if not summary:
            return ""

        return (
            "Search evidence:\n"
            f"Question focus: {question}\n"
            f"Tool used: {search_result.get('tool_name', 'search')}\n"
            f"Key findings:\n{summary}\n"
        )

    def build_planned_search_evidence_block(
        self,
        *,
        search_runs: list[dict[str, Any]],
        question: str,
        max_queries: int = 3,
        max_chars_per_query: int = 320,
        max_total_chars: int = 1000,
    ) -> str:
        """build_planned_search_evidence_block 的主要實作。"""
        if not search_runs:
            return ""

        lines = [
            "Search evidence:",
            f"Question focus: {question}",
            "Search plan:",
        ]

        query_sections: list[str] = []
        for idx, run in enumerate(search_runs[:max_queries], start=1):
            query = str(run.get("query", "") or "").strip()
            search_result = run.get("result") or {}
            if not search_result.get("ok"):
                continue

            summary = self.summarize_structured_search_result(
                search_result=search_result,
                question=question,
                max_results=2,
                max_chars_per_result=180,
                max_total_chars=max_chars_per_query,
            ) or self.summarize_search_output(
                search_output_text=search_result.get("output_text", ""),
                question=question,
                max_sections=2,
                max_chars_per_section=180,
                max_total_chars=max_chars_per_query,
            )
            if not summary:
                continue

            lines.append(f"- Q{idx}: {query}")
            query_sections.append(f"[Q{idx}] {summary}")

        if not query_sections:
            return ""

        body = "\n".join(lines + ["Key findings:", "\n".join(query_sections)]).strip()
        if len(body) > max_total_chars:
            body = body[:max_total_chars].rstrip() + " ..."
        return body

    def build_verified_search_context(
        self,
        *,
        question: str,
        best_verified_candidate: dict[str, Any] | None,
    ) -> str:
        """build_verified_search_context 的主要實作。"""
        if not best_verified_candidate:
            return (
                "Search evidence:\n"
                "Verified candidate: None\n"
                "Instruction: do not invent a factual answer from search."
            )

        answer = self._clean_search_text(best_verified_candidate.get("answer", ""))
        answer_type = self._clean_search_text(best_verified_candidate.get("answer_type", "entity"))
        score = best_verified_candidate.get("verification_score", 0)
        lines = [
            "Search evidence:",
            "Verified candidate:",
            f"- answer: {answer}",
            f"- answer_type: {answer_type}",
            f"- verification_score: {score}",
            "Instruction: use answer exactly if verification_score is high, unless stronger tool evidence contradicts it.",
        ]
        return "\n".join(lines).strip()

    def _clean_search_text(self, text: Any) -> str:
        """_clean_search_text 的內部輔助實作。"""
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\\n", "\n")
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _split_search_sections(self, text: str) -> list[str]:
        """_split_search_sections 的內部輔助實作。"""
        if not text:
            return []

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if blocks:
            return blocks

        return [line.strip() for line in text.splitlines() if line.strip()]

    def _extract_query_keywords(self, question: str) -> set[str]:
        """_extract_query_keywords 的內部輔助實作。"""
        text = normalize_text(question).lower()
        text = re.sub(r"[^\w\s]", " ", text)

        stopwords = {
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
        }

        tokens = [token for token in text.split() if len(token) > 1 and token not in stopwords]
        return set(tokens)

    def _looks_like_metadata_section(self, section: str) -> bool:
        """_looks_like_metadata_section 的內部輔助實作。"""
        lower = section.lower().strip()

        metadata_markers = [
            "source:",
            "url:",
            "http://",
            "https://",
        ]

        if any(marker in lower for marker in metadata_markers):
            return True

        if len(lower) < 20:
            return True

        return False

    def _score_search_section(self, section: str, question_keywords: set[str]) -> int:
        """_score_search_section 的內部輔助實作。"""
        lower = normalize_text(section).lower()
        score = 0

        matched_keywords = [kw for kw in question_keywords if kw in lower]
        score += len(set(matched_keywords)) * 2

        if len(set(matched_keywords)) >= 2:
            score += 3

        if self._looks_like_metadata_section(section):
            score -= 2

        answer_markers = ["is", "means", "refers to", "capital", "founded", "released"]
        if any(marker in lower for marker in answer_markers):
            score += 1

        return score

    def _rank_search_sections(self, sections: list[str], question: str) -> list[str]:
        """_rank_search_sections 的內部輔助實作。"""
        question_keywords = self._extract_query_keywords(question)

        scored_sections = []
        for idx, section in enumerate(sections):
            score = self._score_search_section(section, question_keywords)
            scored_sections.append((score, idx, section))

        scored_sections.sort(key=lambda entry: (-entry[0], entry[1]))
        return [section for _, _, section in scored_sections]
