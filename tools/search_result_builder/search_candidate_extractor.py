from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from utils.network_utils import normalize_text


class SearchCandidateExtractor:
    """SearchCandidateExtractor 類別，封裝此模組的資料結構與服務邏輯。"""

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
    }

    def extract_candidates(
        self,
        *,
        question: str,
        search_runs: list[dict[str, Any]],
        max_candidates: int = 3,
    ) -> list[dict[str, Any]]:
        """
        ????????????????
        
        Args:
            - ????????????
        
        Returns:
            - ???????
        """
        question_type = self._question_type(question)
        question_keywords = self._keywords(question)
        grouped: dict[str, dict[str, Any]] = {}

        for run in search_runs:
            query = str(run.get("query", "") or "")
            for item in self._iter_result_items(run.get("result") or {}):
                title = normalize_text(item.get("title", ""))
                url = normalize_text(item.get("url", ""))
                body = normalize_text(item.get("raw_content") or item.get("content") or "")
                text = normalize_text(f"{title}. {body}")
                if not text:
                    continue
                for answer in self._extract_answers(text, title=title, question_type=question_type):
                    if not self._valid_answer(answer):
                        continue
                    key = self._answer_key(answer)
                    entry = grouped.setdefault(
                        key,
                        {
                            "answer": answer,
                            "answer_type": question_type,
                            "support_count": 0,
                            "sources": [],
                            "score": 0,
                        },
                    )
                    keyword_hits = sum(1 for keyword in question_keywords if keyword in text.lower())
                    entry["support_count"] += 1
                    entry["score"] += 1 + min(keyword_hits, 4)
                    entry["sources"].append(
                        {
                            "query": query,
                            "title": title,
                            "url": url,
                            "supporting_text": body[:360],
                        }
                    )

        candidates = list(grouped.values())
        candidates.sort(key=lambda item: (item.get("score", 0), item.get("support_count", 0)), reverse=True)
        return candidates[:max_candidates]

    def verify_candidates(
        self,
        *,
        question: str,
        candidates: list[dict[str, Any]],
        verification_runs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        ????????????????????
        
        Args:
            - ????????????
        
        Returns:
            - ?????????
        """
        question_keywords = self._keywords(question)
        runs_by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in verification_runs:
            answer_key = self._answer_key(run.get("candidate_answer", ""))
            if answer_key:
                runs_by_answer[answer_key].append(run)

        verified: list[dict[str, Any]] = []
        for candidate in candidates:
            answer = str(candidate.get("answer", "") or "")
            key = self._answer_key(answer)
            support_sources: list[dict[str, Any]] = []
            verification_score = int(candidate.get("score", 0) or 0)
            for run in runs_by_answer.get(key, []):
                for item in self._iter_result_items(run.get("result") or {}):
                    title = normalize_text(item.get("title", ""))
                    url = normalize_text(item.get("url", ""))
                    body = normalize_text(item.get("raw_content") or item.get("content") or "")
                    haystack = normalize_text(f"{title}. {body}")
                    if key and key in self._answer_key(haystack):
                        keyword_hits = sum(1 for keyword in question_keywords if keyword in haystack.lower())
                        verification_score += 3 + min(keyword_hits, 4)
                        support_sources.append(
                            {
                                "query": run.get("query", ""),
                                "title": title,
                                "url": url,
                                "supporting_text": body[:360],
                            }
                        )

            if support_sources:
                item = dict(candidate)
                item["verified"] = True
                item["verification_score"] = verification_score
                item["verified_sources"] = support_sources[:3]
                verified.append(item)

        verified.sort(
            key=lambda item: (
                int(item.get("verification_score", 0) or 0),
                int(item.get("support_count", 0) or 0),
            ),
            reverse=True,
        )
        return verified

    def _iter_result_items(self, search_result: dict[str, Any]) -> list[dict[str, Any]]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        payload = search_result.get("raw_result")
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [item for item in payload.get("results", []) if isinstance(item, dict)]
        return []

    def _question_type(self, question: str) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
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

    def _extract_answers(self, text: str, *, title: str, question_type: str) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        candidates: list[str] = []
        if question_type == "website":
            candidates.extend(re.findall(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+", text))
        if question_type == "date":
            candidates.extend(re.findall(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b", text))
        if question_type in {"person", "place", "entity"}:
            candidates.extend(
                match.strip()
                for match in re.findall(
                    r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}\b",
                    text,
                )
            )
        if question_type == "title":
            title_candidate = re.split(r"[-|:]", title)[0].strip()
            if title_candidate:
                candidates.append(title_candidate)
            candidates.extend(re.findall(r'"([^"]{4,120})"', text))
        if not candidates and title:
            candidates.append(re.split(r"[-|:]", title)[0].strip())
        return self._dedupe(candidates)

    def _valid_answer(self, answer: str) -> bool:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
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
        return True

    def _keywords(self, question: str) -> set[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        lowered = re.sub(r"[^\w\s]", " ", normalize_text(question).lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "when",
            "where", "why", "how", "of", "in", "on", "at", "for", "to", "and", "or", "do",
            "does", "did", "can", "could", "should", "would", "answer", "question",
        }
        return {token for token in lowered.split() if len(token) > 2 and token not in stopwords}

    def _answer_key(self, answer: Any) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        return re.sub(r"\s+", " ", normalize_text(answer).lower()).strip(" .,;:-")

    def _dedupe(self, values: list[str]) -> list[str]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = normalize_text(value).strip(" .,;:-")
            key = self._answer_key(cleaned)
            if key and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result
