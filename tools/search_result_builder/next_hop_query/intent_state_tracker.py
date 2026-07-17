from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text

from ..query.search_intent_plan import SearchIntentPlan


class SearchIntentStateTracker:
    """
    Update SearchIntentPlan as retrieval runtime state.

    Args:
        - None.

    Returns:
        - SearchIntentStateTracker: Stateless transition helper for retrieval rounds.
    """

    MULTI_HOP_MARKERS = (
        "prior",
        "previous",
        "earlier",
        "earliest",
        "first paper",
        "first article",
        "authored before",
        "had authored",
        "the one that",
        "the author who",
        "that author's",
        "whose",
    )
    DEFINITION_TERMS = (
        "definition",
        "classification",
        "classified",
        "botanical",
        "criteria",
        "rule",
        "distinguish",
        "distinguishing",
    )
    MEDIA_TERMS = ("transcript", "caption", "youtube", "video", "quote")

    def update(
        self,
        *,
        plan: SearchIntentPlan,
        question: str,
        documents: list[Any],
    ) -> SearchIntentPlan:
        """
        Update intent state from the current retrieval documents.

        Args:
            - plan: Current SearchIntentPlan.
            - question: Original task question.
            - documents: Retrieved document traces for the current round.

        Returns:
            - SearchIntentPlan: New plan with updated state and term coverage.
        """
        combined = self._combined_text(documents)
        completed = [
            term for term in plan.must_include if self._covers(combined, term)
        ]
        missing = [
            term for term in plan.must_include if term not in completed
        ]
        preferred_domain_seen = self._preferred_domain_seen(
            plan.preferred_domain,
            documents,
        )
        answer_type = self._answer_type(question)
        answer_seen = self._answer_candidate_seen(answer_type, combined)
        preferred_domain_text = self._combined_text(
            self._documents_for_domain(plan.preferred_domain, documents)
        )
        preferred_answer_seen = (
            self._answer_candidate_seen(answer_type, preferred_domain_text)
            if preferred_domain_text
            else False
        )
        all_required_seen = not missing if plan.must_include else bool(combined)
        multi_hop = self._multi_hop_expected(question)

        state = self._transition(
            plan=plan,
            question=question,
            combined_text=combined,
            all_required_seen=all_required_seen,
            preferred_domain_seen=preferred_domain_seen,
            answer_type=answer_type,
            answer_seen=answer_seen,
            preferred_answer_seen=preferred_answer_seen,
            multi_hop=multi_hop,
        )
        missing_terms = list(missing)
        if state != "sufficient" and answer_type != "unknown" and not answer_seen:
            missing_terms.append(f"answer_candidate:{answer_type}")
        if plan.preferred_domain and not preferred_domain_seen:
            missing_terms.append(f"preferred_domain:{plan.preferred_domain}")
        completed_terms = list(completed)
        if answer_seen:
            completed_terms.append(f"answer_candidate:{answer_type}")
        if preferred_domain_seen:
            completed_terms.append(f"preferred_domain:{plan.preferred_domain}")

        return plan.replace(
            state=state,
            completed_terms=self._dedupe(completed_terms),
            missing_terms=self._dedupe(missing_terms),
        )

    def _transition(
        self,
        *,
        plan: SearchIntentPlan,
        question: str,
        combined_text: str,
        all_required_seen: bool,
        preferred_domain_seen: bool,
        answer_type: str,
        answer_seen: bool,
        preferred_answer_seen: bool,
        multi_hop: bool,
    ) -> str:
        if not combined_text:
            return "pending"
        intent = normalize_text(plan.intent).lower()
        if intent == "official_page":
            if plan.preferred_domain:
                if preferred_domain_seen and preferred_answer_seen:
                    return "sufficient"
                if preferred_domain_seen:
                    return "first_hop_satisfied"
                return "pending"
            if all_required_seen and answer_seen:
                return "sufficient"
            return "first_hop_satisfied" if all_required_seen else "pending"

        if intent == "paper":
            if plan.state == "needs_next_hop":
                if self._later_hop_evidence_seen(combined_text) and answer_seen:
                    return "sufficient"
                return "needs_next_hop"
            if not all_required_seen:
                return "pending"
            if multi_hop:
                return "needs_next_hop"
            if answer_seen:
                return "sufficient"
            return "first_hop_satisfied"

        if intent == "definition":
            if self._definition_rule_seen(question, combined_text, plan):
                return "sufficient"
            return "pending"

        if intent == "media":
            media_seen = any(term in combined_text.casefold() for term in self.MEDIA_TERMS)
            if media_seen and answer_seen:
                return "sufficient"
            if media_seen:
                return "first_hop_satisfied"
            return "pending"

        if all_required_seen and answer_seen:
            return "sufficient"
        if all_required_seen:
            return "first_hop_satisfied"
        return "pending"

    def _combined_text(self, documents: list[Any]) -> str:
        parts: list[str] = []
        for document in documents:
            if isinstance(document, dict):
                title = str(document.get("title", "") or "")
                text = str(document.get("text", "") or "")
                url = str(document.get("url", "") or "")
                useful_tokens = document.get("useful_tokens") or []
                sequence_tag = str(document.get("sequence_tag", "") or "")
            else:
                title = str(getattr(document, "title", "") or "")
                text = str(getattr(document, "text", "") or "")
                url = str(getattr(document, "url", "") or "")
                useful_tokens = getattr(document, "useful_tokens", []) or []
                sequence_tag = str(getattr(document, "sequence_tag", "") or "")
            parts.extend([title, text[:1600], url, sequence_tag])
            parts.extend(str(token) for token in useful_tokens)
        return normalize_text(" ".join(part for part in parts if part))

    def _preferred_domain_seen(self, domain: str, documents: list[Any]) -> bool:
        if not domain:
            return False
        domain_key = domain.lower().removeprefix("www.")
        for document in documents:
            url = (
                str(document.get("url", "") or "")
                if isinstance(document, dict)
                else str(getattr(document, "url", "") or "")
            ).lower()
            if domain_key and domain_key in url.removeprefix("www."):
                return True
        return False

    def _documents_for_domain(self, domain: str, documents: list[Any]) -> list[Any]:
        if not domain:
            return []
        domain_key = domain.lower().removeprefix("www.")
        result: list[Any] = []
        for document in documents:
            url = (
                str(document.get("url", "") or "")
                if isinstance(document, dict)
                else str(getattr(document, "url", "") or "")
            ).lower()
            if domain_key and domain_key in url.removeprefix("www."):
                result.append(document)
        return result

    def _covers(self, text: str, term: str) -> bool:
        text_key = self._key(text)
        term_key = self._key(term)
        if not text_key or not term_key:
            return False
        if f" {term_key} " in f" {text_key} ":
            return True
        term_tokens = term_key.split()
        if len(term_tokens) <= 1:
            return False
        text_tokens = set(text_key.split())
        informative = [token for token in term_tokens if len(token) >= 3]
        return bool(informative) and all(token in text_tokens for token in informative)

    def _answer_type(self, question: str) -> str:
        lowered = normalize_text(question).casefold()
        if "zip code" in lowered or "zipcode" in lowered or "five-digit" in lowered:
            return "zip_code"
        if re.search(r"\blist\b|\bseparated by commas\b|\bcomma-separated\b", lowered):
            return "list"
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", lowered):
            return "number"
        if re.search(r"\bwhen\b|\bwhat date\b|\bwhich year\b|\bwhat year\b", lowered):
            return "date"
        if re.search(r"\bwhere\b|\bwhich country\b|\bwhich city\b|\bwhich place\b", lowered):
            return "location"
        if re.search(r"\bwho\b|\bwhose\b|\bwriter\b|\bauthor\b", lowered):
            return "person"
        if re.search(r"\btitle\b|\bname of\b|\bcalled\b", lowered):
            return "title"
        return "short_phrase"

    def _answer_candidate_seen(self, answer_type: str, text: str) -> bool:
        if answer_type == "zip_code":
            return bool(re.search(r"\b\d{5}\b", text))
        if answer_type == "number":
            return bool(re.search(r"[-+]?\b\d+(?:\.\d+)?%?\b", text))
        if answer_type == "date":
            return bool(
                re.search(r"\b(?:18|19|20)\d{2}\b", text)
                or re.search(
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}",
                    text,
                )
                or re.search(r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b", text)
            )
        if answer_type == "location":
            return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text))
        if answer_type == "person":
            return self._person_candidate_seen(text)
        if answer_type == "title":
            return bool(
                re.search(
                    r"\b[A-Z][A-Za-z0-9'&:,-]+(?:\s+[A-Z0-9][A-Za-z0-9'&:,-]+){1,10}\b",
                    text,
                )
            )
        if answer_type == "list":
            return "," in text or ";" in text
        return len(normalize_text(text)) >= 80

    def _person_candidate_seen(self, text: str) -> bool:
        organization_markers = {
            "agency",
            "association",
            "company",
            "corporation",
            "department",
            "foundation",
            "group",
            "inc",
            "institute",
            "journal",
            "llc",
            "org",
            "organization",
            "university",
            "word",
            "day",
            "jingoism",
            "merriam",
            "webster",
        }
        for candidate in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
            words = {word.casefold().strip(".,") for word in candidate.split()}
            if words & organization_markers:
                continue
            return True
        return False

    def _multi_hop_expected(self, question: str) -> bool:
        lowered = normalize_text(question).casefold()
        return any(marker in lowered for marker in self.MULTI_HOP_MARKERS)

    def _later_hop_evidence_seen(self, text: str) -> bool:
        lowered = normalize_text(text).casefold()
        return any(
            marker in lowered
            for marker in (
                "prior",
                "previous",
                "earlier",
                "earliest",
                "first paper",
                "first article",
                "publications",
                "publication",
                "authored",
                "cited",
            )
        )

    def _definition_rule_seen(
        self,
        question: str,
        text: str,
        plan: SearchIntentPlan,
    ) -> bool:
        del question
        lowered = text.casefold()
        rule_terms = [term for term in self.DEFINITION_TERMS if term in lowered]
        plan_terms = [
            term
            for term in plan.must_include
            if self._covers(text, term)
        ]
        return len(rule_terms) >= 1 and len(plan_terms) >= max(1, min(2, len(plan.must_include)))

    def _key(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold()).strip()

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            result.append(cleaned)
            seen.add(key)
        return result


__all__ = ["SearchIntentStateTracker"]
