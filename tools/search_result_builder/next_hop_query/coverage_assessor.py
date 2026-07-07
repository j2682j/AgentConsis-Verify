from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable

from utils.network_utils import normalize_text


@dataclass
class CoverageAssessment:
    sufficient: bool
    coverage_score: float
    missing_constraints: list[str] = field(default_factory=list)
    covered_constraints: list[str] = field(default_factory=list)
    answer_type: str = "unknown"
    answer_type_covered: bool = False
    bridge_terms: list[str] = field(default_factory=list)
    trigger_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoverageAssessor:
    """
    Judge whether retrieved documents cover the original question well enough to stop.

    The assessor is intentionally lightweight: it checks generic constraints,
    answer-type evidence, and bridge terms without calling another LLM.
    """

    STOPWORDS = {
        "about",
        "according",
        "after",
        "also",
        "answer",
        "before",
        "between",
        "from",
        "give",
        "have",
        "into",
        "need",
        "only",
        "question",
        "that",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
    }
    SOURCE_HINT_TERMS = {
        "arxiv",
        "imdb",
        "usgs",
        "official",
        "github",
        "wikipedia",
        "paper",
        "article",
        "report",
        "database",
        "journal",
        "newspaper",
    }

    def __init__(
        self,
        *,
        sufficiency_threshold: float = 0.72,
        min_constraint_coverage: float = 0.70,
        min_evidence_chars: int = 80,
        max_bridge_terms: int = 8,
    ) -> None:
        self.sufficiency_threshold = max(0.0, min(1.0, sufficiency_threshold))
        self.min_constraint_coverage = max(0.0, min(1.0, min_constraint_coverage))
        self.min_evidence_chars = max(1, min_evidence_chars)
        self.max_bridge_terms = max(1, max_bridge_terms)

    def assess(
        self,
        *,
        question: str,
        documents: Iterable[Any],
    ) -> CoverageAssessment:
        question_text = normalize_text(question)
        document_list = list(documents or [])
        evidence_texts = [
            self._document_text(document)
            for document in document_list
            if self._document_text(document)
        ]
        combined_evidence = normalize_text(" ".join(evidence_texts))
        if not question_text:
            return CoverageAssessment(
                sufficient=False,
                coverage_score=0.0,
                trigger_reason="empty_question",
            )
        if not combined_evidence:
            return CoverageAssessment(
                sufficient=False,
                coverage_score=0.0,
                missing_constraints=self._constraints(question_text),
                answer_type=self._answer_type(question_text),
                trigger_reason="no_retrieved_evidence",
            )

        constraints = self._constraints(question_text)
        covered_constraints = [
            constraint
            for constraint in constraints
            if self._constraint_covered(constraint, combined_evidence)
        ]
        missing_constraints = [
            constraint for constraint in constraints if constraint not in covered_constraints
        ]
        constraint_score = (
            len(covered_constraints) / len(constraints)
            if constraints
            else 1.0
        )

        answer_type = self._answer_type(question_text)
        answer_type_covered = self._answer_type_covered(answer_type, combined_evidence)
        answer_score = 1.0 if answer_type_covered or answer_type == "unknown" else 0.0
        availability_score = min(1.0, len(combined_evidence) / max(1, self.min_evidence_chars * 2))
        useful_token_count = sum(len(getattr(document, "useful_tokens", []) or []) for document in document_list)
        useful_score = min(1.0, useful_token_count / 4.0)
        bridge_terms = self._bridge_terms(question_text, document_list)
        bridge_score = min(1.0, len(bridge_terms) / 3.0)

        coverage_score = round(
            0.42 * constraint_score
            + 0.30 * answer_score
            + 0.16 * availability_score
            + 0.07 * useful_score
            + 0.05 * bridge_score,
            6,
        )
        sufficient = (
            coverage_score >= self.sufficiency_threshold
            and constraint_score >= self.min_constraint_coverage
            and (answer_type == "unknown" or answer_type_covered)
        )
        trigger_reason = "coverage_sufficient" if sufficient else self._trigger_reason(
            missing_constraints=missing_constraints,
            answer_type=answer_type,
            answer_type_covered=answer_type_covered,
            evidence_text=combined_evidence,
        )
        return CoverageAssessment(
            sufficient=sufficient,
            coverage_score=coverage_score,
            missing_constraints=missing_constraints,
            covered_constraints=covered_constraints,
            answer_type=answer_type,
            answer_type_covered=answer_type_covered,
            bridge_terms=bridge_terms,
            trigger_reason=trigger_reason,
            details={
                "constraint_score": round(constraint_score, 6),
                "answer_score": round(answer_score, 6),
                "availability_score": round(availability_score, 6),
                "useful_score": round(useful_score, 6),
                "bridge_score": round(bridge_score, 6),
                "useful_token_count": useful_token_count,
                "evidence_char_count": len(combined_evidence),
            },
        )

    def _constraints(self, question: str) -> list[str]:
        constraints: list[str] = []
        lowered = question.casefold()
        for year in re.findall(r"\b(?:18|19|20)\d{2}\b", question):
            constraints.append(f"year:{year}")
        for match in re.finditer(r"\b(before|after|since|until)\s+((?:18|19|20)\d{2})\b", lowered):
            constraints.append(f"time_constraint:{match.group(1)} {match.group(2)}")
        for phrase in re.findall(r'"([^"]{3,80})"|' + r"'([^']{3,80})'", question):
            value = normalize_text(phrase[0] or phrase[1])
            if value:
                constraints.append(f"phrase:{value}")
        for term in sorted(self.SOURCE_HINT_TERMS):
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                constraints.append(f"source:{term}")
        if "zip code" in lowered or "zipcode" in lowered:
            constraints.append("answer_hint:zip_code")
        if "five-digit" in lowered or "5 digit" in lowered:
            constraints.append("answer_hint:five_digit")
        return self._dedupe(constraints)

    def _constraint_covered(self, constraint: str, evidence_text: str) -> bool:
        lowered = evidence_text.casefold()
        if constraint.startswith("year:"):
            return constraint.split(":", 1)[1] in lowered
        if constraint.startswith("time_constraint:"):
            _, value = constraint.split(":", 1)
            return any(part in lowered for part in value.split())
        if constraint.startswith("phrase:"):
            return self._contains_phrase(lowered, constraint.split(":", 1)[1])
        if constraint.startswith("source:"):
            return re.search(rf"\b{re.escape(constraint.split(':', 1)[1])}\b", lowered) is not None
        if constraint == "answer_hint:zip_code":
            return bool(re.search(r"\b\d{5}\b", evidence_text))
        if constraint == "answer_hint:five_digit":
            return bool(re.search(r"\b\d{5}\b", evidence_text))
        return self._contains_phrase(lowered, constraint)

    def _answer_type(self, question: str) -> str:
        lowered = question.casefold()
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
        if re.search(r"\bwho\b|\bwhose\b", lowered):
            return "person"
        if re.search(r"\btitle\b|\bname of\b|\bcalled\b", lowered):
            return "title"
        return "short_phrase"

    def _answer_type_covered(self, answer_type: str, evidence_text: str) -> bool:
        if answer_type == "zip_code":
            return bool(re.search(r"\b\d{5}\b", evidence_text))
        if answer_type == "number":
            return bool(re.search(r"[-+]?\b\d+(?:\.\d+)?%?\b", evidence_text))
        if answer_type == "date":
            return bool(
                re.search(r"\b(?:18|19|20)\d{2}\b", evidence_text)
                or re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}", evidence_text)
                or re.search(r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b", evidence_text)
            )
        if answer_type == "location":
            return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", evidence_text))
        if answer_type == "person":
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
                "llc",
                "org",
                "organization",
                "university",
            }
            candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", evidence_text)
            for candidate in candidates:
                words = {word.casefold().strip(".,") for word in candidate.split()}
                if words & organization_markers:
                    continue
                return True
            return False
        if answer_type == "title":
            return bool(re.search(r"\b[A-Z][A-Za-z0-9'&:-]+(?:\s+[A-Z0-9][A-Za-z0-9'&:-]+){1,8}\b", evidence_text))
        if answer_type == "list":
            return "," in evidence_text or ";" in evidence_text
        return len(normalize_text(evidence_text)) >= self.min_evidence_chars

    def _bridge_terms(self, question: str, documents: list[Any]) -> list[str]:
        question_terms = {term.casefold() for term in self._tokens(question)}
        scored: dict[str, float] = {}
        for document in documents:
            base_score = float(getattr(document, "retrieval_score", 0.0) or 0.0)
            text = " ".join(
                [
                    str(getattr(document, "title", "") or ""),
                    str(getattr(document, "text", "") or "")[:1200],
                    " ".join(str(token) for token in getattr(document, "useful_tokens", []) or []),
                ]
            )
            for token in self._tokens(text):
                key = token.casefold()
                if key in question_terms or key in self.STOPWORDS:
                    continue
                score = base_score
                if any(char.isdigit() for char in token):
                    score += 0.25
                if token[:1].isupper() and len(token) > 3:
                    score += 0.18
                if len(token) >= 8:
                    score += 0.10
                scored[token] = max(scored.get(token, 0.0), score)
        return [
            term
            for term, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0].casefold()))
        ][: self.max_bridge_terms]

    def _tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]{2,}", normalize_text(text)):
            cleaned = token.strip("'_.-")
            if len(cleaned) < 3 or cleaned.casefold() in self.STOPWORDS:
                continue
            tokens.append(cleaned)
        return tokens

    def _document_text(self, document: Any) -> str:
        if isinstance(document, dict):
            title = str(document.get("title", "") or "")
            text = str(document.get("text", "") or "")
        else:
            title = str(getattr(document, "title", "") or "")
            text = str(getattr(document, "text", "") or "")
        return normalize_text(" ".join(part for part in (title, text) if part))

    def _trigger_reason(
        self,
        *,
        missing_constraints: list[str],
        answer_type: str,
        answer_type_covered: bool,
        evidence_text: str,
    ) -> str:
        reasons: list[str] = []
        if len(evidence_text) < self.min_evidence_chars:
            reasons.append("evidence_too_short")
        if missing_constraints:
            reasons.append("missing_constraints")
        if answer_type != "unknown" and not answer_type_covered:
            reasons.append("answer_type_not_covered")
        return "+".join(reasons) or "coverage_insufficient"

    def _contains_phrase(self, lowered_text: str, phrase: str) -> bool:
        key = re.sub(r"[^a-z0-9]+", " ", normalize_text(phrase).casefold()).strip()
        if not key:
            return False
        normalized_text = re.sub(r"[^a-z0-9]+", " ", lowered_text).strip()
        return f" {key} " in f" {normalized_text} "

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if key and key not in seen:
                result.append(value)
                seen.add(key)
        return result


__all__ = ["CoverageAssessment", "CoverageAssessor"]
