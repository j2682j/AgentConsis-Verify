from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from utils.network_utils import normalize_text

from ..query.relation_plan import RelationPlan
from .relation_evidence import RelationEvidence
from .relation_goal_resolver import RelationGoalResolver


@dataclass(frozen=True)
class RelationBindingResult:
    """Store grounded relation evidence and rejected candidates for diagnostics."""

    evidence: list[RelationEvidence] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


class RelationEvidenceBinder:
    """Bind grounded passage spans to the active subject-relation-object contract."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_.-]*", re.UNICODE)
    SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
    STOPWORDS = {
        "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
        "what", "which", "who", "where", "when", "how",
    }

    def __init__(self, *, resolver: RelationGoalResolver | None = None) -> None:
        self.resolver = resolver or RelationGoalResolver()

    def bind(
        self,
        *,
        plan: RelationPlan,
        documents: Iterable[object],
    ) -> RelationBindingResult:
        goal = plan.active_goal
        if goal is None:
            return RelationBindingResult()
        active_index = next(
            (index for index, item in enumerate(plan.goals) if item.goal_id == goal.goal_id),
            -1,
        )
        expected_next_goal_id = next(
            (
                item.goal_id
                for item in plan.goals[active_index + 1 :]
                if item.state == "pending"
            ),
            "",
        )

        subjects = self.resolver.effective_subjects(plan)
        evidence: list[RelationEvidence] = []
        rejected: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            document_id = normalize_text(getattr(document, "document_id", ""))
            text = normalize_text(getattr(document, "text", ""))
            title = normalize_text(getattr(document, "title", ""))
            contracts = self._bridge_contracts(document)
            for contract in contracts:
                span = normalize_text(str(contract.get("bridge_span") or ""))
                contract_goal_id = normalize_text(str(contract.get("goal_id") or ""))
                if contract_goal_id != goal.goal_id:
                    rejected.append(
                        {
                            "document_id": document_id,
                            "span": span,
                            "reason": "bridge_contract_goal_mismatch",
                        }
                    )
                    continue
                contract_document_id = normalize_text(
                    str(contract.get("document_id") or "")
                )
                if contract_document_id and contract_document_id != document_id:
                    rejected.append(
                        {
                            "document_id": document_id,
                            "span": span,
                            "reason": "bridge_contract_document_mismatch",
                        }
                    )
                    continue
                contract_next_goal_id = normalize_text(
                    str(contract.get("next_goal_id") or "")
                )
                if contract_next_goal_id != expected_next_goal_id:
                    rejected.append(
                        {
                            "document_id": document_id,
                            "span": span,
                            "reason": "bridge_contract_next_goal_mismatch",
                        }
                    )
                    continue
                context = (
                    normalize_text(str(contract.get("context") or ""))
                    or self._context_for_span(text, span)
                    or text
                )
                if span.casefold() not in context.casefold():
                    rejected.append(
                        {
                            "document_id": document_id,
                            "span": span,
                            "reason": "bridge_span_not_grounded_in_contract_context",
                        }
                    )
                    continue
                reason = self._rejection_reason(
                    span=span,
                    context=" ".join(part for part in [title, context] if part),
                    subjects=subjects,
                    relation=goal.relation,
                    target=goal.target,
                )
                if reason:
                    rejected.append(
                        {"document_id": document_id, "span": span, "reason": reason}
                    )
                    continue
                subject = self._matching_subject(subjects, context, title) or (
                    subjects[0] if subjects else goal.subject
                )
                key = (document_id.casefold(), span.casefold())
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    RelationEvidence(
                        goal_id=goal.goal_id,
                        subject=subject,
                        relation=goal.relation,
                        object=span,
                        context=context,
                        document_id=document_id,
                    )
                )
        return RelationBindingResult(evidence=evidence, rejected=rejected)

    def _bridge_contracts(self, document: object) -> list[dict[str, str]]:
        values = list(getattr(document, "bridge_contracts", []) or [])
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, dict):
                to_dict = getattr(value, "to_dict", None)
                if not callable(to_dict):
                    continue
                value = to_dict()
            span = normalize_text(str(value.get("bridge_span") or "")).strip(
                " \"'`.,;:"
            )
            key = span.casefold()
            if len(span) < 2 or key in seen:
                continue
            output.append(dict(value))
            seen.add(key)
        return output

    def _rejection_reason(
        self,
        *,
        span: str,
        context: str,
        subjects: list[str],
        relation: str,
        target: str,
    ) -> str:
        if subjects and not any(self._contains(context, subject) for subject in subjects):
            return "subject_not_grounded_in_context"
        if any(self._equivalent(span, subject) for subject in subjects):
            return "object_repeats_subject"
        relation_terms = self._content_terms(" ".join([relation, target]))
        context_terms = self._content_terms(context)
        if relation_terms and not relation_terms.intersection(context_terms):
            return "relation_not_supported_by_context"
        return ""

    def _context_for_span(self, text: str, span: str) -> str:
        span_key = span.casefold()
        sentences = [normalize_text(item.group(0)) for item in self.SENTENCE_RE.finditer(text)]
        for index, sentence in enumerate(sentences):
            if span_key not in sentence.casefold():
                continue
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            return normalize_text(" ".join(sentences[start:end]))
        return ""

    def _matching_subject(self, subjects: list[str], context: str, title: str) -> str:
        haystack = " ".join([title, context])
        return next((subject for subject in subjects if self._contains(haystack, subject)), "")

    def _contains(self, text: str, phrase: str) -> bool:
        return self._normalize_key(phrase) in self._normalize_key(text)

    def _equivalent(self, left: str, right: str) -> bool:
        return self._normalize_key(left) == self._normalize_key(right)

    def _content_terms(self, text: str) -> set[str]:
        return {
            self._stem(token)
            for token in self.TOKEN_RE.findall(normalize_text(text).casefold())
            if len(token) > 1 and token not in self.STOPWORDS
        }

    def _stem(self, token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]
        return token

    def _normalize_key(self, text: str) -> str:
        return " ".join(self.TOKEN_RE.findall(normalize_text(text).casefold()))


__all__ = ["RelationBindingResult", "RelationEvidenceBinder"]
