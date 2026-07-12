from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..query.search_intent_planner import SearchIntentPlan


@dataclass(frozen=True)
class RejectedNextHopSpan:
    """
    Store a rejected next-hop evidence span for diagnostics.

    Args:
        - document_id: Source document ID.
        - span: Rejected bridge span.
        - reason: Rejection reason.

    Returns:
        - RejectedNextHopSpan: Debug record for next-hop evidence selection.
    """

    document_id: str
    span: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NextHopEvidenceSelection:
    """
    Store bridge spans selected for next-hop query composition.

    Args:
        - bridge_spans: Clean bridge spans allowed into the composer.
        - selected_document_ids: Documents contributing selected bridge spans.
        - rejected: Rejected span diagnostics.
        - metadata: Aggregate selection metadata.

    Returns:
        - NextHopEvidenceSelection: Strict next-hop evidence input.
    """

    bridge_spans: list[str] = field(default_factory=list)
    selected_document_ids: list[str] = field(default_factory=list)
    rejected: list[RejectedNextHopSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_spans": list(self.bridge_spans),
            "selected_document_ids": list(self.selected_document_ids),
            "rejected": [item.to_dict() for item in self.rejected],
            "metadata": dict(self.metadata),
        }


class NextHopEvidenceSelector:
    """
    Select strict evidence-side bridge spans for next-hop query composition.

    Args:
        - max_bridge_spans: Maximum bridge spans exposed to the composer.
        - min_span_chars: Minimum span length unless the span contains letters and digits.
        - max_span_chars: Maximum span length.

    Returns:
        - NextHopEvidenceSelector: Document-to-bridge-span selector.
    """

    NOISE_TERMS = {
        "about",
        "advertisement",
        "captcha",
        "cloudflare",
        "content",
        "copyright",
        "e-mail",
        "email",
        "headings",
        "login",
        "metadata",
        "password",
        "privacy",
        "sign in",
        "source",
        "table",
        "title",
        "user name",
        "verification",
        "wordplays",
    }

    def __init__(
        self,
        *,
        max_bridge_spans: int = 3,
        min_span_chars: int = 3,
        max_span_chars: int = 80,
    ) -> None:
        self.max_bridge_spans = max(1, max_bridge_spans)
        self.min_span_chars = max(1, min_span_chars)
        self.max_span_chars = max(self.min_span_chars, max_span_chars)

    def select(
        self,
        *,
        documents: list[Any],
        question: str,
        intent_plan: SearchIntentPlan | None = None,
    ) -> NextHopEvidenceSelection:
        """
        Select bridge spans from valid next-hop documents only.

        Args:
            - documents: RetrievedDocumentTrace-like objects.
            - question: Original task question.
            - intent_plan: Planner state, kept only for diagnostics.

        Returns:
            - NextHopEvidenceSelection: Clean bridge spans and rejected diagnostics.
        """
        question_key = self._match_key(question)
        selected: list[str] = []
        selected_document_ids: list[str] = []
        rejected: list[RejectedNextHopSpan] = []
        seen: set[str] = set()
        candidate_count = 0

        for document in sorted(
            documents,
            key=lambda item: float(getattr(item, "retrieval_score", 0.0) or 0.0),
            reverse=True,
        ):
            document_id = normalize_text(str(getattr(document, "document_id", "") or ""))
            if not bool(getattr(document, "valid_for_next_hop", False)):
                rejected.append(self._reject(document_id, "", "document_not_valid_for_next_hop"))
                continue
            if normalize_text(str(getattr(document, "support_level", "") or "")) != "bridge":
                rejected.append(self._reject(document_id, "", "document_not_bridge_support"))
                continue

            spans = list(getattr(document, "bridge_spans", []) or [])
            if not spans:
                rejected.append(self._reject(document_id, "", "document_without_bridge_spans"))
                continue

            for span in spans:
                candidate_count += 1
                cleaned, reason = self._clean_span(span, question_key=question_key)
                if reason:
                    rejected.append(self._reject(document_id, str(span or ""), reason))
                    continue
                key = self._match_key(cleaned)
                if not key:
                    rejected.append(self._reject(document_id, cleaned, "empty_match_key"))
                    continue
                if key in seen or self._contained_by_existing(key, seen):
                    rejected.append(self._reject(document_id, cleaned, "duplicate_or_contained_span"))
                    continue
                selected.append(cleaned)
                selected_document_ids.append(document_id)
                seen.add(key)
                if len(selected) >= self.max_bridge_spans:
                    return self._result(
                        selected=selected,
                        selected_document_ids=selected_document_ids,
                        rejected=rejected,
                        candidate_count=candidate_count,
                        intent_plan=intent_plan,
                    )

        return self._result(
            selected=selected,
            selected_document_ids=selected_document_ids,
            rejected=rejected,
            candidate_count=candidate_count,
            intent_plan=intent_plan,
        )

    def _clean_span(self, span: object, *, question_key: str) -> tuple[str, str]:
        text = normalize_text(str(span or "")).strip(" \"'`.,;:")
        if not text:
            return "", "empty_span"
        if len(text) < self.min_span_chars:
            return "", "span_too_short"
        if len(text) > self.max_span_chars:
            return "", "span_too_long"
        if re.fullmatch(r"[\W_]+", text):
            return "", "punctuation_span"
        if re.fullmatch(r"\d+(?:\.\d+)?", text) or re.fullmatch(r"(?:18|19|20)\d{2}", text):
            return "", "pure_number_or_year"

        key = self._match_key(text)
        if not key:
            return "", "empty_match_key"
        if question_key and key in question_key:
            return "", "question_echo_span"
        if self._noise_ratio(text) >= 0.5:
            return "", "noise_span"
        return text, ""

    def _result(
        self,
        *,
        selected: list[str],
        selected_document_ids: list[str],
        rejected: list[RejectedNextHopSpan],
        candidate_count: int,
        intent_plan: SearchIntentPlan | None,
    ) -> NextHopEvidenceSelection:
        return NextHopEvidenceSelection(
            bridge_spans=list(selected),
            selected_document_ids=list(dict.fromkeys(selected_document_ids)),
            rejected=list(rejected),
            metadata={
                "method": "strict_valid_bridge_span_selection",
                "candidate_count": candidate_count,
                "selected_count": len(selected),
                "rejected_count": len(rejected),
                "answer_role": normalize_text(
                    str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
                ),
            },
        )

    def _reject(self, document_id: str, span: str, reason: str) -> RejectedNextHopSpan:
        return RejectedNextHopSpan(
            document_id=document_id,
            span=normalize_text(span),
            reason=reason,
        )

    def _contained_by_existing(self, key: str, seen: set[str]) -> bool:
        return any(key in existing or existing in key for existing in seen)

    def _noise_ratio(self, text: str) -> float:
        tokens = self._keywords(text)
        if not tokens:
            return 1.0
        noisy = sum(1 for token in tokens if token in self.NOISE_TERMS)
        return noisy / len(tokens)

    def _keywords(self, text: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]{1,}", normalize_text(text).casefold()):
            token = token.strip("'_.-")
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _match_key(self, text: str) -> str:
        return " ".join(self._keywords(text))


__all__ = [
    "NextHopEvidenceSelection",
    "NextHopEvidenceSelector",
    "RejectedNextHopSpan",
]
