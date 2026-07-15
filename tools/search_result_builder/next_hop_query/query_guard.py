from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from utils.network_utils import normalize_text

from ..query.search_intent_planner import SearchIntentPlan


@dataclass(frozen=True)
class NextHopQueryGuardResult:
    """
    記錄 next-hop query 是否符合目前搜尋意圖。

    Args:
     - accepted: proposed query 是否可直接使用。
     - query: 最終採用的 query。
     - proposed_query: Filter 或 fallback 原本產生的 query。
     - fallback_query: rejected 時由 intent plan 產生的備援 query。
     - reason: 接受或拒絕原因。
     - must_include_coverage: 必要詞覆蓋比例。
     - missing_terms: proposed query 缺少的必要詞。

    Returns:
     - NextHopQueryGuardResult: query guard 的判斷結果。

    """

    accepted: bool
    query: str
    proposed_query: str
    fallback_query: str = ""
    reason: str = ""
    must_include_coverage: float = 0.0
    missing_terms: list[str] = field(default_factory=list)
    retained_terms: list[str] = field(default_factory=list)
    support_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NextHopQueryGuard:
    """
    檢查下一跳 query 是否仍保留原始搜尋意圖。

    Args:
     - min_must_include_coverage: proposed query 至少需覆蓋的必要詞比例。
     - max_useful_spans: fallback query 最多加入的 evidence spans 數量。
     - max_query_chars: fallback query 的最大長度。

    Returns:
     - NextHopQueryGuard: retrieval control 使用的 query 品質守門器。

    """

    GENERIC_TOKENS = {
        "abstract",
        "academic",
        "advantage",
        "analysis",
        "article",
        "articles",
        "caption",
        "captions",
        "content",
        "contents",
        "definition",
        "example",
        "external",
        "history",
        "index",
        "introduction",
        "journal",
        "list",
        "name",
        "official",
        "page",
        "paper",
        "pdf",
        "references",
        "review",
        "search",
        "source",
        "title",
        "video",
        "wikipedia",
    }
    NOISE_TOKENS = {
        "coughing",
        "powering",
        "wheezing",
        "whirring",
        "subscribe",
        "login",
        "advertisement",
        "cookies",
        "javascript",
        "menu",
        "navigation",
        "privacy",
        "share",
    }

    def __init__(
        self,
        *,
        min_must_include_coverage: float = 0.5,
        max_useful_spans: int = 3,
        max_query_chars: int = 260,
    ) -> None:
        self.min_must_include_coverage = max(0.0, min(1.0, min_must_include_coverage))
        self.max_useful_spans = max(0, max_useful_spans)
        self.max_query_chars = max(80, max_query_chars)

    def validate(
        self,
        *,
        original_question: str,
        current_query: str,
        proposed_next_query: str,
        intent_plan: SearchIntentPlan | None,
        useful_spans: Iterable[str] | None = None,
        seen_query_keys: set[str] | None = None,
    ) -> NextHopQueryGuardResult:
        """
        驗證 proposed next-hop query，必要時回退到 intent-aware fallback。

        Args:
         - original_question: 原始任務問題。
         - current_query: 目前這一輪 retrieval 使用的 query。
         - proposed_next_query: Filter 或 fallback 產生的下一跳 query。
         - intent_plan: SearchIntentPlanner 產生並由 state tracker 更新的 plan。
         - useful_spans: Labeler 或 evidence 中抽出的有用片段。
         - seen_query_keys: 已搜尋 query 的正規化 key。

        Returns:
         - NextHopQueryGuardResult: 最終採用 query 與 guard 診斷。

        """
        proposed = normalize_text(proposed_next_query)
        spans = self._clean_spans(useful_spans or [])
        fallback = self.build_fallback_query(
            original_question=original_question,
            current_query=current_query,
            intent_plan=intent_plan,
            useful_spans=spans,
        )
        coverage, missing, retained = self._must_include_coverage(
            proposed,
            intent_plan,
        )
        support_requirements = self._support_requirements(intent_plan)

        reject_reason = self._reject_reason(
            proposed=proposed,
            current_query=current_query,
            intent_plan=intent_plan,
            coverage=coverage,
            missing_terms=missing,
            seen_query_keys=seen_query_keys or set(),
        )
        if not reject_reason:
            return NextHopQueryGuardResult(
                accepted=True,
                query=proposed,
                proposed_query=proposed,
                fallback_query=fallback,
                reason="accepted",
                must_include_coverage=coverage,
                missing_terms=missing,
                retained_terms=retained,
                support_requirements=support_requirements,
            )

        selected = fallback if fallback else proposed
        if selected and self._is_duplicate_query(selected, seen_query_keys or set()):
            selected = ""
            reject_reason = f"{reject_reason}; fallback_duplicate"
        return NextHopQueryGuardResult(
            accepted=False,
            query=selected,
            proposed_query=proposed,
            fallback_query=fallback,
            reason=reject_reason,
            must_include_coverage=coverage,
            missing_terms=missing,
            retained_terms=retained,
            support_requirements=support_requirements,
        )

    def build_fallback_query(
        self,
        *,
        original_question: str,
        current_query: str,
        intent_plan: SearchIntentPlan | None,
        useful_spans: Iterable[str] | None = None,
    ) -> str:
        """
        根據 intent plan 與 missing terms 建立保守的下一跳 query。

        Args:
         - original_question: 原始任務問題。
         - current_query: 目前 retrieval query。
         - intent_plan: 目前搜尋狀態。
         - useful_spans: 可補充的 evidence span。

        Returns:
         - str: intent-aware fallback query。

        """
        parts: list[str] = []
        if intent_plan is not None and intent_plan.preferred_domain:
            parts.append(f"site:{intent_plan.preferred_domain}")
        if intent_plan is not None and intent_plan.target:
            parts.append(intent_plan.target)

        if intent_plan is not None:
            for term in list(intent_plan.missing_terms or []) + list(intent_plan.must_include or []):
                if self._is_internal_requirement(term):
                    continue
                if not self._contains_equivalent(parts, term):
                    parts.append(term)
        else:
            parts.extend(self._keywords(original_question)[:6])

        for span in self._clean_spans(useful_spans or []):
            if len(parts) >= 8:
                break
            if not self._is_noise_span(span) and not self._contains_equivalent(parts, span):
                parts.append(span)

        if len(parts) < 3:
            for term in self._keywords(current_query)[:6]:
                if not self._contains_equivalent(parts, term):
                    parts.append(term)

        return normalize_text(" ".join(parts))[: self.max_query_chars].strip()

    def _reject_reason(
        self,
        *,
        proposed: str,
        current_query: str,
        intent_plan: SearchIntentPlan | None,
        coverage: float,
        missing_terms: list[str],
        seen_query_keys: set[str],
    ) -> str:
        if not proposed:
            return "empty_query"
        if self._is_duplicate_query(proposed, seen_query_keys):
            return "duplicate_query"
        if self._query_key(proposed) == self._query_key(current_query):
            return "same_as_current_query"
        if intent_plan is not None and intent_plan.preferred_domain:
            domain = intent_plan.preferred_domain.casefold()
            if f"site:{domain}" in current_query.casefold() and domain not in proposed.casefold():
                return "preferred_domain_dropped"
        if missing_terms and coverage < self.min_must_include_coverage:
            return "low_must_include_coverage"
        if self._noise_ratio(proposed) >= 0.45:
            return "noise_dominated_query"
        if self._generic_ratio(proposed) >= 0.65 and len(self._keywords(proposed)) >= 4:
            return "generic_query"
        return ""

    def _must_include_coverage(
        self,
        query: str,
        intent_plan: SearchIntentPlan | None,
    ) -> tuple[float, list[str], list[str]]:
        if intent_plan is None:
            return 1.0, [], []
        missing_terms = self._search_terms(intent_plan.missing_terms or [])
        must_include_terms = self._search_terms(intent_plan.must_include or [])
        terms = missing_terms or must_include_terms
        if not terms:
            return 1.0, [], []
        missing: list[str] = []
        retained: list[str] = []
        for term in terms:
            if self._term_covered(term, query):
                retained.append(term)
            else:
                missing.append(term)
        return round(len(retained) / len(terms), 6), missing, retained

    def _support_requirements(self, intent_plan: SearchIntentPlan | None) -> list[str]:
        if intent_plan is None:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for term in list(intent_plan.missing_terms or []) + list(intent_plan.must_include or []):
            cleaned = normalize_text(str(term or "")).strip()
            if not cleaned or not self._is_support_requirement(cleaned):
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _search_terms(self, terms: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = normalize_text(str(term or "")).strip()
            if not cleaned or self._is_internal_requirement(cleaned):
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _is_support_requirement(self, term: str) -> bool:
        return str(term or "").strip().casefold().startswith("answer_support:")

    def _is_internal_requirement(self, term: str) -> bool:
        text = str(term or "").strip().casefold()
        return text.startswith(
            (
                "answer_support:",
                "answer_candidate:",
                "preferred_domain:",
            )
        )

    def _term_covered(self, term: str, query: str) -> bool:
        term_tokens = self._keywords(term)
        query_tokens = set(self._keywords(query))
        if not term_tokens:
            return True
        if normalize_text(term).casefold() in normalize_text(query).casefold():
            return True
        important = [
            token
            for token in term_tokens
            if token not in self.GENERIC_TOKENS and len(token) > 2
        ] or term_tokens
        required = max(1, min(len(important), int(round(len(important) * 0.67))))
        return sum(1 for token in important if token in query_tokens) >= required

    def _clean_spans(self, spans: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for span in spans:
            cleaned = normalize_text(str(span or "")).strip(" \"'`.,;:")
            if not cleaned or len(cleaned) < 3 or len(cleaned) > 80:
                continue
            key = self._query_key(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
            if len(result) >= self.max_useful_spans:
                break
        return result

    def _is_noise_span(self, span: str) -> bool:
        tokens = self._keywords(span)
        if not tokens:
            return True
        if all(re.fullmatch(r"\d{1,4}(?:[-:/]\d{1,4})?", token) for token in tokens):
            return True
        return self._noise_ratio(span) >= 0.45

    def _noise_ratio(self, query: str) -> float:
        tokens = self._keywords(query)
        if not tokens:
            return 1.0
        noisy = sum(1 for token in tokens if token in self.NOISE_TOKENS)
        return noisy / len(tokens)

    def _generic_ratio(self, query: str) -> float:
        tokens = self._keywords(query)
        if not tokens:
            return 1.0
        generic = sum(1 for token in tokens if token in self.GENERIC_TOKENS)
        return generic / len(tokens)

    def _contains_equivalent(self, parts: Iterable[str], term: str) -> bool:
        return any(self._term_covered(term, part) or self._term_covered(part, term) for part in parts)

    def _is_duplicate_query(self, query: str, seen_query_keys: set[str]) -> bool:
        key = self._query_key(query)
        return bool(key and key in seen_query_keys)

    def _query_key(self, query: str) -> str:
        return normalize_text(query).casefold().strip(" \"'`.,;:-")

    def _keywords(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{1,}", normalize_text(text).casefold())
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip("_-'")
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result


__all__ = ["NextHopQueryGuard", "NextHopQueryGuardResult"]
