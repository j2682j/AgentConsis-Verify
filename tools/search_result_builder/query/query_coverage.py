from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import TYPE_CHECKING, Any

from utils.network_utils import normalize_text

if TYPE_CHECKING:
    from .search_intent_planner import SearchIntentPlan


_SPACY_MODEL = None


@dataclass(frozen=True)
class QueryConstraint:
    """
    保存 query 必須盡量保留的問題資訊單位。

    Args:
        - text: 原始 constraint 文字。
        - kind: constraint 類型，例如 salient_span、entity、date_number。
        - source: constraint 來源。

    Returns:
        - QueryConstraint: Query coverage 計算用的資訊單位。
    """

    text: str
    kind: str
    source: str


@dataclass(frozen=True)
class QueryCoverageResult:
    """
    保存單條 query 的 coverage 結果。

    Args:
        - query: 原始 query。
        - coverage_score: 直接計算的 constraint 覆蓋率。
        - covered: 已覆蓋 constraints。
        - missing: 未覆蓋 constraints。
        - original_index: query 原始順序。

    Returns:
        - QueryCoverageResult: 可排序與寫入 diagnostics 的結果。
    """

    query: str
    coverage_score: float
    covered: list[QueryConstraint]
    missing: list[QueryConstraint]
    original_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "coverage_score": self.coverage_score,
            "covered": [asdict(item) for item in self.covered],
            "missing": [asdict(item) for item in self.missing],
            "original_index": self.original_index,
        }


class QueryCoverageChecker:
    """
    以本地 constraint coverage 檢查 query 是否保留問題硬條件。

    coverage_score 直接定義為 covered_constraints / total_constraints，
    不使用手刻加權分數。Constraint 類型只用於 diagnostics 與 repair query
    組裝，不參與分數加權。

    Args:
        - min_repair_coverage: 低於此覆蓋率時新增 repair query。
        - max_repair_terms: repair query 最多使用多少個 constraint。
        - max_repair_chars: repair query 最大字元數。

    Returns:
        - QueryCoverageChecker: Query rerank / repair 工具。
    """

    QUOTED_RE = re.compile(r"['\"]([^'\"]{2,120})['\"]")
    YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
    NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
    DATE_RE = re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    )
    CAPITALIZED_PHRASE_RE = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'&.-]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'&.-]{2,}|of|the|and|for|in|on|to)){1,7}\b"
    )
    SOURCE_HINT_RE = re.compile(
        r"\b(?:wikipedia|official website|official site|paper|article|journal|"
        r"dataset|transcript|youtube|video|pdf|github|imdb|archive|"
        r"website|webpage|page|table)\b",
        re.IGNORECASE,
    )
    ANSWER_ROLE_RE = re.compile(
        r"\b(?:city|country|surname|first name|last name|contract number|"
        r"species|title|color|colour|count|number|year|date|name|"
        r"organization|organisation|place|location|author|team|code|"
        r"identifier|value|answer)\b",
        re.IGNORECASE,
    )
    WORD_RE = re.compile(r"[a-z0-9][a-z0-9'&._-]*", re.IGNORECASE)
    WEAK_CONSTRAINTS = {
        "answer",
        "article",
        "attached",
        "document",
        "evidence",
        "final",
        "find",
        "give",
        "image",
        "question",
        "search",
        "source",
        "table",
        "text",
        "unknown",
        "website",
    }
    SPACY_LABELS = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "DATE",
        "TIME",
        "FAC",
        "WORK_OF_ART",
        "PRODUCT",
        "EVENT",
        "LAW",
        "LANGUAGE",
        "NORP",
    }

    def __init__(
        self,
        *,
        min_repair_coverage: float = 0.55,
        max_repair_terms: int = 14,
        max_repair_chars: int = 180,
    ) -> None:
        self.min_repair_coverage = max(0.0, min(1.0, min_repair_coverage))
        self.max_repair_terms = max(1, max_repair_terms)
        self.max_repair_chars = max(40, max_repair_chars)
        self.last_diagnostics: dict[str, Any] = {}

    def improve_queries(
        self,
        *,
        question: str,
        queries: list[str],
        salient_spans: list[Any] | None = None,
        intent_plan: "SearchIntentPlan | None" = None,
        max_queries: int = 5,
    ) -> tuple[list[str], dict[str, Any]]:
        """
        依 constraint coverage 重新排序 query，必要時加入 repair query。

        Args:
            - question: 原始問題。
            - queries: 模型產生的 query candidates。
            - salient_spans: semantic-impact spans。
            - max_queries: 最多輸出 query 數量。

        Returns:
            - tuple[list[str], dict[str, Any]]: 改善後 queries 與 diagnostics。
        """
        cleaned_queries = self._dedupe_texts(queries)
        if intent_plan is not None:
            return self._improve_queries_with_intent(
                question=question,
                queries=cleaned_queries,
                salient_spans=salient_spans or [],
                intent_plan=intent_plan,
                max_queries=max_queries,
            )

        constraints = self.extract_constraints(
            question=question,
            salient_spans=salient_spans or [],
        )
        results = [
            self.score_query(query, constraints, original_index=index)
            for index, query in enumerate(cleaned_queries)
        ]
        ranked = sorted(
            results,
            key=lambda result: (-result.coverage_score, result.original_index),
        )

        repair_query = self.build_repair_query(constraints)
        repair_added = False
        best_score = ranked[0].coverage_score if ranked else 0.0
        should_repair = not ranked or best_score < self.min_repair_coverage
        if repair_query and should_repair:
            if self._query_key(repair_query) not in {
                self._query_key(result.query) for result in ranked
            }:
                repair_result = self.score_query(
                    repair_query,
                    constraints,
                    original_index=-1,
                )
                ranked.insert(0, repair_result)
                repair_added = True

        output_queries = self._dedupe_texts(
            [result.query for result in ranked]
        )[: max(1, max_queries)]
        diagnostics = {
            "method": "direct_constraint_coverage",
            "score_formula": "covered_constraints / total_constraints",
            "min_repair_coverage": self.min_repair_coverage,
            "constraints": [asdict(item) for item in constraints],
            "query_results": [result.to_dict() for result in ranked],
            "repair_query": repair_query,
            "repair_added": repair_added,
        }
        self.last_diagnostics = diagnostics
        return output_queries, diagnostics

    def _improve_queries_with_intent(
        self,
        *,
        question: str,
        queries: list[str],
        salient_spans: list[Any],
        intent_plan: "SearchIntentPlan",
        max_queries: int,
    ) -> tuple[list[str], dict[str, Any]]:
        has_intent_constraints = bool(
            intent_plan.must_include
            or intent_plan.avoid_terms
            or intent_plan.preferred_domain
        )
        if not has_intent_constraints:
            queries, diagnostics = self.improve_queries(
                question=question,
                queries=queries,
                salient_spans=salient_spans,
                intent_plan=None,
                max_queries=max_queries,
            )
            diagnostics["intent_plan"] = intent_plan.to_dict()
            diagnostics["intent_constraints_used"] = False
            return queries, diagnostics

        constraints = self.extract_constraints(
            question=question,
            salient_spans=salient_spans,
        )
        intent_constraints = self._intent_constraints(intent_plan)
        intent_queries = self._intent_seed_queries(intent_plan)
        candidates = self._dedupe_texts(intent_queries + queries)
        scored = [
            self._score_intent_query(
                query,
                intent_plan=intent_plan,
                intent_constraints=intent_constraints,
                original_index=index,
            )
            for index, query in enumerate(candidates)
        ]
        kept = [item for item in scored if not item["avoid_terms_found"]]
        if not kept:
            kept = scored
        ranked = sorted(
            kept,
            key=lambda item: (
                -item["coverage_score"],
                -int(item["preferred_domain_used"]),
                item["avoid_violation_count"],
                item["original_index"],
            ),
        )
        output_queries = self._dedupe_texts(
            [item["query"] for item in ranked]
        )[: max(1, max_queries)]
        diagnostics = {
            "method": "intent_constraint_coverage",
            "score_formula": "intent must_include coverage minus avoid violations",
            "intent_plan": intent_plan.to_dict(),
            "legacy_constraints": [asdict(item) for item in constraints],
            "constraints": [asdict(item) for item in intent_constraints],
            "query_results": ranked,
            "repair_query": intent_queries[0] if intent_queries else "",
            "repair_added": bool(intent_queries),
        }
        self.last_diagnostics = diagnostics
        return output_queries, diagnostics

    def extract_constraints(
        self,
        *,
        question: str,
        salient_spans: list[Any],
    ) -> list[QueryConstraint]:
        text = normalize_text(question)
        constraints: list[QueryConstraint] = []
        for span in salient_spans:
            span_text = getattr(span, "text", span)
            constraints.extend(
                self._constraint_items(
                    [str(span_text or "")],
                    kind="salient_span",
                    source="semantic_impact",
                )
            )
        constraints.extend(
            self._constraint_items(
                self.QUOTED_RE.findall(text),
                kind="quoted_phrase",
                source="question",
            )
        )
        constraints.extend(
            self._constraint_items(
                self._spacy_entities(text),
                kind="entity",
                source="spacy_ner",
            )
        )
        constraints.extend(
            self._constraint_items(
                self.DATE_RE.findall(text)
                + self.YEAR_RE.findall(text)
                + self.NUMBER_RE.findall(text),
                kind="date_number",
                source="regex",
            )
        )
        constraints.extend(
            self._constraint_items(
                self.CAPITALIZED_PHRASE_RE.findall(text),
                kind="capitalized_phrase",
                source="regex",
            )
        )
        constraints.extend(
            self._constraint_items(
                self.SOURCE_HINT_RE.findall(text),
                kind="source_hint",
                source="regex",
            )
        )
        constraints.extend(
            self._constraint_items(
                self.ANSWER_ROLE_RE.findall(text),
                kind="answer_role",
                source="regex",
            )
        )
        return self._dedupe_constraints(constraints)

    def score_query(
        self,
        query: str,
        constraints: list[QueryConstraint],
        *,
        original_index: int,
    ) -> QueryCoverageResult:
        covered: list[QueryConstraint] = []
        missing: list[QueryConstraint] = []
        for constraint in constraints:
            if self._covers(query, constraint.text):
                covered.append(constraint)
            else:
                missing.append(constraint)
        if not constraints:
            coverage_score = 1.0 if normalize_text(query) else 0.0
        else:
            coverage_score = len(covered) / len(constraints)
        return QueryCoverageResult(
            query=normalize_text(query),
            coverage_score=round(max(0.0, min(1.0, coverage_score)), 6),
            covered=covered,
            missing=missing,
            original_index=original_index,
        )

    def _intent_constraints(self, intent_plan: "SearchIntentPlan") -> list[QueryConstraint]:
        constraints = self._constraint_items(
            intent_plan.must_include,
            kind="must_include",
            source="search_intent",
        )
        if intent_plan.preferred_domain:
            constraints.append(
                QueryConstraint(
                    text=intent_plan.preferred_domain,
                    kind="preferred_domain",
                    source="search_intent",
                )
            )
        return self._dedupe_constraints(constraints)

    def _intent_seed_queries(self, intent_plan: "SearchIntentPlan") -> list[str]:
        terms = [term for term in intent_plan.must_include if normalize_text(term)]
        target = normalize_text(intent_plan.target)
        queries: list[str] = []
        base = normalize_text(" ".join(terms[:4]) or target)
        if intent_plan.preferred_domain and base:
            queries.append(f"site:{intent_plan.preferred_domain} {base}")
        if base:
            queries.append(base)
        elif target:
            queries.append(target)
        return queries

    def _score_intent_query(
        self,
        query: str,
        *,
        intent_plan: "SearchIntentPlan",
        intent_constraints: list[QueryConstraint],
        original_index: int,
    ) -> dict[str, Any]:
        must_constraints = [
            item for item in intent_constraints if item.kind == "must_include"
        ]
        covered = [
            item for item in must_constraints if self._covers(query, item.text)
        ]
        missing = [
            item for item in must_constraints if not self._covers(query, item.text)
        ]
        avoid_terms_found = [
            term for term in intent_plan.avoid_terms if self._covers(query, term)
        ]
        preferred_domain_used = (
            bool(intent_plan.preferred_domain)
            and self._query_uses_domain(query, intent_plan.preferred_domain)
        )
        if must_constraints:
            coverage_score = len(covered) / len(must_constraints)
        else:
            coverage_score = 1.0 if normalize_text(query) else 0.0
        if intent_plan.preferred_domain and preferred_domain_used:
            coverage_score = min(1.0, coverage_score + 0.05)
        if avoid_terms_found:
            coverage_score = max(0.0, coverage_score - 0.25 * len(avoid_terms_found))
        return {
            "query": normalize_text(query),
            "coverage_score": round(max(0.0, min(1.0, coverage_score)), 6),
            "covered": [asdict(item) for item in covered],
            "missing": [asdict(item) for item in missing],
            "must_include_covered": [item.text for item in covered],
            "must_include_missing": [item.text for item in missing],
            "avoid_terms_found": avoid_terms_found,
            "avoid_violation_count": len(avoid_terms_found),
            "preferred_domain_used": preferred_domain_used,
            "preferred_domain": intent_plan.preferred_domain,
            "original_index": original_index,
        }

    def build_repair_query(self, constraints: list[QueryConstraint]) -> str:
        ordered = sorted(
            constraints,
            key=lambda item: (
                self._repair_priority(item.kind),
                len(self._tokens(item.text)),
                len(item.text),
            ),
        )
        parts: list[str] = []
        seen: set[str] = set()
        for constraint in ordered:
            text = normalize_text(constraint.text)
            key = self._constraint_key(text)
            if not text or key in seen:
                continue
            candidate = " ".join(parts + [text])
            if len(candidate) > self.max_repair_chars:
                continue
            parts.append(text)
            seen.add(key)
            if len(parts) >= self.max_repair_terms:
                break
        return normalize_text(" ".join(parts))[: self.max_repair_chars].strip()

    def _constraint_items(
        self,
        texts: list[str],
        *,
        kind: str,
        source: str,
    ) -> list[QueryConstraint]:
        return [
            QueryConstraint(text=text, kind=kind, source=source)
            for text in texts
            if self._is_valid_constraint(text)
        ]

    def _dedupe_constraints(
        self,
        constraints: list[QueryConstraint],
    ) -> list[QueryConstraint]:
        result: list[QueryConstraint] = []
        seen: set[str] = set()
        for constraint in constraints:
            text = normalize_text(constraint.text)
            key = self._constraint_key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(QueryConstraint(text=text, kind=constraint.kind, source=constraint.source))
        return result

    def _is_valid_constraint(self, text: str) -> bool:
        cleaned = normalize_text(text).strip(" ,.;:!?()[]{}'\"")
        if not cleaned:
            return False
        key = self._constraint_key(cleaned)
        if not key or key in self.WEAK_CONSTRAINTS:
            return False
        tokens = self._tokens(cleaned)
        if not tokens:
            return False
        if len(tokens) == 1:
            token = tokens[0]
            if token in self.WEAK_CONSTRAINTS:
                return False
            if token.isdigit():
                return len(token) >= 2
            return len(token) >= 4
        return True

    def _covers(self, query: str, constraint: str) -> bool:
        query_norm = self._normalized_space(query)
        constraint_norm = self._normalized_space(constraint)
        if not query_norm or not constraint_norm:
            return False
        if constraint_norm in query_norm:
            return True
        query_tokens = set(self._tokens(query_norm))
        constraint_tokens = self._tokens(constraint_norm)
        if not constraint_tokens:
            return False
        if len(constraint_tokens) == 1:
            token = constraint_tokens[0]
            if token.isdigit() and len(token) <= 3:
                return token in query_tokens
            return any(
                token == query_token
                or (len(token) >= 6 and token in query_token)
                or (len(query_token) >= 6 and query_token in token)
                for query_token in query_tokens
            )
        informative = [
            token
            for token in constraint_tokens
            if token not in self.WEAK_CONSTRAINTS and len(token) >= 3
        ]
        if not informative:
            informative = constraint_tokens
        matched = sum(1 for token in informative if token in query_tokens)
        return matched == len(informative)

    def _spacy_entities(self, text: str) -> list[str]:
        global _SPACY_MODEL
        if _SPACY_MODEL is False:
            return []
        if _SPACY_MODEL is None:
            try:
                import spacy  # type: ignore

                try:
                    _SPACY_MODEL = spacy.load("en_core_web_md")
                except Exception:
                    _SPACY_MODEL = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_MODEL = False
                return []
        if _SPACY_MODEL is False:
            return []
        try:
            doc = _SPACY_MODEL(text)
        except Exception:
            return []
        return [
            ent.text
            for ent in doc.ents
            if ent.label_ in self.SPACY_LABELS
        ]

    def _repair_priority(self, kind: str) -> int:
        order = {
            "quoted_phrase": 0,
            "entity": 1,
            "date_number": 2,
            "capitalized_phrase": 3,
            "salient_span": 4,
            "source_hint": 5,
            "answer_role": 6,
        }
        return order.get(kind, 9)

    def _dedupe_texts(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for text in texts:
            cleaned = normalize_text(text)
            key = self._query_key(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _query_key(self, text: str) -> str:
        return self._normalized_space(text)

    def _constraint_key(self, text: str) -> str:
        return self._normalized_space(text)

    def _normalized_space(self, text: str) -> str:
        tokens = self._tokens(text)
        return " ".join(tokens)

    def _tokens(self, text: str) -> list[str]:
        normalized = normalize_text(text).casefold()
        return [
            token.strip("'&._-")
            for token in self.WORD_RE.findall(normalized)
            if token.strip("'&._-")
        ]

    def _query_uses_domain(self, query: str, domain: str) -> bool:
        query_norm = normalize_text(query).casefold()
        domain_norm = normalize_text(domain).casefold()
        return f"site:{domain_norm}" in query_norm or domain_norm in query_norm


__all__ = [
    "QueryConstraint",
    "QueryCoverageChecker",
    "QueryCoverageResult",
]
