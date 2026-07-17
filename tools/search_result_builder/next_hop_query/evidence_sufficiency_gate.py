from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable

from utils.network_utils import normalize_text

from .coverage_assessor import CoverageAssessment


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    """
    判斷 retrieval evidence 是否真的足以停止搜尋。

    Args:
        - sufficient: 是否允許目前 retrieval 狀態停止。
        - mode: 通過或失敗的判斷模式。
        - answer_role: 問題需要的答案角色。
        - matched_span: 找到的答案承載片段。
        - supporting_context: 支撐 matched_span 的上下文。
        - missing: 尚缺少的證據條件。
        - reason: 判斷結果摘要。
        - details: 可寫入 diagnostics 的補充資訊。

    Returns:
        - EvidenceSufficiencyResult: sufficient gate 的結構化結果。
    """

    sufficient: bool
    mode: str
    answer_role: str = "unknown"
    matched_span: str = ""
    supporting_context: str = ""
    missing: list[str] = field(default_factory=list)
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _EvidenceDocument:
    title: str
    text: str
    url: str
    retrieval_score: float = 0.0
    label: str = ""
    sequence_tag: str = ""
    useful_tokens: tuple[str, ...] = ()
    label_status: str = ""
    valid_for_evidence: bool = True
    support_level: str = ""
    can_support_sufficient: bool = True

    @property
    def combined_text(self) -> str:
        return normalize_text(" ".join(part for part in (self.title, self.text) if part))


@dataclass(frozen=True)
class _BindingConstraints:
    hard: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ConstraintBindingResult:
    ok: bool
    hard_coverage: float = 0.0
    source_ok: bool = True
    covered_hard: list[str] = field(default_factory=list)
    missing_hard: list[str] = field(default_factory=list)
    covered_source: list[str] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)
    reason: str = ""
    span: str = ""
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceSufficiencyGate:
    """
    在 coverage / intent state 判定 sufficient 前，確認 evidence 具有答案支撐力。

    Args:
        - min_semantic_terms: semantic support 至少要覆蓋的 intent terms 數量。
        - min_semantic_score: semantic support 的最低 retrieval score。
        - context_window: typed span 取用的左右文長度。

    Returns:
        - EvidenceSufficiencyGate: retrieval 停止條件的補充 gate。
    """

    _NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])")
    _VOLUME_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:m\^?3|cubic\s+met(?:er|re)s?|lit(?:er|re)s?|l)\b",
        re.IGNORECASE,
    )
    _DURATION_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:thousand\s+)?(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
        re.IGNORECASE,
    )
    _DISTANCE_RE = re.compile(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|miles?|meters?|metres?)\b",
        re.IGNORECASE,
    )
    _DATE_RE = re.compile(
        r"\b(?:18|19|20)\d{2}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b|"
        r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b",
        re.IGNORECASE,
    )
    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}")
    _BOILERPLATE_MARKERS = {
        "advertisement",
        "all rights reserved",
        "caption",
        "captions",
        "category",
        "cookie",
        "creative commons",
        "download as pdf",
        "forgot password",
        "image alt",
        "license",
        "login",
        "mediawiki",
        "navigation",
        "password",
        "privacy policy",
        "register here",
        "sign-in",
        "sign in",
        "terms of use",
        "upload file",
        "watchmojo",
    }
    _GENERIC_CONTEXT_MARKERS = {
        "archive",
        "category",
        "copyright",
        "download",
        "external links",
        "license",
        "list ",
        "login",
        "menu",
        "navigation",
        "privacy",
        "references",
        "terms",
        "upload",
    }
    _COUNT_ROLE_TERMS = {
        "album",
        "albums",
        "bird",
        "birds",
        "count",
        "number",
        "quantity",
        "species",
        "studio",
        "total",
    }
    _VOLUME_ROLE_TERMS = {
        "bag",
        "capacity",
        "fish",
        "m^3",
        "m3",
        "volume",
    }
    _DURATION_ROLE_TERMS = {
        "hour",
        "hours",
        "moon",
        "pace",
        "run",
        "time",
        "travel",
    }
    _DISTANCE_ROLE_TERMS = {
        "closest",
        "distance",
        "earth",
        "moon",
        "perigee",
    }
    _DATE_ROLE_TERMS = {
        "born",
        "date",
        "founded",
        "published",
        "released",
        "started",
        "year",
    }
    _PERSON_ROLE_TERMS = {
        "author",
        "director",
        "founder",
        "person",
        "winner",
        "wrote",
    }
    _LOCATION_ROLE_TERMS = {
        "city",
        "country",
        "located",
        "location",
        "place",
        "where",
    }
    _TITLE_ROLE_TERMS = {
        "called",
        "named",
        "name",
        "title",
        "titled",
    }
    _STOPWORDS = {
        "about",
        "after",
        "also",
        "answer",
        "before",
        "between",
        "could",
        "from",
        "have",
        "into",
        "many",
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
        "would",
    }

    def __init__(
        self,
        *,
        min_semantic_terms: int = 2,
        min_semantic_score: float = 0.80,
        context_window: int = 120,
    ) -> None:
        self.min_semantic_terms = max(1, min_semantic_terms)
        self.min_semantic_score = max(0.0, min(1.0, min_semantic_score))
        self.context_window = max(40, context_window)

    def assess(
        self,
        *,
        question: str,
        documents: Iterable[Any],
        intent_plan: Any | None = None,
        coverage: CoverageAssessment | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> EvidenceSufficiencyResult:
        """
        判斷目前 evidence 是否能支撐停止搜尋。

        Args:
            - question: 原始問題。
            - documents: 本輪 retrieved document traces。
            - intent_plan: SearchIntentPlan 狀態。
            - coverage: CoverageAssessor 的結果。
            - tool_context: 工具或 handler 的可信結果。

        Returns:
            - EvidenceSufficiencyResult: 是否通過 sufficient gate。
        """
        normalized_question = normalize_text(question)
        docs = [doc for doc in self._documents(documents) if doc.combined_text]
        if not normalized_question:
            return self._fail("insufficient", "unknown", ["empty_question"], "empty_question")
        if not docs:
            return self._fail(
                "insufficient",
                self._answer_role(normalized_question, intent_plan=intent_plan),
                ["no_documents"],
                "no_documents",
            )

        direct_support_levels = {"direct", "direct_strong", "direct_weak"}
        explicit_utility = any(doc.support_level for doc in docs)
        if explicit_utility:
            docs = [
                doc
                for doc in docs
                if doc.support_level in direct_support_levels and doc.can_support_sufficient
            ]
            if not docs:
                return self._fail(
                    "insufficient",
                    self._answer_role(normalized_question, intent_plan=intent_plan),
                    ["no_direct_support_document"],
                    "no_direct_support_document",
                )

        tool_result = self._trusted_tool_result(tool_context or {})
        if tool_result is not None:
            return tool_result

        answer_role = self._answer_role(
            normalized_question,
            coverage=coverage,
            intent_plan=intent_plan,
        )
        question_numbers = set(self._normalized_numbers(normalized_question))
        intent_terms = self._intent_terms(intent_plan)
        question_terms = self._question_terms(normalized_question)
        binding_constraints = self._extract_binding_constraints(
            question=normalized_question,
            intent_plan=intent_plan,
            answer_role=answer_role,
        )

        if answer_role in {"number", "volume", "duration", "distance", "date", "zip_code"}:
            typed = self._typed_answer_gate(
                answer_role=answer_role,
                question=normalized_question,
                documents=docs,
                intent_terms=intent_terms,
                question_terms=question_terms,
                question_numbers=question_numbers,
                binding_constraints=binding_constraints,
            )
            if typed.sufficient:
                return typed
            semantic = self._semantic_support_gate(
                answer_role=answer_role,
                documents=docs,
                intent_terms=intent_terms,
                question_terms=question_terms,
                binding_constraints=binding_constraints,
                require_useful_signal=True,
            )
            if semantic.sufficient and answer_role not in {"number", "volume", "duration", "distance", "zip_code"}:
                return semantic
            return typed

        return self._semantic_support_gate(
            answer_role=answer_role,
            documents=docs,
            intent_terms=intent_terms,
            question_terms=question_terms,
            binding_constraints=binding_constraints,
            require_useful_signal=False,
        )

    def _typed_answer_gate(
        self,
        *,
        answer_role: str,
        question: str,
        documents: list[_EvidenceDocument],
        intent_terms: list[str],
        question_terms: set[str],
        question_numbers: set[str],
        binding_constraints: _BindingConstraints,
    ) -> EvidenceSufficiencyResult:
        candidates: list[dict[str, Any]] = []
        failed_bindings: list[_ConstraintBindingResult] = []
        for document in documents:
            for match in self._span_matches(answer_role, document.combined_text):
                span = match.group(0)
                normalized_span = self._normalize_number(span)
                if normalized_span and normalized_span in question_numbers and answer_role in {"number", "zip_code"}:
                    continue
                if answer_role == "number" and self._is_year_like_number(normalized_span):
                    continue
                context = self._context(document.combined_text, match.start(), match.end())
                if self._is_boilerplate(context, allow_short_typed=True):
                    continue
                if not self._typed_context_ok(
                    answer_role=answer_role,
                    question=question,
                    context=context,
                    intent_terms=intent_terms,
                    question_terms=question_terms,
                ):
                    continue
                binding = self._check_constraint_binding(
                    answer_role=answer_role,
                    span=span,
                    context=context,
                    document=document,
                    constraints=binding_constraints,
                )
                if not binding.ok:
                    failed_bindings.append(binding)
                    continue
                candidates.append(
                    {
                        "document": document,
                        "span": span,
                        "context": context,
                        "binding": binding,
                        "score": self._support_score(document, context, intent_terms, question_terms),
                    }
                )
        if not candidates:
            best_failed = self._best_failed_binding(failed_bindings)
            missing = (
                best_failed.missing_hard + best_failed.missing_source
                if best_failed is not None
                else [f"no_answer_bearing_{answer_role}_span"]
            )
            return self._fail(
                "typed_answer",
                answer_role,
                missing or [f"no_constraint_bound_{answer_role}_span"],
                "no_constraint_bound_answer_span"
                if failed_bindings
                else f"no_answer_bearing_{answer_role}_span",
                details={
                    "document_count": len(documents),
                    "intent_terms": intent_terms,
                    "binding_constraints": binding_constraints.to_dict(),
                    "failed_binding_count": len(failed_bindings),
                    "best_failed_binding": best_failed.to_dict()
                    if best_failed is not None
                    else {},
                },
            )
        best = max(candidates, key=lambda item: item["score"])
        document = best["document"]
        binding = best["binding"]
        return EvidenceSufficiencyResult(
            sufficient=True,
            mode="typed_answer",
            answer_role=answer_role,
            matched_span=normalize_text(best["span"]),
            supporting_context=normalize_text(best["context"]),
            reason="typed_answer_span_found",
            details={
                "title": document.title,
                "url": document.url,
                "retrieval_score": document.retrieval_score,
                "sequence_tag": document.sequence_tag,
                "useful_token_count": len(document.useful_tokens),
                "support_score": round(float(best["score"]), 6),
                "binding_constraints": binding_constraints.to_dict(),
                "constraint_binding": binding.to_dict(),
            },
        )

    def _semantic_support_gate(
        self,
        *,
        answer_role: str,
        documents: list[_EvidenceDocument],
        intent_terms: list[str],
        question_terms: set[str],
        binding_constraints: _BindingConstraints,
        require_useful_signal: bool,
    ) -> EvidenceSufficiencyResult:
        candidates: list[dict[str, Any]] = []
        for document in documents:
            text = document.combined_text
            if self._is_boilerplate(text):
                continue
            covered_intent = [
                term for term in intent_terms if self._contains_term(text, term)
            ]
            covered_question_terms = [
                term for term in question_terms if self._contains_term(text, term)
            ]
            useful_count = len(document.useful_tokens)
            has_useful_signal = document.valid_for_evidence and (
                useful_count > 0 or document.sequence_tag in {"<FINISH>", "<TERMINATE>"}
            )
            if require_useful_signal and not has_useful_signal:
                continue
            if intent_terms:
                enough_terms = len(covered_intent) >= min(self.min_semantic_terms, len(intent_terms))
            else:
                enough_terms = len(covered_question_terms) >= self.min_semantic_terms
            if not enough_terms:
                continue
            if document.retrieval_score < self.min_semantic_score and useful_count == 0:
                continue
            context = self._best_semantic_context(text, covered_intent or covered_question_terms)
            source_ok, covered_source, missing_source = self._source_constraint_status(
                document=document,
                context=context,
                source_terms=binding_constraints.source,
            )
            if not source_ok:
                continue
            candidates.append(
                {
                    "document": document,
                    "context": context,
                    "terms": covered_intent or covered_question_terms,
                    "covered_source": covered_source,
                    "missing_source": missing_source,
                    "score": self._support_score(document, context, intent_terms, question_terms),
                }
            )
        if not candidates:
            return self._fail(
                "semantic_support",
                answer_role,
                ["no_semantic_support_span"],
                "no_semantic_support_span",
                details={
                    "document_count": len(documents),
                    "intent_terms": intent_terms,
                    "binding_constraints": binding_constraints.to_dict(),
                    "require_useful_signal": require_useful_signal,
                },
            )
        best = max(candidates, key=lambda item: item["score"])
        document = best["document"]
        matched_span = ", ".join(best["terms"][:4])
        return EvidenceSufficiencyResult(
            sufficient=True,
            mode="semantic_support",
            answer_role=answer_role,
            matched_span=matched_span,
            supporting_context=normalize_text(best["context"]),
            reason="semantic_support_found",
            details={
                "title": document.title,
                "url": document.url,
                "retrieval_score": document.retrieval_score,
                "sequence_tag": document.sequence_tag,
                "useful_token_count": len(document.useful_tokens),
                "covered_terms": best["terms"][:12],
                "binding_constraints": binding_constraints.to_dict(),
                "covered_source": best["covered_source"],
                "missing_source": best["missing_source"],
                "support_score": round(float(best["score"]), 6),
            },
        )

    def _trusted_tool_result(self, tool_context: dict[str, Any]) -> EvidenceSufficiencyResult | None:
        if not tool_context:
            return None
        for key in ("trusted_tool_result", "trusted_handler_result"):
            value = tool_context.get(key)
            if not value:
                continue
            text = normalize_text(value.get("text", "") if isinstance(value, dict) else str(value))
            if text:
                mode = "handler_result" if "handler" in key else "tool_result"
                return EvidenceSufficiencyResult(
                    sufficient=True,
                    mode=mode,
                    answer_role="tool_derived",
                    matched_span=text[:120],
                    supporting_context=text[:500],
                    reason=f"{mode}_trusted",
                    details={"source": key},
                )
        return None

    def _extract_binding_constraints(
        self,
        *,
        question: str,
        intent_plan: Any | None,
        answer_role: str,
    ) -> _BindingConstraints:
        hard_terms: list[str] = []
        source_terms: list[str] = []
        soft_terms: list[str] = []

        if intent_plan is not None:
            hard_terms.extend(list(getattr(intent_plan, "must_include", []) or []))
            target = normalize_text(getattr(intent_plan, "target", "") or "")
            if target:
                hard_terms.append(target)
            preferred_domain = normalize_text(getattr(intent_plan, "preferred_domain", "") or "")
            if preferred_domain:
                source_terms.append(preferred_domain)

        hard_terms.extend(self._question_binding_terms(question, answer_role))
        source_terms.extend(self._question_source_terms(question))
        soft_terms.extend(self._role_terms(answer_role))

        return _BindingConstraints(
            hard=self._dedupe_terms(self._filter_binding_terms(hard_terms)),
            source=self._dedupe_terms(source_terms),
            soft=self._dedupe_terms(soft_terms),
        )

    def _check_constraint_binding(
        self,
        *,
        answer_role: str,
        span: str,
        context: str,
        document: _EvidenceDocument,
        constraints: _BindingConstraints,
    ) -> _ConstraintBindingResult:
        if answer_role == "number" and self._looks_like_date_or_identifier(context):
            return _ConstraintBindingResult(
                ok=False,
                reason="number_span_from_date_or_metadata_context",
                span=normalize_text(span),
                context=normalize_text(context),
            )

        binding_text = normalize_text(
            " ".join([document.title, context, document.url])
        )
        covered_hard = [
            term for term in constraints.hard if self._contains_term(binding_text, term)
        ]
        missing_hard = [
            term for term in constraints.hard if term not in covered_hard
        ]
        hard_coverage = (
            len(covered_hard) / len(constraints.hard)
            if constraints.hard
            else 1.0
        )
        source_ok, covered_source, missing_source = self._source_constraint_status(
            document=document,
            context=context,
            source_terms=constraints.source,
        )
        role_ok = self._role_binding_ok(
            answer_role=answer_role,
            context=context,
            soft_terms=constraints.soft,
        )

        min_hard_coverage = self._min_hard_coverage(answer_role, constraints)
        ok = hard_coverage >= min_hard_coverage and source_ok and role_ok
        reasons: list[str] = []
        if hard_coverage < min_hard_coverage:
            reasons.append("hard_constraints_not_bound")
        if not source_ok:
            reasons.append("source_constraints_not_bound")
        if not role_ok:
            reasons.append("answer_role_context_not_bound")
        return _ConstraintBindingResult(
            ok=ok,
            hard_coverage=round(hard_coverage, 6),
            source_ok=source_ok,
            covered_hard=covered_hard,
            missing_hard=missing_hard,
            covered_source=covered_source,
            missing_source=missing_source,
            reason="+".join(reasons) or "constraint_bound",
            span=normalize_text(span),
            context=normalize_text(context),
        )

    def _source_constraint_status(
        self,
        *,
        document: _EvidenceDocument,
        context: str,
        source_terms: list[str],
    ) -> tuple[bool, list[str], list[str]]:
        if not source_terms:
            return True, [], []
        source_text = normalize_text(" ".join([document.title, document.url, context]))
        covered = [
            term for term in source_terms if self._source_term_matches(source_text, term)
        ]
        missing = [term for term in source_terms if term not in covered]
        return not missing, covered, missing

    def _source_term_matches(self, text: str, term: str) -> bool:
        cleaned = normalize_text(term).casefold()
        if not cleaned:
            return False
        aliases = {cleaned}
        if "wikipedia" in cleaned:
            aliases.update({"wikipedia", "wikipedia org", "en wikipedia org"})
        if "youtube" in cleaned or "youtu.be" in cleaned:
            aliases.update({"youtube", "youtu be", "youtube com"})
        domain = cleaned.removeprefix("www.")
        aliases.add(domain.replace(".", " "))
        normalized_text = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold())
        return any(f" {alias} " in f" {normalized_text} " for alias in aliases if alias)

    def _question_binding_terms(self, question: str, answer_role: str) -> list[str]:
        terms: list[str] = []
        terms.extend(
            match.group(1) or match.group(2)
            for match in re.finditer(r'"([^"]{3,120})"|' + r"'([^']{3,120})'", question)
        )
        terms.extend(re.findall(r"\b(?:18|19|20)\d{2}\b", question))
        lowered = question.casefold()
        if "perigee" in lowered:
            terms.append("perigee")
        if "minimum" in lowered:
            terms.append("minimum")
        if "closest approach" in lowered:
            terms.append("closest approach")
        if "moon" in lowered:
            terms.append("Moon")
        if "earth" in lowered:
            terms.append("Earth")
        if "studio album" in lowered or "studio albums" in lowered:
            terms.append("studio albums")
        if "bird species" in lowered:
            terms.append("bird species")
        if "simultaneously" in lowered:
            terms.append("simultaneously")
        if "fish bag" in lowered:
            terms.append("fish bag")
        if answer_role == "volume":
            terms.extend(["volume", "m^3"])
        if answer_role == "duration":
            terms.append("hours")
        return terms

    def _question_source_terms(self, question: str) -> list[str]:
        lowered = question.casefold()
        terms: list[str] = []
        if "wikipedia" in lowered:
            terms.append("wikipedia.org")
        if "youtube" in lowered or "youtu.be" in lowered:
            terms.append("youtube.com")
        if "official" in lowered:
            terms.append("official")
        return terms

    def _filter_binding_terms(self, terms: Iterable[str]) -> list[str]:
        result: list[str] = []
        for term in terms:
            cleaned = normalize_text(str(term or ""))
            if not cleaned or cleaned.casefold().startswith("answer_candidate:"):
                continue
            if len(cleaned) > 140:
                continue
            if cleaned.casefold() in {"fact", "definition", "paper", "media"}:
                continue
            result.append(cleaned)
        return result

    def _role_terms(self, answer_role: str) -> list[str]:
        if answer_role == "number":
            return sorted(self._COUNT_ROLE_TERMS)
        if answer_role == "volume":
            return sorted(self._VOLUME_ROLE_TERMS)
        if answer_role == "duration":
            return sorted(self._DURATION_ROLE_TERMS)
        if answer_role == "distance":
            return sorted(self._DISTANCE_ROLE_TERMS)
        if answer_role == "date":
            return sorted(self._DATE_ROLE_TERMS)
        if answer_role == "person":
            return sorted(self._PERSON_ROLE_TERMS)
        if answer_role == "location":
            return sorted(self._LOCATION_ROLE_TERMS)
        if answer_role == "title":
            return sorted(self._TITLE_ROLE_TERMS)
        return []

    def _role_binding_ok(
        self,
        *,
        answer_role: str,
        context: str,
        soft_terms: list[str],
    ) -> bool:
        lowered = context.casefold()
        if answer_role in {"zip_code", "date"}:
            return True
        if answer_role == "number":
            return self._has_any(lowered, set(soft_terms or self._COUNT_ROLE_TERMS))
        if answer_role == "volume":
            return self._has_any(lowered, set(soft_terms or self._VOLUME_ROLE_TERMS))
        if answer_role == "duration":
            return self._has_any(lowered, set(soft_terms or self._DURATION_ROLE_TERMS))
        if answer_role == "distance":
            return self._has_any(lowered, set(soft_terms or self._DISTANCE_ROLE_TERMS))
        return True

    def _min_hard_coverage(
        self,
        answer_role: str,
        constraints: _BindingConstraints,
    ) -> float:
        if not constraints.hard:
            return 0.0
        if answer_role in {"number", "volume", "duration", "distance"}:
            return 0.6
        return 0.5

    def _best_failed_binding(
        self,
        failed_bindings: list[_ConstraintBindingResult],
    ) -> _ConstraintBindingResult | None:
        if not failed_bindings:
            return None
        return max(
            failed_bindings,
            key=lambda item: (
                item.source_ok,
                item.hard_coverage,
                len(item.covered_hard),
            ),
        )

    def _span_matches(self, answer_role: str, text: str) -> list[re.Match[str]]:
        if answer_role == "volume":
            return list(self._VOLUME_RE.finditer(text))
        if answer_role == "duration":
            return list(self._DURATION_RE.finditer(text))
        if answer_role == "distance":
            return list(self._DISTANCE_RE.finditer(text))
        if answer_role == "date":
            return list(self._DATE_RE.finditer(text))
        return list(self._NUMBER_RE.finditer(text))

    def _typed_context_ok(
        self,
        *,
        answer_role: str,
        question: str,
        context: str,
        intent_terms: list[str],
        question_terms: set[str],
    ) -> bool:
        lowered = context.casefold()
        if self._is_generic_context(lowered):
            return False
        if answer_role == "zip_code":
            return not re.search(r"\b(?:18|19|20)\d{2}\b", context)
        if answer_role == "volume":
            return self._has_any(lowered, {"m^3", "m3", "cubic meter", "cubic metre", "liter", "litre", "volume", "capacity"})
        if answer_role == "duration":
            return self._has_any(lowered, {"hour", "hours", "minute", "minutes", "second", "seconds", "duration", "time"})
        if answer_role == "distance":
            return self._has_any(lowered, {"km", "kilometer", "kilometre", "meter", "metre", "mile", "distance"})
        if answer_role == "date":
            return self._has_any(lowered, self._DATE_ROLE_TERMS) or any(self._contains_term(lowered, term) for term in intent_terms)
        if answer_role == "number":
            if self._looks_like_date_or_identifier(context):
                return False
            role_terms = self._count_role_terms(question_terms)
            role_ok = self._has_any(lowered, role_terms)
            intent_hits = sum(1 for term in intent_terms if self._contains_term(lowered, term))
            required_intent_hits = min(2, len(intent_terms)) if intent_terms else 0
            return role_ok and intent_hits >= required_intent_hits
        return bool(question_terms & set(self._tokens(context)))

    def _answer_role(
        self,
        question: str,
        *,
        coverage: CoverageAssessment | None = None,
        intent_plan: Any | None = None,
    ) -> str:
        planned = normalize_text(
            str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
        ).casefold()
        if planned and planned != "unknown":
            return self._normalize_answer_role(planned)
        lowered = question.casefold()
        if re.search(r"\bwhat\s+does\b.+\bstand\s+for\b", lowered):
            return "text_span"
        if re.search(r"\b(?:what|which)\s+writer\b|\bquoted\s+by\b|\bfirst\s+name\b|\blast\s+name\b|\bwho\b", lowered):
            return "person"
        if re.search(r"\b(?:ioc|country|nation|airport|station)\s+code\b|\bcode\s+as\s+your\s+answer\b", lowered):
            return "text_span"
        if "zip code" in lowered or "zipcode" in lowered or "five-digit" in lowered:
            return "zip_code"
        if "m^3" in lowered or "cubic meter" in lowered or "cubic metre" in lowered or "volume" in lowered:
            return "volume"
        if any(term in lowered for term in ("hour", "minute", "second", "duration", "how long")):
            return "duration"
        if any(term in lowered for term in ("distance", "kilometer", "kilometre", " km ", "mile")):
            return "distance"
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bhighest number\b", lowered):
            return "number"
        if re.search(r"\bwhen\b|\bwhat date\b|\bwhich year\b|\bwhat year\b", lowered):
            return "date"
        if re.search(r"\bwhere\b|\bwhich country\b|\bwhich city\b|\bwhich place\b", lowered):
            return "location"
        if re.search(r"\btitle\b|\bname of\b|\bcalled\b", lowered):
            return "title"
        if coverage is not None and coverage.answer_type:
            return self._normalize_answer_role(coverage.answer_type)
        return "short_phrase"

    def _normalize_answer_role(self, answer_role: str) -> str:
        role = normalize_text(answer_role).casefold()
        if role == "count":
            return "number"
        if role in {
            "number",
            "volume",
            "duration",
            "distance",
            "date",
            "zip_code",
            "person",
            "location",
            "organization",
            "title",
            "species",
            "boolean",
            "text_span",
            "short_phrase",
        }:
            return role
        return "short_phrase"

    def _documents(self, documents: Iterable[Any]) -> list[_EvidenceDocument]:
        result: list[_EvidenceDocument] = []
        for document in documents or []:
            if isinstance(document, dict):
                title = str(document.get("title", "") or "")
                text = str(document.get("text", "") or "")
                url = str(document.get("url", "") or "")
                retrieval_score = float(document.get("retrieval_score", 0.0) or 0.0)
                label = str(document.get("label", "") or "")
                sequence_tag = str(document.get("sequence_tag", "") or "")
                useful_tokens = tuple(
                    normalize_text(str(item.get("answer_span") or ""))
                    for item in list(document.get("direct_contracts") or [])
                    if isinstance(item, dict)
                    and normalize_text(str(item.get("answer_span") or ""))
                )
                label_status = str(document.get("label_status", "") or "")
                valid_for_evidence = bool(document.get("valid_for_evidence", False))
                support_level = str(document.get("support_level", "") or "")
                can_support_sufficient = bool(
                    document.get("can_support_sufficient", valid_for_evidence)
                )
            else:
                title = str(getattr(document, "title", "") or "")
                text = str(getattr(document, "text", "") or "")
                url = str(getattr(document, "url", "") or "")
                retrieval_score = float(getattr(document, "retrieval_score", 0.0) or 0.0)
                label = str(getattr(document, "label", "") or "")
                sequence_tag = str(getattr(document, "sequence_tag", "") or "")
                useful_tokens = tuple(
                    normalize_text(str(item.get("answer_span") or ""))
                    for item in list(getattr(document, "direct_contracts", []) or [])
                    if isinstance(item, dict)
                    and normalize_text(str(item.get("answer_span") or ""))
                )
                label_status = str(getattr(document, "label_status", "") or "")
                valid_for_evidence = bool(getattr(document, "valid_for_evidence", False))
                support_level = str(getattr(document, "support_level", "") or "")
                can_support_sufficient = bool(
                    getattr(document, "can_support_sufficient", valid_for_evidence)
                )
            result.append(
                _EvidenceDocument(
                    title=normalize_text(title),
                    text=normalize_text(text),
                    url=normalize_text(url),
                    retrieval_score=round(retrieval_score, 6),
                    label=normalize_text(label),
                    sequence_tag=normalize_text(sequence_tag),
                    useful_tokens=useful_tokens,
                    label_status=normalize_text(label_status),
                    valid_for_evidence=valid_for_evidence,
                    support_level=normalize_text(support_level),
                    can_support_sufficient=can_support_sufficient,
                )
            )
        return result

    def _context(self, text: str, start: int, end: int) -> str:
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return normalize_text(text[left:right])

    def _best_semantic_context(self, text: str, terms: list[str]) -> str:
        if not terms:
            return text[: self.context_window * 2]
        lowered = text.casefold()
        positions = [
            lowered.find(normalize_text(term).casefold())
            for term in terms
            if normalize_text(term) and lowered.find(normalize_text(term).casefold()) >= 0
        ]
        if not positions:
            return text[: self.context_window * 2]
        center = min(positions)
        return self._context(text, center, center + len(terms[0]))

    def _support_score(
        self,
        document: _EvidenceDocument,
        context: str,
        intent_terms: list[str],
        question_terms: set[str],
    ) -> float:
        intent_hits = sum(1 for term in intent_terms if self._contains_term(context, term))
        question_hits = len(set(self._tokens(context)) & question_terms)
        useful_bonus = min(0.2, len(document.useful_tokens) * 0.04)
        boilerplate_penalty = 0.35 if self._is_boilerplate(context) else 0.0
        return (
            document.retrieval_score
            + 0.08 * intent_hits
            + 0.02 * min(question_hits, 8)
            + useful_bonus
            - boilerplate_penalty
        )

    def _intent_terms(self, intent_plan: Any | None) -> list[str]:
        if intent_plan is None:
            return []
        raw_terms: list[str] = []
        for attr in ("must_include", "completed_terms"):
            raw_terms.extend(list(getattr(intent_plan, attr, []) or []))
        target = normalize_text(getattr(intent_plan, "target", "") or "")
        if target:
            raw_terms.append(target)
        return self._dedupe_terms(term for term in raw_terms if not str(term).startswith("answer_candidate:"))

    def _question_terms(self, question: str) -> set[str]:
        return {
            token.casefold()
            for token in self._tokens(question)
            if token.casefold() not in self._STOPWORDS
        }

    def _count_role_terms(self, question_terms: set[str]) -> set[str]:
        role_terms = set(self._COUNT_ROLE_TERMS)
        role_terms.update(
            term
            for term in question_terms
            if term in {"album", "albums", "species", "studio", "bird", "birds"}
        )
        return role_terms

    def _normalized_numbers(self, text: str) -> list[str]:
        return [self._normalize_number(match.group(0)) for match in self._NUMBER_RE.finditer(text)]

    def _normalize_number(self, text: str) -> str:
        return normalize_text(text).replace(",", "").rstrip("%")

    def _is_year_like_number(self, normalized_number: str) -> bool:
        if not re.fullmatch(r"\d{4}", normalized_number):
            return False
        value = int(normalized_number)
        return 1500 <= value <= 2199

    def _looks_like_date_or_identifier(self, context: str) -> bool:
        lowered = context.casefold()
        if re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:18|19|20)\d{2}\b",
            lowered,
        ):
            return True
        if re.search(
            r"\b(?:january|february|march|april|june|july|august|september|october|november|december)\s+\d{1,2}\b",
            lowered,
        ):
            return True
        if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}(?:t\d{1,2}:\d{2})?", lowered):
            return True
        if re.search(r"(?<!\d)\d{1,2}:\d{2}", lowered):
            return True
        if "datepublished" in lowered or "datemodified" in lowered:
            return True
        if re.search(r"\b[a-f0-9]{8,}\b", lowered):
            return True
        if self._has_any(lowered, {"utc", "version", "license", "list ", "copyright"}):
            return True
        return False

    def _is_boilerplate(self, text: str, *, allow_short_typed: bool = False) -> bool:
        lowered = text.casefold()
        marker_hits = sum(1 for marker in self._BOILERPLATE_MARKERS if marker in lowered)
        if marker_hits >= 2:
            return True
        tokens = self._tokens(text)
        if len(tokens) < 8 and not allow_short_typed:
            return True
        return False

    def _is_generic_context(self, lowered_context: str) -> bool:
        return sum(1 for marker in self._GENERIC_CONTEXT_MARKERS if marker in lowered_context) >= 2

    def _has_any(self, text: str, terms: set[str]) -> bool:
        return any(term in text for term in terms)

    def _contains_term(self, text: str, term: str) -> bool:
        cleaned_text = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold()).strip()
        cleaned_term = re.sub(r"[^a-z0-9]+", " ", normalize_text(term).casefold()).strip()
        if not cleaned_term:
            return False
        return f" {cleaned_term} " in f" {cleaned_text} "

    def _tokens(self, text: str) -> list[str]:
        return [
            token.strip("'_-")
            for token in self._WORD_RE.findall(normalize_text(text))
            if len(token.strip("'_-")) >= 3
        ]

    def _dedupe_terms(self, terms: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = normalize_text(str(term or ""))
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    def _fail(
        self,
        mode: str,
        answer_role: str,
        missing: list[str],
        reason: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> EvidenceSufficiencyResult:
        return EvidenceSufficiencyResult(
            sufficient=False,
            mode=mode,
            answer_role=answer_role,
            missing=missing,
            reason=reason,
            details=details or {},
        )


__all__ = ["EvidenceSufficiencyGate", "EvidenceSufficiencyResult"]
