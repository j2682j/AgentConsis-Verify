from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from ..config import EvidenceItem, SearchSignals
from ..query.semantic_impact import SemanticImpactScorer
from .answer_target_extractor import AnswerTarget, AnswerTargetExtractor

_SPACY_CACHE: dict[str, Any] = {}


@dataclass
class RetrievalDecision:
    """
    保存第一次搜尋的證據充足性判定。

    Args:
        - need_next_hop: 是否需要執行下一跳搜尋。
        - reason: 主要判定原因。
        - confidence: 證據充足性分數。
        - missing_info: 未達最低門檻的訊號。
        - scores: NER、語意相關性與限制覆蓋分數。

    Returns:
        - RetrievalDecision: Retrieval control 判定結果。
    """

    need_next_hop: bool
    reason: str
    confidence: float = 0.0
    missing_info: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)


class RetrievalController:
    """
    使用 NER coverage、encoder relevance 與 constraint coverage 判斷證據是否充足。

    Args:
        - semantic_scorer: 共用的 encoder embedding scorer。
        - answer_target_extractor: dependency-based 答案焦點抽取器。
        - nlp: 可注入的 spaCy language pipeline。
        - spacy_model: 預設載入的 spaCy 模型。
        - sufficiency_threshold: 三項訊號加權後的停止門檻。
        - min_entity_coverage: 題目實體最低覆蓋率。
        - min_semantic_relevance: 問題與證據最低語意相關性。
        - min_constraint_coverage: 年份、來源與答案角色最低覆蓋率。
        - semantic_top_k: 語意相關性採用的最高分 evidence 數量。

    Returns:
        - RetrievalController: 第一次搜尋的證據充足性控制器。
    """

    ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE"}
    ANSWER_ROLE_LABELS = {
        "person": {"PERSON"},
        "organization": {"ORG"},
        "place": {"GPE", "LOC", "FAC"},
        "date": {"DATE", "TIME"},
        "number": {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"},
        "duration": {"TIME", "QUANTITY", "CARDINAL"},
        "distance": {"QUANTITY", "CARDINAL"},
        "volume": {"QUANTITY", "CARDINAL"},
        "percentage": {"PERCENT", "CARDINAL"},
    }
    ROLE_UNIT_TERMS = {
        "duration": {
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "second",
            "seconds",
            "week",
            "weeks",
            "year",
            "years",
        },
        "distance": {
            "kilometer",
            "kilometers",
            "km",
            "meter",
            "meters",
            "metre",
            "metres",
            "mile",
            "miles",
        },
        "volume": {
            "liter",
            "liters",
            "litre",
            "litres",
            "ml",
            "m3",
            "cubic meter",
            "cubic meters",
        },
        "percentage": {"percent", "percentage", "%"},
    }
    SOURCE_TERMS = {
        "article",
        "book",
        "database",
        "journal",
        "newspaper",
        "official",
        "paper",
        "publication",
        "report",
        "source",
        "study",
        "website",
    }

    def __init__(
        self,
        *,
        semantic_scorer: SemanticImpactScorer | None = None,
        answer_target_extractor: AnswerTargetExtractor | None = None,
        nlp: Any | None = None,
        spacy_model: str = "en_core_web_md",
        sufficiency_threshold: float = 0.65,
        min_entity_coverage: float = 0.75,
        min_semantic_relevance: float = 0.45,
        min_constraint_coverage: float = 0.75,
        semantic_top_k: int = 3,
    ) -> None:
        self.semantic_scorer = semantic_scorer or SemanticImpactScorer()
        self.nlp = nlp
        self.spacy_model = spacy_model
        self.answer_target_extractor = answer_target_extractor
        self.sufficiency_threshold = sufficiency_threshold
        self.min_entity_coverage = min_entity_coverage
        self.min_semantic_relevance = min_semantic_relevance
        self.min_constraint_coverage = min_constraint_coverage
        self.semantic_top_k = max(1, semantic_top_k)

    def assess(
        self,
        *,
        evidence_items: list[EvidenceItem],
        question: str = "",
        search_signals: SearchSignals | None = None,
    ) -> RetrievalDecision:
        """
        判斷目前證據是否足以停止搜尋。

        Args:
            - evidence_items: evidence conversion 產生的 evidence chunks。
            - question: 原始問題。
            - search_signals: 搜尋訊號，目前保留於介面供診斷使用。

        Returns:
            - RetrievalDecision: 是否需要 next-hop 與三項判定分數。
        """
        del search_signals
        if not evidence_items:
            return RetrievalDecision(
                need_next_hop=True,
                reason="no_evidence",
                confidence=0.0,
                missing_info=["evidence"],
                scores=self._empty_scores(),
            )

        evidence_texts = [
            normalize_text(" ".join(part for part in (item.title, item.text) if part))
            for item in evidence_items
        ]
        evidence_texts = [text for text in evidence_texts if text]
        combined_evidence = normalize_text(" ".join(evidence_texts))

        entity_coverage, entity_details = self._entity_coverage(question, combined_evidence)
        semantic_relevance, semantic_details = self._semantic_relevance(question, evidence_texts)
        constraint_coverage, constraint_details = self._constraint_coverage(
            question,
            combined_evidence,
        )

        sufficiency_score = round(
            max(
                0.0,
                min(
                    1.0,
                    0.30 * entity_coverage
                    + 0.40 * semantic_relevance
                    + 0.30 * constraint_coverage,
                ),
            ),
            6,
        )
        missing_info: list[str] = []
        if entity_details["required_count"] > 0 and entity_coverage < self.min_entity_coverage:
            missing_info.append("entity_coverage")
        if semantic_relevance < self.min_semantic_relevance:
            missing_info.append("semantic_relevance")
        if (
            constraint_details["required_count"] > 0
            and constraint_coverage < self.min_constraint_coverage
        ):
            missing_info.append("constraint_coverage")

        scores: dict[str, Any] = {
            "sufficiency_score": sufficiency_score,
            "entity_coverage": round(entity_coverage, 6),
            "semantic_relevance": round(semantic_relevance, 6),
            "constraint_coverage": round(constraint_coverage, 6),
            "entity_details": entity_details,
            "semantic_details": semantic_details,
            "constraint_details": constraint_details,
        }
        if sufficiency_score < self.sufficiency_threshold or missing_info:
            return RetrievalDecision(
                need_next_hop=True,
                reason="insufficient_evidence",
                confidence=sufficiency_score,
                missing_info=missing_info,
                scores=scores,
            )
        return RetrievalDecision(
            need_next_hop=False,
            reason="sufficient_evidence",
            confidence=sufficiency_score,
            missing_info=[],
            scores=scores,
        )

    def rank_evidence(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        """
        使用充足性判定的三項訊號重新排序 evidence。

        Args:
            - question: 原始問題。
            - evidence_items: H1/H2 合併並去重後的 evidence。

        Returns:
            - list[EvidenceItem]: 依綜合相關性由高至低排序的 evidence。
            - list[dict[str, Any]]: 每個 evidence 的排序分數明細。
        """
        if not evidence_items:
            return [], []

        evidence_texts = [
            normalize_text(" ".join(part for part in (item.title, item.text) if part))
            for item in evidence_items
        ]
        try:
            similarities = self.semantic_scorer.semantic_similarities(question, evidence_texts)
        except Exception:
            similarities = [0.0] * len(evidence_items)

        ranked: list[tuple[float, int, EvidenceItem, dict[str, Any]]] = []
        for index, (item, text) in enumerate(zip(evidence_items, evidence_texts)):
            entity_coverage, _ = self._entity_coverage(question, text)
            constraint_coverage, _ = self._constraint_coverage(question, text)
            semantic_relevance = (
                max(0.0, min(1.0, float(similarities[index])))
                if index < len(similarities)
                else 0.0
            )
            score = round(
                0.30 * entity_coverage
                + 0.40 * semantic_relevance
                + 0.30 * constraint_coverage,
                6,
            )
            item.evidence_quality = score
            if "cross_hop_rerank" not in item.cleaning_reasons:
                item.cleaning_reasons.append("cross_hop_rerank")
            details = {
                "evidence_id": item.evidence_id,
                "query_id": item.query_id,
                "source_id": item.source_id,
                "score": score,
                "entity_coverage": round(entity_coverage, 6),
                "semantic_relevance": round(semantic_relevance, 6),
                "constraint_coverage": round(constraint_coverage, 6),
            }
            ranked.append((score, -index, item, details))

        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return (
            [entry[2] for entry in ranked],
            [entry[3] for entry in ranked],
        )

    def _entity_coverage(self, question: str, evidence_text: str) -> tuple[float, dict[str, Any]]:
        nlp = self._get_nlp()
        if nlp is None:
            return 1.0, {
                "available": False,
                "required_count": 0,
                "matched_count": 0,
                "required": [],
                "matched": [],
                "missing": [],
            }

        required = self._entities(question, labels=self.ENTITY_LABELS)
        if not required:
            return 1.0, {
                "available": True,
                "required_count": 0,
                "matched_count": 0,
                "required": [],
                "matched": [],
                "missing": [],
            }

        evidence_key = self._match_key(evidence_text)
        matched = [entity for entity in required if self._entity_in_text(entity, evidence_key)]
        missing = [entity for entity in required if entity not in matched]
        return len(matched) / len(required), {
            "available": True,
            "required_count": len(required),
            "matched_count": len(matched),
            "required": required,
            "matched": matched,
            "missing": missing,
        }

    def _semantic_relevance(
        self,
        question: str,
        evidence_texts: list[str],
    ) -> tuple[float, dict[str, Any]]:
        if not evidence_texts:
            return 0.0, {"top_k": 0, "similarities": [], "error": ""}
        try:
            similarities = self.semantic_scorer.semantic_similarities(question, evidence_texts)
        except Exception as exc:
            return 0.0, {
                "top_k": 0,
                "similarities": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        bounded = [max(0.0, min(1.0, float(score))) for score in similarities]
        ranked = sorted(bounded, reverse=True)
        selected = ranked[: min(self.semantic_top_k, len(ranked))]
        score = sum(selected) / len(selected) if selected else 0.0
        return score, {
            "top_k": len(selected),
            "similarities": [round(value, 6) for value in bounded],
            "selected": [round(value, 6) for value in selected],
            "error": "",
        }

    def _constraint_coverage(
        self,
        question: str,
        evidence_text: str,
    ) -> tuple[float, dict[str, Any]]:
        constraints: list[tuple[str, str, bool]] = []
        evidence_key = self._match_key(evidence_text)

        for year in self._dedupe(re.findall(r"\b(?:18|19|20)\d{2}\b", question)):
            constraints.append(("year", year, year in evidence_text))

        lowered_question = normalize_text(question).lower()
        for source_term in sorted(self.SOURCE_TERMS):
            if re.search(rf"\b{re.escape(source_term)}\b", lowered_question):
                constraints.append(
                    ("source", source_term, f" {source_term} " in evidence_key)
                )

        answer_target = self._answer_target(question)
        if answer_target.role:
            role_matched = self._answer_target_covered(
                answer_target,
                evidence_text,
            )
            constraints.append(
                ("answer_role", answer_target.role, role_matched)
            )

        if not constraints:
            return 1.0, {
                "required_count": 0,
                "matched_count": 0,
                "required": [],
                "missing": [],
            }

        matched = [f"{kind}:{value}" for kind, value, covered in constraints if covered]
        missing = [f"{kind}:{value}" for kind, value, covered in constraints if not covered]
        return len(matched) / len(constraints), {
            "required_count": len(constraints),
            "matched_count": len(matched),
            "required": [f"{kind}:{value}" for kind, value, _ in constraints],
            "matched": matched,
            "missing": missing,
            "answer_target": answer_target.to_dict(),
        }

    def _get_nlp(self) -> Any | None:
        if self.nlp is not None:
            return self.nlp
        cached = _SPACY_CACHE.get(self.spacy_model)
        if cached is not None:
            self.nlp = cached
            return cached
        try:
            import spacy

            self.nlp = spacy.load(self.spacy_model)
        except Exception:
            try:
                import spacy

                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None
        if self.nlp is not None:
            _SPACY_CACHE[self.spacy_model] = self.nlp
        return self.nlp

    def _entities(self, text: str, *, labels: set[str]) -> list[str]:
        nlp = self._get_nlp()
        if nlp is None or not normalize_text(text):
            return []
        entities = [
            normalize_text(entity.text)
            for entity in nlp(text).ents
            if entity.label_ in labels and normalize_text(entity.text)
        ]
        return self._dedupe(entities)

    def _entity_in_text(self, entity: str, evidence_key: str) -> bool:
        key = self._match_key(entity).strip()
        return bool(key and f" {key} " in evidence_key)

    def _answer_target(self, question: str) -> AnswerTarget:
        extractor = self.answer_target_extractor
        if extractor is None:
            extractor = AnswerTargetExtractor(
                nlp=self._get_nlp(),
                semantic_scorer=self.semantic_scorer,
            )
            self.answer_target_extractor = extractor
        elif extractor.nlp is None:
            extractor.nlp = self._get_nlp()
        return extractor.extract(question)

    def _answer_target_covered(
        self,
        target: AnswerTarget,
        evidence_text: str,
    ) -> bool:
        labels = self.ANSWER_ROLE_LABELS.get(target.role, set())
        entity_match = bool(
            labels and self._entities(evidence_text, labels=labels)
        )
        if target.role not in self.ROLE_UNIT_TERMS:
            return entity_match

        evidence_key = self._match_key(evidence_text)
        unit_terms = set(self.ROLE_UNIT_TERMS[target.role])
        if target.unit:
            unit_terms.add(target.unit)
            unit_terms.add(f"{target.unit}s")
        unit_match = any(
            f" {self._match_key(term).strip()} " in evidence_key
            for term in unit_terms
            if self._match_key(term).strip()
        )
        numeric_match = entity_match or any(
            char.isdigit() for char in evidence_text
        )
        return numeric_match and unit_match

    def _match_key(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).lower())
        return f" {' '.join(cleaned.split())} "

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = self._match_key(value).strip()
            if key and key not in seen:
                result.append(value)
                seen.add(key)
        return result

    def _empty_scores(self) -> dict[str, Any]:
        return {
            "sufficiency_score": 0.0,
            "entity_coverage": 0.0,
            "semantic_relevance": 0.0,
            "constraint_coverage": 0.0,
            "entity_details": {},
            "semantic_details": {},
            "constraint_details": {},
        }


__all__ = ["RetrievalController", "RetrievalDecision"]
