from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from .question_role_extractor import QuestionRole
from .semantic_impact import SemanticImpactScorer
from .span_repair import SalientSpan


@dataclass
class ClassifiedSpan:
    """
    依照 span 與 question role 的關係分類後的搜尋片段。

    Args:
        - text: 修復後的 span 文字。
        - role: query generation 使用的功能角色。
        - confidence: 最佳角色與第二佳角色的分數差。
        - similarities: relation text 與 prototype 的原始相似度。
        - score: span semantic impact 分數。
        - repair_source: span repair 來源。
        - original_text: 修復前 span。
        - context: span 附近上下文。
        - question_role: 問題答案角色資訊。
        - entity_label: spaCy NER label。
        - role_scores: 加上 soft prior 後的最終角色分數。

    Returns:
        - ClassifiedSpan: 可序列化的 span classification 結果。
    """

    text: str
    role: str
    confidence: float
    similarities: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    repair_source: str = ""
    original_text: str = ""
    context: str = ""
    question_role: dict[str, Any] = field(default_factory=dict)
    entity_label: str = ""
    role_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpanRoleClassifier:
    """
    使用 relation-to-question-role similarity 判斷 span 的搜尋功能角色。

    Args:
        - scorer: 共用 encoder embedding scorer。
        - context_chars: span 前後保留的上下文字元數。
        - min_confidence: 最佳角色與第二佳角色的最低差距。

    Returns:
        - SpanRoleClassifier: query span 功能分類器。
    """

    PROTOTYPES: dict[str, str] = {
        "source_clue": (
            "The span is the main subject, source, named entity, title, URL, organization, "
            "person, work, page, video, paper, episode, or lookup anchor used to find the answer target."
        ),
        "constraint": (
            "The span is a condition, time range, version, source restriction, official source, "
            "date, year, location filter, qualifier, or requirement that narrows which answer is valid."
        ),
        "answer_target": (
            "The span describes the requested answer type, value, field, count target, measurement, "
            "entity type, title type, location type, or exact information that should be returned."
        ),
        "format_instruction": (
            "The span describes output formatting, rounding, units to report, comma separators, "
            "exact wording, capitalization, or how the final answer should be written."
        ),
        "other": (
            "The span is incidental, vague, local narrative, background text, or not useful for web retrieval."
        ),
    }
    ROLE_ORDER = [
        "source_clue",
        "constraint",
        "answer_target",
        "format_instruction",
        "other",
    ]
    SOURCE_ENTITY_LABELS = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "EVENT",
        "WORK_OF_ART",
        "PRODUCT",
        "LAW",
        "NORP",
    }
    CONSTRAINT_ENTITY_LABELS = {
        "DATE",
        "TIME",
        "MONEY",
        "QUANTITY",
        "PERCENT",
        "CARDINAL",
        "ORDINAL",
    }
    FORMAT_TERMS = {
        "capitalization",
        "comma",
        "exact",
        "exactly",
        "format",
        "lowercase",
        "nearest",
        "round",
        "rounded",
        "separator",
        "separators",
        "uppercase",
    }
    CONSTRAINT_TERMS = {
        "after",
        "before",
        "between",
        "closest",
        "during",
        "first",
        "included",
        "last",
        "latest",
        "maximum",
        "minimum",
        "official",
        "page",
        "script",
        "source",
        "version",
        "wikipedia",
    }

    def __init__(
        self,
        *,
        scorer: SemanticImpactScorer,
        context_chars: int = 80,
        min_confidence: float = 0.015,
    ) -> None:
        self.scorer = scorer
        self.context_chars = max(20, context_chars)
        self.min_confidence = max(0.0, min_confidence)
        self._nlp: Any | None = None
        self._nlp_load_attempted = False

    def classify(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        question_role: QuestionRole | None = None,
    ) -> list[ClassifiedSpan]:
        """
        將 repaired spans 分成 clue / constraint / answer target 等角色。

        Args:
            - question: 原始任務問題。
            - spans: 已修復與重新計分的 salient spans。
            - question_role: QuestionRoleExtractor 抽出的答案角色。

        Returns:
            - list[ClassifiedSpan]: role-labeled spans。
        """
        if not spans:
            return []
        question_role = question_role or QuestionRole()
        prototypes = [self.PROTOTYPES[role] for role in self.ROLE_ORDER]
        entity_labels = self._entity_labels_for_spans(question, spans)
        output: list[ClassifiedSpan] = []

        for span in spans:
            context = self._span_context(question, span)
            relation_text = self._classification_text(
                question=question,
                span=span,
                context=context,
                question_role=question_role,
            )
            try:
                similarities = self.scorer.semantic_similarities(relation_text, prototypes)
            except Exception:
                similarities = [0.0 for _ in prototypes]
            similarity_map = self._similarity_map(similarities)
            entity_label = entity_labels.get((span.start, span.end), "")
            role_scores = self._role_scores(
                span=span,
                context=context,
                similarities=similarity_map,
                question_role=question_role,
                entity_label=entity_label,
            )
            role, confidence = self._role_from_scores(role_scores)
            output.append(
                ClassifiedSpan(
                    text=span.text,
                    role=role,
                    confidence=confidence,
                    similarities=similarity_map,
                    score=float(span.score or 0.0),
                    repair_source=span.repair_source,
                    original_text=span.original_text or span.text,
                    context=context,
                    question_role=question_role.to_dict(),
                    entity_label=entity_label,
                    role_scores=role_scores,
                )
            )
        return output

    def grouped(self, spans: list[ClassifiedSpan]) -> dict[str, list[ClassifiedSpan]]:
        grouped: dict[str, list[ClassifiedSpan]] = {
            role: [] for role in [*self.ROLE_ORDER, "weak_generic"]
        }
        for span in spans:
            grouped.setdefault(span.role, []).append(span)
        for values in grouped.values():
            values.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return grouped

    def _similarity_map(self, similarities: list[float]) -> dict[str, float]:
        return {
            role: round(float(score), 6)
            for role, score in zip(self.ROLE_ORDER, similarities, strict=False)
        }

    def _role_scores(
        self,
        *,
        span: SalientSpan,
        context: str,
        similarities: dict[str, float],
        question_role: QuestionRole,
        entity_label: str,
    ) -> dict[str, float]:
        scores = {role: float(similarities.get(role, 0.0)) for role in self.ROLE_ORDER}
        span_text = normalize_text(span.text)
        lowered = f" {normalize_text(span.text + ' ' + context).lower()} "

        if self._overlaps_question_role(span, question_role):
            scores["answer_target"] += 0.08
        if self._looks_like_url(span_text):
            scores["source_clue"] += 0.08
        if entity_label in self.SOURCE_ENTITY_LABELS:
            scores["source_clue"] += 0.045
        if entity_label in self.CONSTRAINT_ENTITY_LABELS:
            scores["constraint"] += 0.035
        if self._looks_like_title(span_text):
            scores["source_clue"] += 0.035
        if any(f" {term} " in lowered for term in self.CONSTRAINT_TERMS):
            scores["constraint"] += 0.035
        if any(f" {term} " in lowered for term in self.FORMAT_TERMS):
            scores["format_instruction"] += 0.05
        if len(span_text.split()) <= 1 and not entity_label and not self._looks_like_url(span_text):
            scores["other"] += 0.025
        return {role: round(score, 6) for role, score in scores.items()}

    def _role_from_scores(self, role_scores: dict[str, float]) -> tuple[str, float]:
        pairs = [(role, round(float(score), 6)) for role, score in role_scores.items()]
        pairs.sort(key=lambda item: item[1], reverse=True)
        best_role, best_score = pairs[0] if pairs else ("other", 0.0)
        second_score = pairs[1][1] if len(pairs) > 1 else 0.0
        confidence = round(max(0.0, best_score - second_score), 6)
        if confidence < self.min_confidence:
            best_role = "other"
        return best_role, confidence

    def _classification_text(
        self,
        *,
        question: str,
        span: SalientSpan,
        context: str,
        question_role: QuestionRole,
    ) -> str:
        return normalize_text(
            "Question: "
            + question
            + "\nAnswer role: "
            + question_role.answer_role
            + "\nAnswer target: "
            + question_role.answer_target
            + "\nSpan: "
            + span.text
            + "\nLocal context: "
            + context
            + "\nOriginal span: "
            + (span.original_text or span.text)
            + "\nDecide whether the span is a lookup clue, a constraint, an answer target, a format instruction, or other."
        )

    def _span_context(self, question: str, span: SalientSpan) -> str:
        start = max(0, int(span.start) - self.context_chars)
        end = min(len(question), int(span.end) + self.context_chars)
        return normalize_text(question[start:end])

    def _overlaps_question_role(self, span: SalientSpan, question_role: QuestionRole) -> bool:
        targets = [
            normalize_text(question_role.head_span).lower(),
            normalize_text(question_role.answer_target).lower(),
        ]
        span_key = normalize_text(span.text).lower()
        if not span_key:
            return False
        return any(target and (span_key in target or target in span_key) for target in targets)

    def _looks_like_url(self, text: str) -> bool:
        lowered = normalize_text(text).lower()
        return lowered.startswith(("http://", "https://", "www.")) or "://" in lowered

    def _looks_like_title(self, text: str) -> bool:
        cleaned = normalize_text(text)
        if len(cleaned.split()) < 2:
            return False
        title_tokens = [token for token in cleaned.split() if token[:1].isupper()]
        return len(title_tokens) >= 2

    def _entity_labels_for_spans(self, question: str, spans: list[SalientSpan]) -> dict[tuple[int, int], str]:
        doc = self._spacy_doc(question)
        if doc is None:
            return {}
        labels: dict[tuple[int, int], str] = {}
        for span in spans:
            best_label = ""
            best_overlap = 0
            for ent in doc.ents:
                overlap = max(0, min(int(span.end), int(ent.end_char)) - max(int(span.start), int(ent.start_char)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = str(ent.label_)
            if best_overlap > 0:
                labels[(span.start, span.end)] = best_label
        return labels

    def _spacy_doc(self, question: str) -> Any | None:
        if self._nlp is None and not self._nlp_load_attempted:
            self._nlp_load_attempted = True
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_md")
                except Exception:
                    self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        if self._nlp is None:
            return None
        try:
            return self._nlp(question)
        except Exception:
            return None


__all__ = ["ClassifiedSpan", "SpanRoleClassifier"]
