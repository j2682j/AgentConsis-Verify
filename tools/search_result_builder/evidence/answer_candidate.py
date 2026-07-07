from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable

from utils.network_utils import normalize_for_exact, normalize_text, semantic_similarity_score


_QA_PIPELINE = None


@dataclass
class AnswerCandidate:
    """
    Store one evidence-grounded answer candidate and the three scores used to rank it.
    """

    text: str
    source_id: str = ""
    evidence_id: str = ""
    title: str = ""
    context: str = ""
    answer_type: str = "unknown"
    score: float = 0.0
    method: str = "extractive_qa"
    qa_span_score: float = 0.0
    question_context_relevance: float = 0.0
    answer_type_compatibility: float = 0.0
    question_answer_type: str = "unknown"
    candidate_answer_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceAnswerExtractor:
    """
    Extract short answer candidates from selected evidence and rerank them.

    The reranker intentionally uses only:
    1. extractive QA span score,
    2. question-context semantic relevance,
    3. embedding-based answer type compatibility.
    """

    TYPE_DESCRIPTIONS = {
        "number": "the expected answer is a number, score, count, amount, or measurement",
        "person": "the expected answer is a person's name",
        "organization": "the expected answer is an organization, institution, company, or team",
        "location": "the expected answer is a country, city, region, address, or place",
        "date": "the expected answer is a date, year, month, day, or time",
        "title": "the expected answer is a title of a book, article, movie, song, paper, artwork, or episode",
        "boolean": "the expected answer is yes or no",
        "list": "the expected answer is a list of names, values, or items",
        "short_phrase": "the expected answer is a short word or phrase",
    }

    COMPATIBLE_TYPES = {
        ("number", "date"),
        ("date", "number"),
        ("title", "short_phrase"),
        ("short_phrase", "title"),
        ("organization", "short_phrase"),
        ("person", "short_phrase"),
        ("location", "short_phrase"),
    }

    def __init__(
        self,
        *,
        model_name: str = "deepset/minilm-uncased-squad2",
        device: int = -1,
        max_evidence_items: int = 5,
        top_k_per_evidence: int = 3,
        max_candidates: int = 5,
        qa_pipeline: Callable[..., Any] | None = None,
        similarity_fn: Callable[[str, str], float | None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_evidence_items = max(1, max_evidence_items)
        self.top_k_per_evidence = max(1, top_k_per_evidence)
        self.max_candidates = max(1, max_candidates)
        self.qa_pipeline = qa_pipeline
        self.similarity_fn = similarity_fn or semantic_similarity_score

    def extract(
        self,
        *,
        question: str,
        evidence_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        question = normalize_text(question)
        if not question or not evidence_items:
            return []

        pipeline = self.qa_pipeline or self._load_qa_pipeline()
        if pipeline is None:
            return []

        question_type = self._infer_answer_type(question)
        candidates: list[AnswerCandidate] = []
        for item in evidence_items[: self.max_evidence_items]:
            context = normalize_text(item.get("text", ""))
            if not context:
                continue
            for result in self._run_qa(pipeline, question=question, context=context):
                text = self._clean_answer_text(result.get("answer", ""))
                if not self._is_candidate_text(text):
                    continue
                local_context = self._context_window(context, text)
                qa_span_score = self._clamp01(result.get("score", 0.0))
                relevance = self._semantic_relevance(question, local_context or context)
                candidate_type = self._infer_answer_type(
                    " ".join(part for part in [text, local_context] if part)
                )
                compatibility = self._answer_type_compatibility(
                    question_type,
                    candidate_type,
                )
                score = self._candidate_score(
                    qa_span_score=qa_span_score,
                    question_context_relevance=relevance,
                    answer_type_compatibility=compatibility,
                )
                candidates.append(
                    AnswerCandidate(
                        text=text,
                        source_id=str(item.get("source_id", "") or ""),
                        evidence_id=str(item.get("evidence_id", "") or ""),
                        title=str(item.get("title", "") or ""),
                        context=local_context,
                        answer_type=candidate_type,
                        score=score,
                        qa_span_score=qa_span_score,
                        question_context_relevance=relevance,
                        answer_type_compatibility=compatibility,
                        question_answer_type=question_type,
                        candidate_answer_type=candidate_type,
                    )
                )

        return [
            candidate.to_dict()
            for candidate in self._dedupe_and_rank(candidates)[: self.max_candidates]
        ]

    def _load_qa_pipeline(self):
        global _QA_PIPELINE
        if _QA_PIPELINE is not None:
            return _QA_PIPELINE
        try:
            from transformers import pipeline

            _QA_PIPELINE = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
            )
        except Exception:
            _QA_PIPELINE = None
        return _QA_PIPELINE

    def _run_qa(self, pipeline: Callable[..., Any], *, question: str, context: str) -> list[dict[str, Any]]:
        payload = {"question": question, "context": context}
        try:
            result = pipeline(
                payload,
                top_k=self.top_k_per_evidence,
                handle_impossible_answer=True,
            )
        except TypeError:
            result = pipeline(payload, top_k=self.top_k_per_evidence)
        except Exception:
            return []
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []

    def _candidate_score(
        self,
        *,
        qa_span_score: float,
        question_context_relevance: float,
        answer_type_compatibility: float,
    ) -> float:
        return round(
            (0.4375 * qa_span_score)
            + (0.3125 * question_context_relevance)
            + (0.25 * answer_type_compatibility),
            6,
        )

    def _semantic_relevance(self, question: str, context: str) -> float:
        if not question or not context:
            return 0.0
        try:
            score = self.similarity_fn(question, context)
        except Exception:
            return 0.0
        if score is None:
            return 0.0
        return self._clamp01((float(score) + 1.0) / 2.0)

    def _infer_answer_type(self, text: str) -> str:
        text = normalize_text(text)
        if not text:
            return "unknown"

        best_type = "short_phrase"
        best_score = -1.0
        for answer_type, description in self.TYPE_DESCRIPTIONS.items():
            try:
                score = self.similarity_fn(text, description)
            except Exception:
                score = None
            if score is None:
                continue
            if float(score) > best_score:
                best_type = answer_type
                best_score = float(score)
        return best_type

    def _answer_type_compatibility(self, question_type: str, candidate_type: str) -> float:
        if not question_type or not candidate_type:
            return 0.0
        if question_type == candidate_type:
            return 1.0
        if (question_type, candidate_type) in self.COMPATIBLE_TYPES:
            return 0.75
        if "short_phrase" in {question_type, candidate_type}:
            return 0.55
        return 0.2

    def _dedupe_and_rank(self, candidates: list[AnswerCandidate]) -> list[AnswerCandidate]:
        best_by_text: dict[str, AnswerCandidate] = {}
        for candidate in candidates:
            key = normalize_for_exact(candidate.text)
            if not key:
                continue
            current = best_by_text.get(key)
            if current is None or candidate.score > current.score:
                best_by_text[key] = candidate
        return sorted(
            best_by_text.values(),
            key=lambda item: (
                item.score,
                item.qa_span_score,
                item.question_context_relevance,
            ),
            reverse=True,
        )

    def _context_window(self, context: str, answer: str, *, max_chars: int = 320) -> str:
        if not answer:
            return self._truncate(context, max_chars=max_chars)
        match = re.search(re.escape(answer), context, flags=re.IGNORECASE)
        if not match:
            return self._truncate(context, max_chars=max_chars)
        start = max(0, match.start() - 140)
        end = min(len(context), match.end() + 140)
        left_boundary = context.rfind(".", 0, match.start())
        if left_boundary >= 0 and match.start() - left_boundary < 160:
            start = left_boundary + 1
        right_boundary = context.find(".", match.end())
        if right_boundary >= 0 and right_boundary - match.end() < 160:
            end = right_boundary + 1
        return self._truncate(context[start:end].strip(), max_chars=max_chars)

    def _clean_answer_text(self, text: Any) -> str:
        cleaned = normalize_text(text)
        cleaned = cleaned.strip(" \t\r\n\"'`*")
        cleaned = re.sub(r"^(answer|final answer|final_answer)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _is_candidate_text(self, text: str) -> bool:
        if not text:
            return False
        if len(text) > 120:
            return False
        if len(text.split()) > 20:
            return False
        if text.casefold() in {"unknown", "none", "n/a", "not enough information"}:
            return False
        return True

    def _truncate(self, text: str, *, max_chars: int) -> str:
        text = normalize_text(text)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    def _clamp01(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))


__all__ = ["AnswerCandidate", "EvidenceAnswerExtractor"]
