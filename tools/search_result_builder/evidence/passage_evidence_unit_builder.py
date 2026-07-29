from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class PassageEvidenceUnit:
    """Represent one grounded sentence-level unit selected from a passage."""

    document_index: int
    document_id: str
    text: str
    local_context: str
    source_title: str = ""
    source_type: str = "web"
    semantic_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassageEvidenceUnitResult:
    """Store passage units prepared without EfficientRAG token labels."""

    units: list[PassageEvidenceUnit] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units],
            "diagnostics": dict(self.diagnostics),
        }


class PassageEvidenceUnitBuilder:
    """
    Convert retrieved passages into bounded semantic units for role classification.

    The builder performs structural sentence segmentation, then uses the retriever's
    already-loaded embedding model to retain units closest to the task. It does not
    predict CONTINUE/FINISH tags and does not create token-level labels.
    """

    _BOUNDARY_RE = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-Z0-9\"'])")
    _CLAUSE_RE = re.compile(r"\s*(?:;|\s[|]\s|\s[-–:]\s)\s*")

    def __init__(
        self,
        *,
        max_units: int = 10,
        max_units_per_document: int = 6,
        max_unit_chars: int = 160,
        max_context_chars: int = 600,
    ) -> None:
        self.max_units = max(1, max_units)
        self.max_units_per_document = max(1, max_units_per_document)
        self.max_unit_chars = max(60, max_unit_chars)
        self.max_context_chars = max(160, max_context_chars)

    def build(
        self,
        *,
        question: str,
        documents: Sequence[Mapping[str, Any]],
        embedder: Any | None = None,
    ) -> PassageEvidenceUnitResult:
        candidates: list[PassageEvidenceUnit] = []
        per_document_counts: dict[int, int] = {}
        for document_index, document in enumerate(documents):
            text = normalize_text(str(document.get("text", "") or ""))
            if not text:
                continue
            title = normalize_text(str(document.get("title", "") or ""))
            document_id = normalize_text(
                str(document.get("id", document.get("document_id", "")) or "")
            )
            source_type = normalize_text(
                str(document.get("record_type", "") or "")
            ) or "web"
            for unit_text in self._units(document, text):
                candidates.append(
                    PassageEvidenceUnit(
                        document_index=document_index,
                        document_id=document_id,
                        text=unit_text,
                        local_context=self._local_context(text, unit_text),
                        source_title=title,
                        source_type=source_type,
                    )
                )

        ranked, ranking_method = self._rank(
            question=question,
            candidates=candidates,
            embedder=embedder,
        )
        selected: list[PassageEvidenceUnit] = []
        seen: set[tuple[int, str]] = set()
        for candidate in ranked:
            key = (candidate.document_index, candidate.text.casefold())
            if key in seen:
                continue
            count = per_document_counts.get(candidate.document_index, 0)
            if count >= self.max_units_per_document:
                continue
            selected.append(candidate)
            seen.add(key)
            per_document_counts[candidate.document_index] = count + 1
            if len(selected) >= self.max_units:
                break

        return PassageEvidenceUnitResult(
            units=selected,
            diagnostics={
                "mode": "labeler_bypass",
                "ranking_method": ranking_method,
                "document_count": len(documents),
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "max_units": self.max_units,
                "max_units_per_document": self.max_units_per_document,
                "selected_units": [unit.to_dict() for unit in selected],
            },
        )

    def _units(self, document: Mapping[str, Any], text: str) -> list[str]:
        record_type = normalize_text(str(document.get("record_type", "") or ""))
        units: list[str] = []
        if record_type and record_type != "passage":
            for value in self._record_values(document):
                units.extend(self._fit_unit(value))
        for sentence in self._BOUNDARY_RE.split(text):
            units.extend(self._fit_unit(sentence))
        return self._dedupe(units)

    def _fit_unit(self, value: str) -> list[str]:
        text = normalize_text(value)
        if not text:
            return []
        if len(text) <= self.max_unit_chars:
            return [text]
        clauses = [normalize_text(item) for item in self._CLAUSE_RE.split(text)]
        clauses = [item for item in clauses if item]
        if len(clauses) > 1 and all(len(item) <= self.max_unit_chars for item in clauses):
            return clauses
        return self._word_windows(text)

    def _word_windows(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        output: list[str] = []
        current: list[str] = []
        for word in words:
            proposed = " ".join([*current, word])
            if current and len(proposed) > self.max_unit_chars:
                output.append(" ".join(current))
                current = current[-4:]
            current.append(word)
        if current:
            output.append(" ".join(current))
        return [normalize_text(item) for item in output if normalize_text(item)]

    def _rank(
        self,
        *,
        question: str,
        candidates: list[PassageEvidenceUnit],
        embedder: Any | None,
    ) -> tuple[list[PassageEvidenceUnit], str]:
        if not candidates or embedder is None:
            return candidates, "retrieval_order"
        try:
            query_text = (
                embedder.prepare_query_text(question)
                if hasattr(embedder, "prepare_query_text")
                else question
            )
            passage_texts = [
                (
                    embedder.prepare_passage_text(candidate.text)
                    if hasattr(embedder, "prepare_passage_text")
                    else candidate.text
                )
                for candidate in candidates
            ]
            vectors = np.asarray(
                embedder.embed([query_text, *passage_texts]),
                dtype=np.float32,
            )
            if vectors.ndim != 2 or vectors.shape[0] != len(candidates) + 1:
                return candidates, "retrieval_order"
            query_vector = vectors[0]
            passage_vectors = vectors[1:]
            query_norm = max(float(np.linalg.norm(query_vector)), 1e-12)
            passage_norms = np.maximum(
                np.linalg.norm(passage_vectors, axis=1),
                1e-12,
            )
            similarities = (passage_vectors @ query_vector) / (
                passage_norms * query_norm
            )
            ranked = [
                PassageEvidenceUnit(
                    **{
                        **candidate.to_dict(),
                        "semantic_similarity": float(similarity),
                    }
                )
                for candidate, similarity in zip(candidates, similarities, strict=False)
            ]
            ranked.sort(
                key=lambda item: (
                    -item.semantic_similarity,
                    item.document_index,
                )
            )
            return ranked, "e5_semantic_similarity"
        except Exception:
            return candidates, "retrieval_order"

    def _local_context(self, text: str, unit: str) -> str:
        index = text.casefold().find(unit.casefold())
        if index < 0:
            return text[: self.max_context_chars]
        half = self.max_context_chars // 2
        start = max(0, index - half)
        end = min(len(text), index + len(unit) + half)
        return normalize_text(text[start:end])

    @staticmethod
    def _record_values(document: Mapping[str, Any]) -> Iterable[str]:
        fields = document.get("record_fields")
        if not isinstance(fields, Mapping):
            return []
        for key, value in fields.items():
            if key in {
                "id",
                "text",
                "title",
                "url",
                "parent_url",
                "content_url",
            }:
                continue
            if isinstance(value, (str, int, float)):
                text = normalize_text(str(value))
                if text:
                    yield f"{key}: {text}"
            elif isinstance(value, (list, tuple, set)):
                text = normalize_text(", ".join(str(item) for item in value if item))
                if text:
                    yield f"{key}: {text}"

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(value)
            key = text.casefold()
            if not text or key in seen:
                continue
            output.append(text)
            seen.add(key)
        return output


__all__ = [
    "PassageEvidenceUnit",
    "PassageEvidenceUnitBuilder",
    "PassageEvidenceUnitResult",
]
