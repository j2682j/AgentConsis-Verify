from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class LabelerPreparedInput:
    """
    Store one direct corpus passage prepared for the EfficientRAG labeler.

    Args:
        - text: Exact labeler input text.
        - selected_passage: Original corpus passage text seen by the labeler.
        - diagnostics: Minimal input diagnostics for export/debugging.

    Returns:
        - LabelerPreparedInput: Labeler-ready passage input.
    """

    text: str
    selected_passage: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelerPreparedBatch:
    """
    Store direct corpus passages and the shared question context.

    Args:
        - question_context: Question/query text sent as the labeler question side.
        - documents: Corpus passages sent as the labeler passage side.

    Returns:
        - LabelerPreparedBatch: Batch input for EfficientRAGLabelerAdapter.
    """

    question_context: str
    documents: list[LabelerPreparedInput] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [document.text for document in self.documents]


class LabelerInputBuilder:
    """
    Convert retrieved corpus passages directly into labeler inputs.

    Args:
        - None.

    Returns:
        - LabelerInputBuilder: Direct corpus-to-labeler input builder.
    """

    def build_batch(
        self,
        *,
        question: str,
        current_query: str,
        documents: list[dict[str, Any]],
        intent_plan: Any | None = None,
    ) -> LabelerPreparedBatch:
        """
        Build labeler inputs without sentence selection or evidence-unit filtering.

        Args:
            - question: Original task question.
            - current_query: Retrieval query used for this round.
            - documents: Corpus passages returned by FAISS retrieval.
            - intent_plan: Unused; kept for caller compatibility.

        Returns:
            - LabelerPreparedBatch: Direct labeler batch.
        """
        del intent_plan
        question_context = self._question_context(
            question=question,
            current_query=current_query,
        )
        return LabelerPreparedBatch(
            question_context=question_context,
            documents=[self._build_document(document) for document in documents],
        )

    def _build_document(self, document: dict[str, Any]) -> LabelerPreparedInput:
        title = normalize_text(str(document.get("title", "") or ""))
        passage = normalize_text(str(document.get("text", "") or ""))
        parts: list[str] = []
        if title:
            parts.append(f"Source title: {title}")
        if passage:
            parts.append(f"Passage: {passage}")
        labeler_text = normalize_text("\n".join(parts)) or passage
        diagnostics = {
            "input_mode": "direct_corpus_passage",
            "labeler_input_text": labeler_text,
            "labeler_input_char_count": len(labeler_text),
            "selected_passage": passage,
            "source_title": title,
            "original_char_count": len(passage),
            "selected_char_count": len(passage),
            "selected_sentence_count": 0,
            "sentence_selection_used": False,
            "sentence_selection_truncated": False,
            "sentence_selection_reasons": [],
        }
        return LabelerPreparedInput(
            text=labeler_text,
            selected_passage=passage,
            diagnostics=diagnostics,
        )

    def _question_context(self, *, question: str, current_query: str) -> str:
        return normalize_text(
            "\n".join(
                [
                    f"Question: {normalize_text(question)}",
                    f"Search query: {normalize_text(current_query)}",
                ]
            )
        )


__all__ = [
    "LabelerInputBuilder",
    "LabelerPreparedBatch",
    "LabelerPreparedInput",
]
