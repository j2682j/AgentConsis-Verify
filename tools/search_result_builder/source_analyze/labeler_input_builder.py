from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text

from .evidence_unit_selector import EvidenceUnitSelector
from .sentence_selector import LabelerSentenceSelector


@dataclass(frozen=True)
class LabelerPreparedInput:
    """
    保存單一文件送入 Labeler 前的輸入與診斷資訊。

    Args:
        - text: 實際送進 Labeler 的 passage 文字。
        - selected_passage: sentence selection 後的純 passage。
        - diagnostics: 精簡的前處理診斷欄位。

    Returns:
        - LabelerPreparedInput: 單一 Labeler 文件輸入。
    """

    text: str
    selected_passage: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelerPreparedBatch:
    """
    保存一批文件的 Labeler question context 與 passage inputs。

    Args:
        - question_context: 批次共用的 Labeler question segment。
        - documents: 每個文件對應的 passage input。

    Returns:
        - LabelerPreparedBatch: 可直接送入 EfficientRAGLabelerAdapter。
    """

    question_context: str
    documents: list[LabelerPreparedInput] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [document.text for document in self.documents]


class LabelerInputBuilder:
    """
    將 retrieval 文件整理成結構化、短版的 Labeler input。

    Args:
        - sentence_selector: Labeler 前選句器。

    Returns:
        - LabelerInputBuilder: Structured labeler input builder。
    """

    def __init__(
        self,
        *,
        sentence_selector: LabelerSentenceSelector | None = None,
        evidence_unit_selector: EvidenceUnitSelector | None = None,
    ) -> None:
        self.sentence_selector = sentence_selector or LabelerSentenceSelector()
        self.evidence_unit_selector = evidence_unit_selector or EvidenceUnitSelector()

    def build_batch(
        self,
        *,
        question: str,
        current_query: str,
        documents: list[dict[str, Any]],
        intent_plan: Any | None = None,
    ) -> LabelerPreparedBatch:
        """
        建立一批可送入 Labeler 的 structured inputs。

        Args:
            - question: 原始問題。
            - current_query: 目前 retrieval query。
            - documents: Retriever 取回的 documents。
            - intent_plan: SearchIntentPlan 狀態。

        Returns:
            - LabelerPreparedBatch: question context 與每份文件的 passage。
        """
        answer_role = normalize_text(
            str(getattr(intent_plan, "answer_role", "") if intent_plan else "")
        ) or "unknown"
        constraints = self._constraints(intent_plan)
        question_context = self._question_context(
            question=question,
            current_query=current_query,
            answer_role=answer_role,
            constraints=constraints,
        )
        prepared_documents = [
            self._build_document(
                question=question,
                current_query=current_query,
                document=document,
                answer_role=answer_role,
                constraints=constraints,
            )
            for document in documents
        ]
        return LabelerPreparedBatch(
            question_context=question_context,
            documents=prepared_documents,
        )

    def _build_document(
        self,
        *,
        question: str,
        current_query: str,
        document: dict[str, Any],
        answer_role: str,
        constraints: list[str],
    ) -> LabelerPreparedInput:
        title = normalize_text(str(document.get("title", "") or ""))
        raw_text = str(document.get("text", "") or "")
        text = normalize_text(raw_text)
        selected = self.sentence_selector.select(
            question=question,
            query=current_query,
            text=text,
            source_title=title,
            answer_role=answer_role,
            constraints=constraints,
        )
        unit_selection = self.evidence_unit_selector.select(
            question=question,
            current_query=current_query,
            source_title=title,
            selected_passage=selected.text,
            raw_text=raw_text,
        )
        passage_text = unit_selection.text
        if (
            not passage_text
            and unit_selection.diagnostics.get("evidence_unit_should_fallback")
        ):
            passage_text = selected.text
        parts = []
        if title:
            parts.append(f"Source title: {title}")
        if passage_text:
            parts.append(f"Passage: {passage_text}")
        labeler_text = normalize_text("\n".join(parts)) or text
        diagnostics = {
            "input_mode": "structured_sentence_selection",
            "answer_role": answer_role,
            "constraint_count": len(constraints),
            "labeler_input_text": labeler_text,
            "labeler_input_char_count": len(labeler_text),
            "selected_sentence_count": selected.selected_count,
            "original_char_count": selected.original_char_count,
            "selected_char_count": selected.selected_char_count,
            "sentence_selection_used": True,
            "sentence_selection_truncated": selected.truncated,
            "sentence_selection_reasons": list(selected.reasons),
            "selected_passage": selected.text,
            "evidence_unit_passage": unit_selection.text,
            "source_title": title,
            **unit_selection.diagnostics,
        }
        return LabelerPreparedInput(
            text=labeler_text,
            selected_passage=selected.text,
            diagnostics=diagnostics,
        )

    def _question_context(
        self,
        *,
        question: str,
        current_query: str,
        answer_role: str,
        constraints: list[str],
    ) -> str:
        lines = [
            f"Question: {normalize_text(question)}",
            f"Search query: {normalize_text(current_query)}",
            f"Answer role: {answer_role or 'unknown'}",
        ]
        if constraints:
            lines.append(f"Required constraints: {', '.join(constraints[:8])}")
        return normalize_text("\n".join(lines))

    def _constraints(self, intent_plan: Any | None) -> list[str]:
        if intent_plan is None:
            return []
        terms: list[str] = []
        target = normalize_text(str(getattr(intent_plan, "target", "") or ""))
        if target:
            terms.append(target)
        for field_name in ("must_include", "missing_terms", "completed_terms"):
            for item in list(getattr(intent_plan, field_name, []) or []):
                text = normalize_text(str(item or ""))
                if text and not text.startswith("answer_candidate:"):
                    terms.append(text)
        return self._dedupe(terms)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = normalize_text(value)
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result


__all__ = [
    "LabelerInputBuilder",
    "LabelerPreparedBatch",
    "LabelerPreparedInput",
]
