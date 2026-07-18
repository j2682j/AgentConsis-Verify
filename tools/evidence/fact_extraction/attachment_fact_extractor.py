from __future__ import annotations

from collections.abc import Iterable
import hashlib
import re
from typing import Any

from utils.network_utils import normalize_text

from .fact_store import TaskFactStore
from .models import EvidenceFact, SemanticExtractionResult, SemanticSourceUnit
from .semantic_fact_extractor import SemanticFactExtractor


class AttachmentFactExtractor:
    """
    將附件 payload 中的非結構化文字與視覺描述轉為可回溯事實。

    Args:
     - semantic_extractor: 處理 PDF、文件文字與視覺描述的語意抽取器。
     - max_semantic_units: 單一附件最多送入模型的短來源單位數。

    Returns:
     - AttachmentFactExtractor: 合併原生 relation 與語意事實的附件抽取器。
    """

    _TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

    def __init__(
        self,
        *,
        semantic_extractor: SemanticFactExtractor | None = None,
        max_semantic_units: int = 8,
    ) -> None:
        self.semantic_extractor = semantic_extractor or SemanticFactExtractor(
            max_units_per_call=max_semantic_units
        )
        self.max_semantic_units = max(1, int(max_semantic_units))

    def extract(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        parsed_payload: dict[str, Any],
    ) -> SemanticExtractionResult:
        store = TaskFactStore()
        structured = self._structured_relation_facts(parsed_payload)
        store.extend(structured)

        units = self._semantic_units(
            question=question,
            parsed_payload=parsed_payload,
        )
        semantic = self.semantic_extractor.extract_batch(
            question=question,
            answer_requirement=answer_requirement,
            current_goal=answer_requirement,
            units=units,
        )
        store.extend(semantic.facts)
        facts = store.all()
        diagnostics = {
            **semantic.diagnostics,
            "structured_fact_count": len(structured),
            "semantic_unit_count": len(units),
            "stored_fact_count": len(facts),
        }
        return SemanticExtractionResult(
            facts=facts,
            rejected_items=list(semantic.rejected_items),
            diagnostics=diagnostics,
        )

    def _structured_relation_facts(
        self,
        payload: dict[str, Any],
    ) -> list[EvidenceFact]:
        provenance = dict(payload.get("provenance") or {})
        source_id = str(
            provenance.get("file_path")
            or provenance.get("source")
            or "attachment"
        )
        source_type = str(provenance.get("file_type") or "attachment")
        facts: list[EvidenceFact] = []
        for relation in list(payload.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            subject = normalize_text(str(relation.get("source") or ""))
            predicate = normalize_text(str(relation.get("relation") or ""))
            object_value = normalize_text(str(relation.get("target") or ""))
            if not subject or not predicate or not object_value:
                continue
            context = f"{subject} {predicate} {object_value}"
            facts.append(
                EvidenceFact(
                    fact_id=self._fact_id(source_id, subject, predicate, object_value),
                    subject=subject,
                    relation=predicate,
                    object=object_value,
                    role="CONTEXT",
                    evidence_spans=[context],
                    context=context,
                    source_id=source_id,
                    source_type=source_type,
                    grounding_status="grounded",
                    extraction_method="structured_attachment_relation",
                )
            )
        return facts

    def _semantic_units(
        self,
        *,
        question: str,
        parsed_payload: dict[str, Any],
    ) -> list[SemanticSourceUnit]:
        provenance = dict(parsed_payload.get("provenance") or {})
        base_source_id = str(
            provenance.get("file_path")
            or provenance.get("source")
            or "attachment"
        )
        source_type = str(provenance.get("file_type") or "attachment")
        candidates: list[tuple[int, int, SemanticSourceUnit]] = []
        question_terms = self._terms(question)

        for index, block in enumerate(list(parsed_payload.get("text_blocks") or []), start=1):
            if not isinstance(block, dict):
                continue
            text = normalize_text(str(block.get("text") or ""))
            if not text:
                continue
            page = block.get("page")
            section = normalize_text(str(block.get("section") or ""))
            unit = SemanticSourceUnit(
                unit_id=f"T{index}",
                text=text,
                source_id=f"{base_source_id}#text-{index}",
                source_type=source_type,
                source_title=section or (f"Page {page}" if page is not None else "Attachment text"),
                metadata={"page": page, "section": section},
            )
            candidates.append((self._overlap(question_terms, text), index, unit))

        offset = len(candidates)
        for index, block in enumerate(list(parsed_payload.get("visual_blocks") or []), start=1):
            if not isinstance(block, dict):
                continue
            text = normalize_text(str(block.get("text") or ""))
            if not text:
                continue
            attributes = dict(block.get("attributes") or {})
            unit = SemanticSourceUnit(
                unit_id=f"V{index}",
                text=text,
                source_id=f"{base_source_id}#visual-{index}",
                source_type="visual_observation",
                source_title=normalize_text(str(block.get("region") or "Visual observation")),
                metadata=attributes,
            )
            candidates.append((self._overlap(question_terms, text) + 1, offset + index, unit))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in candidates[: self.max_semantic_units]]

    def _terms(self, value: str) -> set[str]:
        return {
            match.group(0).casefold()
            for match in self._TERM_RE.finditer(normalize_text(value))
            if len(match.group(0)) > 2
        }

    def _overlap(self, question_terms: set[str], value: str) -> int:
        if not question_terms:
            return 0
        return len(question_terms & self._terms(value))

    @staticmethod
    def _fact_id(source_id: str, *parts: str) -> str:
        raw = "|".join([source_id, *parts])
        return "F-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def render_attachment_facts(facts: Iterable[EvidenceFact], *, max_items: int = 8) -> str:
    lines: list[str] = []
    for fact in list(facts)[: max(1, int(max_items))]:
        if fact.grounding_status != "grounded":
            continue
        qualifier_text = ", ".join(
            f"{key}={value}" for key, value in fact.qualifiers.items()
        )
        statement = f"{fact.subject} --{fact.relation}--> {fact.object}"
        if fact.polarity == "negative":
            statement = "NOT: " + statement
        if qualifier_text:
            statement += f" ({qualifier_text})"
        lines.append(f"- {statement}")
    if not lines:
        return ""
    return "Attachment Facts:\n" + "\n".join(lines)


__all__ = ["AttachmentFactExtractor", "render_attachment_facts"]
