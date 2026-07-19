from __future__ import annotations

from collections.abc import Iterable
import hashlib
import re
from typing import Any

from utils.network_utils import normalize_for_exact, normalize_text

from .fact_store import TaskFactStore
from .models import (
    EvidenceFact,
    FactEvidenceRef,
    SemanticExtractionResult,
    SemanticSourceUnit,
    StructuredRelationRecord,
)
from .semantic_fact_extractor import SemanticFactExtractor
from .context_assembler import CrossContextAssembler
from .cross_context_fact_extractor import CrossContextFactExtractor
from .completeness_contract import CompletenessContractBuilder
from .set_difference_deriver import SetDifferenceFactDeriver
from .gift_assignment_deriver import GiftAssignmentFactDeriver


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
        cross_context_assembler: CrossContextAssembler | None = None,
        cross_context_extractor: CrossContextFactExtractor | None = None,
        completeness_contract_builder: CompletenessContractBuilder | None = None,
        set_difference_deriver: SetDifferenceFactDeriver | None = None,
        gift_assignment_deriver: GiftAssignmentFactDeriver | None = None,
        max_semantic_units: int = 8,
    ) -> None:
        self.semantic_extractor = semantic_extractor or SemanticFactExtractor(
            max_units_per_call=max_semantic_units
        )
        self.cross_context_assembler = (
            cross_context_assembler or CrossContextAssembler(max_windows=6)
        )
        self.cross_context_extractor = (
            cross_context_extractor
            or CrossContextFactExtractor(semantic_extractor=self.semantic_extractor)
        )
        self.completeness_contract_builder = (
            completeness_contract_builder or CompletenessContractBuilder()
        )
        self.set_difference_deriver = (
            set_difference_deriver or SetDifferenceFactDeriver()
        )
        self.gift_assignment_deriver = gift_assignment_deriver or GiftAssignmentFactDeriver()
        self.max_semantic_units = max(1, int(max_semantic_units))

    def extract(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        parsed_payload: dict[str, Any],
    ) -> SemanticExtractionResult:
        store = TaskFactStore()
        completeness_contract = (
            self.completeness_contract_builder.from_attachment_payload(parsed_payload)
        )
        store.add_completeness_contract(completeness_contract)
        structured_records = self._structured_relation_records(parsed_payload)
        structured = self._structured_relation_facts(parsed_payload, structured_records)
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
        windows = self.cross_context_assembler.assemble(
            units,
            anchor_unit_ids=[unit.unit_id for unit in units],
        )
        cross_context = self.cross_context_extractor.extract_windows(
            question=question,
            answer_requirement=answer_requirement,
            answer_target=answer_requirement,
            current_goal=answer_requirement,
            windows=windows,
        )
        store.extend(cross_context.facts)
        set_difference = self._derive_set_difference(
            parsed_payload=parsed_payload,
            completeness_contract=completeness_contract,
            answer_requirement=answer_requirement or question,
        )
        if set_difference is not None:
            derivation, derived_facts = set_difference
            store.add_set_difference_derivation(derivation)
            store.extend(derived_facts)
        provenance = dict(parsed_payload.get("provenance") or {})
        gift_facts, gift_diagnostics = self.gift_assignment_deriver.derive(
            question=question,
            parsed_payload=parsed_payload,
            base_facts=store.all(),
            source_id=str(provenance.get("file_path") or provenance.get("source") or "attachment"),
            source_type=str(provenance.get("file_type") or "attachment"),
        )
        if gift_facts:
            gift_contract = self.completeness_contract_builder.build(
                scope_id="gift-assignment",
                source_id=str(provenance.get("file_path") or "attachment"),
                source_type=str(provenance.get("file_type") or "attachment"),
                expected_units=12,
                processed_units=12,
                complete=True,
                completion_reason="all_assignments_profiles_and_gifts_processed",
                unit_ids=[f"gift-scope-{index}" for index in range(12)],
            )
            store.add_completeness_contract(gift_contract)
            gift_facts = [
                EvidenceFact.from_dict({
                    **fact.to_dict(),
                    "qualifiers": {
                        **fact.qualifiers,
                        "completeness_contract_ids": gift_contract.contract_id,
                    },
                })
                for fact in gift_facts
            ]
            store.extend(gift_facts)
        facts = store.all()
        diagnostics = {
            **semantic.diagnostics,
            "structured_fact_count": len(structured),
            "structured_relation_record_count": len(structured_records),
            "structured_relation_records": [item.to_dict() for item in structured_records],
            "semantic_unit_count": len(units),
            "stored_fact_count": len(facts),
            "cross_context": cross_context.diagnostics,
            "cross_context_window_count": len(windows),
            "cross_context_fact_count": len(cross_context.facts),
            "completeness_contracts": [
                item.to_dict() for item in store.completeness_contracts()
            ],
            "absence_checks": [],
            "set_difference_derivations": [
                item.to_dict() for item in store.set_difference_derivations()
            ],
            "set_difference_fact_count": sum(
                fact.derivation_type == "set_difference" for fact in facts
            ),
            "gift_assignment_derivation": gift_diagnostics,
        }
        return SemanticExtractionResult(
            facts=facts,
            rejected_items=[
                *semantic.rejected_items,
                *cross_context.rejected_items,
            ],
            diagnostics=diagnostics,
        )

    def _derive_set_difference(
        self,
        *,
        parsed_payload: dict[str, Any],
        completeness_contract: Any,
        answer_requirement: str,
    ):
        native = dict(parsed_payload.get("native_metadata") or {})
        spec = parsed_payload.get("set_difference_inputs")
        if not isinstance(spec, dict):
            spec = native.get("set_difference_inputs")
        if not isinstance(spec, dict):
            return None
        universe = list(spec.get("universe_values") or [])
        observed = list(spec.get("observed_values") or [])
        if not universe:
            return None
        provenance = dict(parsed_payload.get("provenance") or {})
        return self.set_difference_deriver.derive(
            universe_values=[str(item) for item in universe],
            observed_values=[str(item) for item in observed],
            completeness_contracts=[completeness_contract],
            universe_fact_ids=[
                str(item) for item in list(spec.get("universe_fact_ids") or [])
            ],
            observed_fact_ids=[
                str(item) for item in list(spec.get("observed_fact_ids") or [])
            ],
            negative_relation=str(spec.get("negative_relation") or "is absent from"),
            negative_object=str(spec.get("negative_object") or "observed set"),
            answer_subject=str(spec.get("answer_subject") or "missing value"),
            answer_relation=str(spec.get("answer_relation") or "is"),
            answer_requirement=answer_requirement,
            goal_id=str(spec.get("goal_id") or ""),
            source_id=str(
                provenance.get("file_path")
                or provenance.get("source")
                or "attachment"
            ),
            source_type=str(provenance.get("file_type") or "attachment"),
        )

    def _structured_relation_facts(
        self,
        payload: dict[str, Any],
        records: list[StructuredRelationRecord] | None = None,
    ) -> list[EvidenceFact]:
        provenance = dict(payload.get("provenance") or {})
        source_id = str(
            provenance.get("file_path")
            or provenance.get("source")
            or "attachment"
        )
        source_type = str(provenance.get("file_type") or "attachment")
        facts: list[EvidenceFact] = []
        for record in records or []:
            columns = list(record.fields)
            if len(columns) < 2:
                continue
            subject = record.fields[columns[0]]
            for column in columns[1:]:
                object_value = record.fields[column]
                if not subject or not object_value:
                    continue
                context = " | ".join(f"{key}: {value}" for key, value in record.fields.items())
                facts.append(
                    EvidenceFact(
                        fact_id=self._fact_id(record.source_id, record.row_id, column, object_value),
                        subject=subject,
                        relation=column,
                        object=object_value,
                        role="BRIDGE",
                        evidence_spans=[context],
                        evidence_refs=list(record.provenance),
                        context=context,
                        source_id=record.source_id,
                        source_type=record.source_type,
                        grounding_status="grounded",
                        extraction_method="structured_relation_record",
                    )
                )
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

    def _structured_relation_records(
        self,
        payload: dict[str, Any],
    ) -> list[StructuredRelationRecord]:
        provenance = dict(payload.get("provenance") or {})
        source_id = str(provenance.get("file_path") or provenance.get("source") or "attachment")
        source_type = str(provenance.get("file_type") or "attachment")
        records: list[StructuredRelationRecord] = []
        for table_index, table in enumerate(list(payload.get("tables") or []), start=1):
            if not isinstance(table, dict):
                continue
            columns = [normalize_text(str(item)) or f"column_{index + 1}" for index, item in enumerate(list(table.get("columns") or []))]
            structure_id = normalize_text(str(table.get("name") or f"table_{table_index}"))
            for row_index, row in enumerate(list(table.get("rows") or []), start=1):
                if not isinstance(row, list):
                    continue
                fields = {
                    (columns[index] if index < len(columns) else f"column_{index + 1}"): normalize_text(str(value))
                    for index, value in enumerate(row)
                    if normalize_text(str(value))
                }
                if not fields:
                    continue
                row_id = f"TABLE{table_index}-R{row_index}"
                text = " | ".join(f"{key}: {value}" for key, value in fields.items())
                records.append(
                    StructuredRelationRecord(
                        record_id=self._fact_id(source_id, structure_id, row_id),
                        source_id=source_id,
                        source_type=source_type,
                        structure_id=structure_id,
                        row_id=row_id,
                        fields=fields,
                        normalized_fields={key: normalize_for_exact(value) for key, value in fields.items()},
                        provenance=[FactEvidenceRef(source_id=source_id, unit_id=row_id, text=text, document_id=structure_id)],
                    )
                )
        return records

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

        order = 0
        for index, block in enumerate(list(parsed_payload.get("text_blocks") or []), start=1):
            if not isinstance(block, dict):
                continue
            text = normalize_text(str(block.get("text") or ""))
            if not text:
                continue
            page = block.get("page")
            section = normalize_text(str(block.get("section") or ""))
            order += 1
            unit = SemanticSourceUnit(
                unit_id=f"T{index}",
                text=text,
                source_id=base_source_id,
                source_type=source_type,
                source_title=section or (f"Page {page}" if page is not None else "Attachment text"),
                metadata={
                    "page": page,
                    "section": section,
                    "order": order,
                    "document_id": f"T{index}",
                    "record_type": normalize_text(str(block.get("block_type") or "text")),
                },
            )
            candidates.append((self._overlap(question_terms, text), index, unit))

        for table_index, table in enumerate(list(parsed_payload.get("tables") or []), start=1):
            if not isinstance(table, dict):
                continue
            columns = [normalize_text(str(item)) for item in list(table.get("columns") or [])]
            table_id = f"TABLE{table_index}"
            if columns:
                order += 1
                header_text = "Columns: " + " | ".join(columns)
                header = SemanticSourceUnit(
                    unit_id=f"{table_id}-H",
                    text=header_text,
                    source_id=base_source_id,
                    source_type=source_type,
                    source_title=normalize_text(str(table.get("name") or "Table")),
                    metadata={
                        "order": order,
                        "document_id": f"{table_id}-H",
                        "record_type": "table",
                        "table_id": table_id,
                        "section": normalize_text(str(table.get("name") or "")),
                    },
                )
                candidates.append((self._overlap(question_terms, header_text) + 1, order, header))
            for row_index, row in enumerate(list(table.get("rows") or []), start=1):
                if not isinstance(row, list):
                    continue
                values = [normalize_text(str(item)) for item in row]
                pairs = [
                    f"{columns[index]}: {value}"
                    for index, value in enumerate(values)
                    if value and index < len(columns)
                ]
                row_text = " | ".join(pairs or values)
                if not row_text:
                    continue
                order += 1
                row_unit = SemanticSourceUnit(
                    unit_id=f"{table_id}-R{row_index}",
                    text=row_text,
                    source_id=base_source_id,
                    source_type=source_type,
                    source_title=normalize_text(str(table.get("name") or "Table")),
                    metadata={
                        "order": order,
                        "document_id": f"{table_id}-R{row_index}",
                        "record_type": "table_row",
                        "table_id": table_id,
                        "row_index": row_index,
                    },
                )
                candidates.append((self._overlap(question_terms, row_text) + 1, order, row_unit))

        offset = len(candidates)
        for index, block in enumerate(list(parsed_payload.get("visual_blocks") or []), start=1):
            if not isinstance(block, dict):
                continue
            text = normalize_text(str(block.get("text") or ""))
            if not text:
                continue
            attributes = dict(block.get("attributes") or {})
            order += 1
            unit = SemanticSourceUnit(
                unit_id=f"V{index}",
                text=text,
                source_id=base_source_id,
                source_type="visual_observation",
                source_title=normalize_text(str(block.get("region") or "Visual observation")),
                metadata={
                    **attributes,
                    "order": order,
                    "document_id": f"V{index}",
                    "record_type": "visual_observation",
                },
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
