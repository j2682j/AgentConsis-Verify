from __future__ import annotations

from dataclasses import replace
import re

from utils.network_utils import normalize_text

from .models import (
    EvidenceFact,
    FactEvidenceRef,
    SemanticSourceUnit,
    VALID_FACT_ROLES,
    VALID_POLARITIES,
)


class FactGroundingValidator:
    """
    確認模型抽取的事實能回到原始來源，而不是模型自行補出的敘述。

    Args:
     - max_evidence_spans: 每筆事實允許綁定的來源片段上限。

    Returns:
     - FactGroundingValidator: 回傳 grounded、ambiguous 或 invalid 的事實。
    """

    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

    def __init__(self, *, max_evidence_spans: int = 2) -> None:
        self.max_evidence_spans = max(1, int(max_evidence_spans))

    def validate(self, fact: EvidenceFact, *, source_text: str) -> EvidenceFact:
        context = normalize_text(source_text)
        role = str(fact.role or "CONTEXT").upper()
        polarity = str(fact.polarity or "positive").lower()
        spans = self._dedupe(
            [self._repair_prompt_label(span, context) for span in fact.evidence_spans]
        )[: self.max_evidence_spans]

        if (
            not fact.subject.strip()
            or not fact.relation.strip()
            or not fact.object.strip()
            or role not in VALID_FACT_ROLES
            or polarity not in VALID_POLARITIES
            or not fact.source_id.strip()
            or not context
            or not spans
        ):
            return replace(
                fact,
                role=role if role in VALID_FACT_ROLES else "CONTEXT",
                polarity=polarity if polarity in VALID_POLARITIES else "positive",
                evidence_spans=spans,
                context=context,
                grounding_status="invalid",
            )

        grounded_spans = [span for span in spans if self._contains(context, span)]
        if len(grounded_spans) != len(spans):
            return replace(
                fact,
                role=role,
                polarity=polarity,
                evidence_spans=grounded_spans,
                context=context,
                grounding_status="invalid",
            )

        support_text = normalize_text(" ".join(grounded_spans))
        subject_grounded = self._entity_grounded(fact.subject, support_text, context)
        object_grounded = self._entity_grounded(fact.object, support_text, context)
        status = "grounded" if subject_grounded and object_grounded else "ambiguous"
        return replace(
            fact,
            role=role,
            polarity=polarity,
            evidence_spans=grounded_spans,
            context=context,
            grounding_status=status,
        )

    def validate_many(
        self,
        facts: list[EvidenceFact],
        *,
        source_text_by_id: dict[str, str],
    ) -> list[EvidenceFact]:
        return [
            self.validate(
                fact,
                source_text=source_text_by_id.get(fact.source_id, fact.context),
            )
            for fact in facts
        ]

    def validate_cross_context(
        self,
        fact: EvidenceFact,
        *,
        units: list[SemanticSourceUnit],
    ) -> EvidenceFact:
        """驗證跨單位事實的每一段文字與來源識別是否可追溯。"""

        unit_by_id = {normalize_text(unit.unit_id): unit for unit in units}
        refs: list[FactEvidenceRef] = []
        invalid_ref = False
        for ref in fact.evidence_refs:
            unit = unit_by_id.get(normalize_text(ref.unit_id))
            if unit is None or normalize_text(ref.source_id) != normalize_text(unit.source_id):
                invalid_ref = True
                continue
            source_text = normalize_text(unit.text)
            evidence_text = normalize_text(ref.text)
            start = source_text.casefold().find(evidence_text.casefold())
            if not evidence_text or start < 0:
                invalid_ref = True
                continue
            metadata = dict(unit.metadata or {})
            refs.append(
                replace(
                    ref,
                    document_id=(
                        normalize_text(ref.document_id)
                        or normalize_text(str(metadata.get("document_id") or unit.unit_id))
                    ),
                    page=(ref.page if ref.page is not None else self._page(metadata.get("page"))),
                    section=(
                        normalize_text(ref.section)
                        or normalize_text(str(metadata.get("section") or ""))
                    ),
                    start_offset=start,
                    end_offset=start + len(evidence_text),
                )
            )

        role = str(fact.role or "CONTEXT").upper()
        polarity = str(fact.polarity or "positive").lower()
        source_ids = {normalize_text(ref.source_id) for ref in refs if ref.source_id}
        unit_ids = {normalize_text(ref.unit_id) for ref in refs if ref.unit_id}
        context = "\n\n".join(
            f"[Unit {unit.unit_id}]\n{normalize_text(unit.text)}" for unit in units
        )
        spans = self._dedupe([ref.text for ref in refs])[: self.max_evidence_spans]
        structurally_valid = bool(
            role in VALID_FACT_ROLES
            and polarity in VALID_POLARITIES
            and fact.subject.strip()
            and fact.relation.strip()
            and fact.object.strip()
            and len(unit_ids) >= 2
            and len(source_ids) == 1
            and not invalid_ref
        )
        if not structurally_valid:
            return replace(
                fact,
                role=role if role in VALID_FACT_ROLES else "CONTEXT",
                polarity=polarity if polarity in VALID_POLARITIES else "positive",
                evidence_spans=spans,
                evidence_refs=refs,
                context=context,
                grounding_status="invalid",
            )

        support_text = normalize_text(" ".join(ref.text for ref in refs))
        subject_grounded = self._entity_grounded(
            fact.subject,
            support_text,
            support_text,
        )
        object_grounded = self._entity_grounded(
            fact.object,
            support_text,
            support_text,
        )
        status = "grounded" if subject_grounded and object_grounded else "ambiguous"
        return replace(
            fact,
            role=role,
            polarity=polarity,
            evidence_spans=spans,
            evidence_refs=refs,
            context=context,
            source_id=next(iter(source_ids)),
            grounding_status=status,
        )

    def _entity_grounded(self, value: str, evidence: str, context: str) -> bool:
        cleaned = normalize_text(value)
        if not cleaned:
            return False
        if self._contains(evidence, cleaned) or self._contains(context, cleaned):
            return True
        terms = {
            match.group(0).casefold()
            for match in self._WORD_RE.finditer(cleaned)
            if len(match.group(0)) > 2
        }
        if not terms:
            return False
        source_terms = {
            match.group(0).casefold()
            for match in self._WORD_RE.finditer(evidence)
        }
        return terms.issubset(source_terms)

    @staticmethod
    def _contains(container: str, value: str) -> bool:
        return normalize_text(value).casefold() in normalize_text(container).casefold()

    @staticmethod
    def _page(value: object) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    def _repair_prompt_label(self, span: str, context: str) -> str:
        cleaned = normalize_text(span)
        if self._contains(context, cleaned):
            return cleaned
        repaired = re.sub(
            r"^(?:span|text|context|evidence)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        return repaired if repaired and self._contains(context, repaired) else cleaned

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result


__all__ = ["FactGroundingValidator"]
