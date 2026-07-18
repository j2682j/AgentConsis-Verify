from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from utils.network_utils import normalize_text

from .answer_bound_validator import AnswerBoundFactValidator
from .context_assembler import CrossContextWindow
from .grounding_validator import FactGroundingValidator
from .models import (
    EvidenceFact,
    FactEvidenceRef,
    SemanticExtractionResult,
)
from .semantic_fact_extractor import SemanticFactExtractor


class CrossContextFactExtractor:
    """
    使用既有語意抽取模型，從同來源的多個相鄰單位中抽取跨段落事實。

    Args:
     - semantic_extractor: 提供模型、LLM client 與答案綁定設定的既有抽取器。
     - max_windows: 每次最多處理的跨上下文視窗數。
     - max_facts_per_window: 每個視窗最多接受的事實數。

    Returns:
     - CrossContextFactExtractor: 具有 multi-unit grounding 的跨上下文抽取器。
    """

    def __init__(
        self,
        *,
        semantic_extractor: SemanticFactExtractor | None = None,
        grounding_validator: FactGroundingValidator | None = None,
        answer_bound_validator: AnswerBoundFactValidator | None = None,
        max_windows: int = 6,
        max_facts_per_window: int = 6,
    ) -> None:
        self.semantic_extractor = semantic_extractor or SemanticFactExtractor()
        self.grounding_validator = (
            grounding_validator
            or self.semantic_extractor.grounding_validator
            or FactGroundingValidator()
        )
        self.answer_bound_validator = (
            answer_bound_validator
            or self.semantic_extractor.answer_bound_validator
            or AnswerBoundFactValidator()
        )
        self.max_windows = max(1, int(max_windows))
        self.max_facts_per_window = max(1, int(max_facts_per_window))

    def extract_windows(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        answer_target: str = "",
        current_goal: str = "",
        current_goal_id: str = "",
        windows: list[CrossContextWindow],
    ) -> SemanticExtractionResult:
        selected = windows[: self.max_windows]
        started = time.perf_counter()
        facts: list[EvidenceFact] = []
        rejected: list[dict[str, Any]] = []
        window_diagnostics: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        for window in selected:
            result = self._extract_window(
                question=question,
                answer_requirement=answer_requirement,
                answer_target=answer_target,
                current_goal=current_goal,
                current_goal_id=current_goal_id,
                window=window,
            )
            facts.extend(result.facts)
            rejected.extend(result.rejected_items)
            diagnostics = dict(result.diagnostics)
            window_diagnostics.append(diagnostics)
            prompt_tokens += int(diagnostics.get("prompt_tokens", 0) or 0)
            completion_tokens += int(diagnostics.get("completion_tokens", 0) or 0)

        return SemanticExtractionResult(
            facts=facts,
            rejected_items=rejected,
            diagnostics={
                "success": all(item.get("success", False) for item in window_diagnostics),
                "window_count": len(selected),
                "fact_count": len(facts),
                "grounded_count": sum(fact.grounding_status == "grounded" for fact in facts),
                "ambiguous_count": sum(fact.grounding_status == "ambiguous" for fact in facts),
                "invalid_count": sum(fact.grounding_status == "invalid" for fact in facts),
                "cross_context_fact_count": len(facts),
                "cross_context_grounded_count": sum(
                    fact.grounding_status == "grounded" for fact in facts
                ),
                "cross_context_ambiguous_count": sum(
                    fact.grounding_status == "ambiguous" for fact in facts
                ),
                "multi_unit_fact_count": sum(len(fact.evidence_refs) >= 2 for fact in facts),
                "multi_source_rejection_count": sum(
                    len({normalize_text(ref.source_id) for ref in fact.evidence_refs}) > 1
                    for fact in facts
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "windows": window_diagnostics,
            },
        )

    def _extract_window(
        self,
        *,
        question: str,
        answer_requirement: str,
        answer_target: str,
        current_goal: str,
        current_goal_id: str,
        window: CrossContextWindow,
    ) -> SemanticExtractionResult:
        diagnostics: dict[str, Any] = {
            "window_id": window.window_id,
            "unit_ids": list(window.unit_ids),
            "boundary_reason": window.boundary_reason,
            "success": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        try:
            response = self.semantic_extractor.llm_client.ollama_native_chat(
                model=self.semantic_extractor.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract only facts that require evidence from at least two supplied units. "
                            "Return JSON only. Do not infer unstated relations or include reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.build_prompt(
                            question=question,
                            answer_requirement=answer_requirement,
                            current_goal=current_goal,
                            current_goal_id=current_goal_id,
                            window=window,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=min(1536, max(512, self.semantic_extractor.max_tokens)),
                think=False,
                json_format=self.json_schema(),
                keep_alive=0,
            )
            parsed = self._parse_response(response.content)
            facts, rejected = self._normalize_and_ground(
                parsed=parsed,
                window=window,
                question=question,
                answer_requirement=answer_requirement,
                answer_target=answer_target,
                current_goal_id=current_goal_id,
            )
            diagnostics.update(
                {
                    "success": True,
                    "fact_count": len(facts),
                    "grounded_count": sum(fact.grounding_status == "grounded" for fact in facts),
                    "ambiguous_count": sum(fact.grounding_status == "ambiguous" for fact in facts),
                    "invalid_count": sum(fact.grounding_status == "invalid" for fact in facts),
                    "prompt_tokens": int(response.prompt_tokens or 0),
                    "completion_tokens": int(response.completion_tokens or 0),
                }
            )
            return SemanticExtractionResult(
                facts=facts,
                rejected_items=rejected,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return SemanticExtractionResult(diagnostics=diagnostics)

    def build_prompt(
        self,
        *,
        question: str,
        answer_requirement: str,
        current_goal: str,
        current_goal_id: str,
        window: CrossContextWindow,
    ) -> str:
        return "\n".join(
            [
                f"Question: {normalize_text(question)}",
                f"Answer Requirement: {normalize_text(answer_requirement) or 'Not specified'}",
                f"Current Relation Goal: {normalize_text(current_goal) or 'Not specified'}",
                f"Current Goal ID: {normalize_text(current_goal_id) or 'Not specified'}",
                "",
                "Ordered Source Units:",
                window.text,
                "",
                "Rules:",
                "- Extract a fact only when at least two different units are required to support it.",
                "- Every fact must include two or three evidence_refs copied exactly from their units.",
                "- subject and object must both be explicitly recoverable from the cited evidence_refs.",
                "- Resolve pronouns only when the antecedent is explicit in another cited unit.",
                "- Use ANSWER_SUPPORT only when object itself satisfies Answer Requirement.",
                "- Use BRIDGE for an intermediate entity or relation needed by the current goal.",
                "- Use CONTEXT when the grounded fact is relevant but neither direct nor a required bridge.",
                "- Use negative only when the cited evidence_refs explicitly state the negation.",
                "- Do not turn a positive table row into a negative fact because the Question asks what is missing.",
                "- Return an empty facts array when no cross-unit fact is explicit.",
            ]
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "relation": {"type": "string"},
                            "object": {"type": "string"},
                            "qualifiers": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "polarity": {"type": "string", "enum": ["positive", "negative"]},
                            "role": {
                                "type": "string",
                                "enum": ["ANSWER_SUPPORT", "BRIDGE", "CONTEXT"],
                            },
                            "goal_id": {"type": "string"},
                            "evidence_refs": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "unit_id": {"type": "string"},
                                        "text": {"type": "string"},
                                    },
                                    "required": ["unit_id", "text"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "subject", "relation", "object", "qualifiers", "polarity",
                            "role", "goal_id", "evidence_refs",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        }

    def _normalize_and_ground(
        self,
        *,
        parsed: Any,
        window: CrossContextWindow,
        question: str,
        answer_requirement: str,
        answer_target: str,
        current_goal_id: str,
    ) -> tuple[list[EvidenceFact], list[dict[str, Any]]]:
        unit_by_id = {normalize_text(unit.unit_id): unit for unit in window.units}
        facts: list[EvidenceFact] = []
        rejected: list[dict[str, Any]] = []
        raw_facts = list(parsed.get("facts") or []) if isinstance(parsed, dict) else []
        for index, raw_fact in enumerate(raw_facts[: self.max_facts_per_window], start=1):
            if not isinstance(raw_fact, dict):
                continue
            refs: list[FactEvidenceRef] = []
            for raw_ref in list(raw_fact.get("evidence_refs") or [])[:3]:
                if not isinstance(raw_ref, dict):
                    continue
                unit_id = normalize_text(str(raw_ref.get("unit_id") or ""))
                unit = unit_by_id.get(unit_id)
                if unit is None:
                    refs.append(
                        FactEvidenceRef(
                            source_id=window.source_id,
                            unit_id=unit_id,
                            text=normalize_text(str(raw_ref.get("text") or "")),
                        )
                    )
                    continue
                metadata = dict(unit.metadata or {})
                refs.append(
                    FactEvidenceRef(
                        source_id=unit.source_id,
                        unit_id=unit.unit_id,
                        text=normalize_text(str(raw_ref.get("text") or "")),
                        document_id=normalize_text(
                            str(metadata.get("document_id") or unit.unit_id)
                        ),
                        page=self._page(metadata.get("page")),
                        section=normalize_text(str(metadata.get("section") or "")),
                    )
                )
            role = normalize_text(str(raw_fact.get("role") or "CONTEXT")).upper()
            goal_id = normalize_text(str(raw_fact.get("goal_id") or ""))
            if role in {"ANSWER_SUPPORT", "BRIDGE"} and current_goal_id:
                goal_id = normalize_text(current_goal_id)
            qualifiers = {
                str(key): str(value)
                for key, value in dict(raw_fact.get("qualifiers") or {}).items()
            }
            qualifiers.update(
                {
                    "cross_context_window_id": window.window_id,
                    "boundary_reason": window.boundary_reason,
                }
            )
            fact = EvidenceFact.from_dict(
                {
                    **raw_fact,
                    "qualifiers": qualifiers,
                    "fact_id": self._fact_id(window.window_id, index, raw_fact),
                    "role": role,
                    "goal_id": goal_id,
                    "evidence_spans": [ref.text for ref in refs],
                    "evidence_refs": [ref.to_dict() for ref in refs],
                    "context": window.text,
                    "source_id": window.source_id,
                    "source_type": window.source_type,
                    "source_title": window.units[0].source_title if window.units else "",
                    "extraction_method": "cross_context_semantic_model",
                }
            )
            grounded = self.grounding_validator.validate_cross_context(
                fact,
                units=window.units,
            )
            grounded = self.semantic_extractor.negative_fact_builder.validate_explicit(
                grounded
            )
            bound = self.answer_bound_validator.bind(
                grounded,
                question=question,
                answer_requirement=answer_requirement,
                answer_target=answer_target,
            )
            facts.append(bound)
            if bound.grounding_status != "grounded":
                rejected.append(
                    {
                        "window_id": window.window_id,
                        "fact_id": bound.fact_id,
                        "reason": bound.grounding_status,
                        "unit_ids": [ref.unit_id for ref in bound.evidence_refs],
                    }
                )
        return facts, rejected

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            return {"facts": []}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"facts": []}
        except json.JSONDecodeError:
            return {"facts": []}

    @staticmethod
    def _fact_id(window_id: str, index: int, value: dict[str, Any]) -> str:
        raw = "|".join(
            [
                window_id,
                str(index),
                str(value.get("subject") or ""),
                str(value.get("relation") or ""),
                str(value.get("object") or ""),
            ]
        )
        return "CF-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _page(value: object) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None


__all__ = ["CrossContextFactExtractor"]
