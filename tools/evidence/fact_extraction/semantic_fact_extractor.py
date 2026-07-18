from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

from core.llm_client import LLMClient
from utils.network_utils import normalize_text

from .answer_bound_validator import AnswerBoundFactValidator
from .grounding_validator import FactGroundingValidator
from .models import EvidenceFact, SemanticExtractionResult, SemanticSourceUnit
from .negative_fact_builder import NegativeFactBuilder


class SemanticFactExtractor:
    """
    使用小型語言模型將少量非結構化來源轉成具來源綁定的語意事實。

    Args:
     - model_name: Ollama 中負責語意事實抽取的模型名稱。
     - max_units_per_call: 單次允許處理的短來源單位數量。
     - max_context_chars: 每個來源單位可送入模型的最大字元數。

    Returns:
     - SemanticFactExtractor: 回傳已經過 grounding validation 的事實集合。
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        grounding_validator: FactGroundingValidator | None = None,
        answer_bound_validator: AnswerBoundFactValidator | None = None,
        negative_fact_builder: NegativeFactBuilder | None = None,
        max_units_per_call: int = 8,
        max_context_chars: int = 700,
        max_tokens: int = 768,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("SEMANTIC_FACT_EXTRACTOR_MODEL")
            or os.getenv("SPAN_ROLE_CLASSIFIER_MODEL")
            or "qwen3:1.7b"
        )
        self.llm_client = llm_client or LLMClient(provider="ollama")
        self.grounding_validator = grounding_validator or FactGroundingValidator()
        self.answer_bound_validator = (
            answer_bound_validator or AnswerBoundFactValidator()
        )
        self.negative_fact_builder = negative_fact_builder or NegativeFactBuilder()
        self.max_units_per_call = max(1, int(max_units_per_call))
        self.max_context_chars = max(160, int(max_context_chars))
        self.max_tokens = max(128, int(max_tokens))

    def extract_batch(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        current_goal: str = "",
        units: list[SemanticSourceUnit],
    ) -> SemanticExtractionResult:
        selected = self._dedupe_units(units)[: self.max_units_per_call]
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {
            "model": self.model_name,
            "input_count": len(selected),
            "success": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        if not selected:
            diagnostics.update({"success": True, "empty_reason": "no_source_units"})
            return SemanticExtractionResult(diagnostics=diagnostics)

        try:
            response = self.llm_client.ollama_native_chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract only facts explicitly stated in the supplied source. "
                            "Return JSON only. Do not answer the question, infer missing facts, "
                            "or include reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.build_prompt(
                            question=question,
                            answer_requirement=answer_requirement,
                            current_goal=current_goal,
                            units=selected,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=min(
                    2048,
                    max(self.max_tokens, 256 + len(selected) * 160),
                ),
                think=False,
                json_format=self.json_schema(),
                keep_alive=0,
            )
            parsed = self._parse_response(response.content)
            facts, rejected = self._normalize_and_ground(
                parsed,
                selected,
                question=question,
                answer_requirement=answer_requirement,
            )
            diagnostics.update(
                {
                    "success": True,
                    "fact_count": len(facts),
                    "grounded_count": sum(
                        fact.grounding_status == "grounded" for fact in facts
                    ),
                    "ambiguous_count": sum(
                        fact.grounding_status == "ambiguous" for fact in facts
                    ),
                    "invalid_count": sum(
                        fact.grounding_status == "invalid" for fact in facts
                    ),
                    "direct_bound_count": sum(
                        fact.qualifiers.get("answer_binding") == "direct"
                        for fact in facts
                    ),
                    "demoted_answer_fact_count": sum(
                        fact.qualifiers.get("original_role") == "ANSWER_SUPPORT"
                        for fact in facts
                    ),
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )
            diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            return SemanticExtractionResult(
                facts=facts,
                rejected_items=rejected,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            diagnostics.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                }
            )
            return SemanticExtractionResult(diagnostics=diagnostics)

    def build_prompt(
        self,
        *,
        question: str,
        answer_requirement: str,
        current_goal: str,
        units: list[SemanticSourceUnit],
    ) -> str:
        source_lines: list[str] = []
        for unit in units:
            source_lines.extend(
                [
                    f"Unit {unit.unit_id}",
                    f"Source ID: {unit.source_id}",
                    f"Source Type: {unit.source_type}",
                    f"Title: {normalize_text(unit.source_title) or 'Unknown'}",
                    f"Requested Role: {unit.requested_role or 'Classify from the question'}",
                    f"Goal ID: {unit.goal_id}",
                    f"Text: {self._truncate(unit.text)}",
                ]
            )
        return "\n".join(
            [
                "Task:",
                f"Question: {normalize_text(question)}",
                f"Answer Requirement: {normalize_text(answer_requirement) or 'Not specified'}",
                f"Current Goal: {normalize_text(current_goal) or 'Not specified'}",
                "",
                "Sources:",
                *source_lines,
                "",
                "Rules:",
                "- Extract only subject-relation-object facts explicitly stated in Text.",
                "- Copy one or two exact evidence spans from Text; do not paraphrase them.",
                "- ANSWER_SUPPORT means object itself is a final answer value for Answer Requirement.",
                "- A row, date, clue, entity, or intermediate value needed for later counting or comparison is BRIDGE.",
                "- For count, maximum, minimum, list, or calculated questions, do not mark an item as ANSWER_SUPPORT unless Text explicitly states the aggregate result.",
                "- Use CONTEXT for topical facts that do not fill the answer or a required intermediate relation.",
                "- Preserve dates, units, locations, counts, and other restrictions in qualifiers.",
                "- Use negative only when an evidence span itself explicitly states the negation.",
                "- Never infer a negative fact merely because the Question asks who or what is missing.",
                "- Return an empty facts array when a unit has no explicit useful fact.",
            ]
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        fact_properties: dict[str, Any] = {
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
            "evidence_spans": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 2,
            },
        }
        return {
            "type": "object",
            "properties": {
                "units": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "unit_id": {"type": "string"},
                            "facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": fact_properties,
                                    "required": list(fact_properties),
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["unit_id", "facts"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["units"],
            "additionalProperties": False,
        }

    def _normalize_and_ground(
        self,
        parsed: Any,
        units: list[SemanticSourceUnit],
        *,
        question: str = "",
        answer_requirement: str = "",
    ) -> tuple[list[EvidenceFact], list[dict[str, Any]]]:
        unit_by_id: dict[str, SemanticSourceUnit] = {}
        for unit in units:
            unit_by_id[unit.unit_id] = unit
            unit_by_id.setdefault(unit.source_id, unit)
        raw_units = parsed.get("units", []) if isinstance(parsed, dict) else []
        facts: list[EvidenceFact] = []
        rejected: list[dict[str, Any]] = []
        for raw_unit in raw_units:
            if not isinstance(raw_unit, dict):
                continue
            unit_id = normalize_text(str(raw_unit.get("unit_id") or ""))
            unit = unit_by_id.get(unit_id)
            if unit is None:
                rejected.append({"unit_id": unit_id, "reason": "unknown_unit_id"})
                continue
            for raw_fact in list(raw_unit.get("facts") or []):
                if not isinstance(raw_fact, dict):
                    continue
                role = normalize_text(str(raw_fact.get("role") or "CONTEXT")).upper()
                fact = EvidenceFact.from_dict(
                    {
                        **raw_fact,
                        "fact_id": self._fact_id(unit, raw_fact),
                        "role": role,
                        "goal_id": unit.goal_id,
                        "context": unit.text,
                        "source_id": unit.source_id,
                        "source_type": unit.source_type,
                        "source_title": unit.source_title,
                        "extraction_method": "semantic_model",
                    }
                )
                grounded = self.grounding_validator.validate(fact, source_text=unit.text)
                grounded = self.negative_fact_builder.validate_explicit(grounded)
                bound = self.answer_bound_validator.bind(
                    grounded,
                    question=question,
                    answer_requirement=answer_requirement,
                    answer_target=normalize_text(
                        str(unit.metadata.get("answer_target") or "")
                    ),
                )
                facts.append(bound)
                if bound.grounding_status != "grounded":
                    rejected.append(
                        {
                            "unit_id": unit_id,
                            "fact_id": bound.fact_id,
                            "reason": bound.grounding_status,
                        }
                    )
        return facts, rejected

    @staticmethod
    def _parse_response(content: str) -> Any:
        text = str(content or "").strip()
        if not text:
            return {"units": []}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"units": parsed}
        except json.JSONDecodeError:
            pass

        recovered: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in SemanticFactExtractor._balanced_json_objects(text):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or "unit_id" not in value or "facts" not in value:
                continue
            unit_id = normalize_text(str(value.get("unit_id") or ""))
            if not unit_id or unit_id in seen_ids:
                continue
            recovered.append(value)
            seen_ids.add(unit_id)
        return {"units": recovered}

    @staticmethod
    def _balanced_json_objects(text: str) -> list[str]:
        objects: list[str] = []
        starts: list[int] = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                starts.append(index)
            elif char == "}" and starts:
                start = starts.pop()
                objects.append(text[start : index + 1])
        return objects

    def _truncate(self, value: str) -> str:
        return normalize_text(value)[: self.max_context_chars]

    @staticmethod
    def _fact_id(unit: SemanticSourceUnit, value: dict[str, Any]) -> str:
        raw = "|".join(
            [
                unit.source_id,
                str(value.get("subject") or ""),
                str(value.get("relation") or ""),
                str(value.get("object") or ""),
                str(value.get("polarity") or "positive"),
            ]
        )
        return "F-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _dedupe_units(units: list[SemanticSourceUnit]) -> list[SemanticSourceUnit]:
        result: list[SemanticSourceUnit] = []
        seen: set[tuple[str, str]] = set()
        for unit in units:
            key = (unit.source_id, normalize_text(unit.text).casefold())
            if not unit.source_id or not normalize_text(unit.text) or key in seen:
                continue
            result.append(unit)
            seen.add(key)
        return result


__all__ = ["SemanticFactExtractor"]
