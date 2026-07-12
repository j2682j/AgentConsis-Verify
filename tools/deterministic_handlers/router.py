from __future__ import annotations

import os
import re
from typing import Any, Callable

from utils.network_utils import normalize_text, semantic_similarity_score

from .base import HandlerInput, HandlerMatch, HandlerResult, render_handler_evidence
from .registry import HandlerRegistry, default_deterministic_registry
from .schema import HandlerIOContract, SCHEMA_VERSION


class DeterministicHandlerRouter:
    """
    Route a closed-world computation task to the most suitable deterministic handler.
    """

    def __init__(
        self,
        *,
        registry: HandlerRegistry | None = None,
        threshold: float | None = None,
        similarity_fn: Callable[[str, str], float | None] | None = None,
    ) -> None:
        self.registry = registry or default_deterministic_registry()
        self.threshold = (
            float(os.getenv("DETERMINISTIC_HANDLER_MATCH_THRESHOLD", "0.62"))
            if threshold is None
            else float(threshold)
        )
        self.similarity_fn = similarity_fn or semantic_similarity_score

    def run(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        attachment_result: str = "",
        search_result: str = "",
        metadata: dict[str, Any] | None = None,
        handler_name: str = "",
        required_handler_role: str = "",
    ) -> HandlerResult:
        handler_input = HandlerInput(
            question=question,
            attachment=attachment or {},
            attachment_result=attachment_result,
            search_result=search_result,
            metadata=metadata or {},
        )
        matches = self.match_handlers(handler_input)
        selected_role = str(required_handler_role or "").strip()
        if selected_role:
            role_handlers = self.registry.find_by_role(selected_role)
            if not role_handlers:
                return HandlerResult.missing_handler(
                    required_handler_role=selected_role,
                    matches=matches,
                )
            role_handler_names = {handler.name for handler in role_handlers}
            matches = [match for match in matches if match.handler_name in role_handler_names]
            if not matches:
                return HandlerResult.missing_handler(
                    required_handler_role=selected_role,
                    matches=[],
                )
        selected_handler_name = str(handler_name or "").strip()
        if selected_handler_name:
            if selected_role:
                role_handlers = self.registry.find_by_role(selected_role)
                role_handler_names = {handler.name for handler in role_handlers}
                if selected_handler_name not in role_handler_names:
                    return HandlerResult.missing_handler(
                        required_handler_role=selected_role,
                        matches=matches,
                    )
            selected = next(
                (match for match in matches if match.handler_name == selected_handler_name),
                None,
            )
            return self._run_selected_handler(
                selected_handler_name,
                handler_input=handler_input,
                matches=matches,
                selected_match=selected,
            )

        best = next((match for match in matches if match.matched), None)
        if best is None:
            best = next(
                (
                    match
                    for match in matches
                    if match.confidence >= self.threshold and match.missing_inputs
                ),
                None,
            )
        if best is None:
            return HandlerResult.no_match(matches=matches)

        return self._run_selected_handler(
            best.handler_name,
            handler_input=handler_input,
            matches=matches,
            selected_match=best,
        )

    def _run_selected_handler(
        self,
        handler_name: str,
        *,
        handler_input: HandlerInput,
        matches: list[HandlerMatch],
        selected_match: HandlerMatch | None,
    ) -> HandlerResult:
        handler = self.registry.get(handler_name)
        if handler is None:
            return HandlerResult.error_result(
                handler_name=handler_name,
                error="selected handler is not registered",
            )
        try:
            handler_inputs = handler.build_input(handler_input)
            result = handler.run(handler_inputs)
        except Exception as exc:
            return HandlerResult.error_result(
                handler_name=handler_name,
                error=str(exc),
            )

        self._apply_contract_metadata(result, handler, handler_inputs)
        result.structured_result.setdefault("matches", [match.to_dict() for match in matches])
        if selected_match is not None:
            result.structured_result.setdefault("selected_match", selected_match.to_dict())
        result.structured_result.setdefault("planned_handler_name", handler_name)
        if result.ok and not result.evidence_text:
            result.evidence_text = render_handler_evidence(result)
        return result

    def match_handlers(self, handler_input: HandlerInput) -> list[HandlerMatch]:
        query = self._routing_text(handler_input)
        matches: list[HandlerMatch] = []
        for handler in self.registry.list_handlers():
            base_confidence = max(
                self._similarity(query, handler.capability_description),
                self._handler_signal_score(query, handler),
            )
            readiness = self._handler_readiness(handler, handler_input)
            confidence = max(base_confidence, readiness.confidence if readiness else 0.0)
            missing_inputs = list(readiness.missing_inputs if readiness else [])
            input_contract = getattr(handler, "input_schema", None)
            registry_role = ""
            role_getter = getattr(self.registry, "role_for_handler", None)
            if callable(role_getter):
                registry_role = str(role_getter(handler.name) or "")
            required_inputs = (
                input_contract.required_input_names()
                if isinstance(input_contract, HandlerIOContract)
                else []
            )
            reason_parts = ["semantic_or_signal_match"]
            if readiness:
                reason_parts.append(readiness.reason)
            matches.append(
                HandlerMatch(
                    handler_name=handler.name,
                    matched=confidence >= self.threshold
                    and not (readiness and not readiness.matched and missing_inputs),
                    confidence=round(confidence, 6),
                    reason=";".join(part for part in reason_parts if part),
                    handler_role=str(getattr(handler, "handler_role", "") or registry_role),
                    missing_inputs=missing_inputs,
                    required_inputs=required_inputs,
                    schema_version=(
                        input_contract.schema_version
                        if isinstance(input_contract, HandlerIOContract)
                        else SCHEMA_VERSION
                    ),
                )
            )
        return sorted(matches, key=lambda match: match.confidence, reverse=True)

    def _apply_contract_metadata(
        self,
        result: HandlerResult,
        handler: Any,
        handler_inputs: dict[str, Any],
    ) -> None:
        input_contract = getattr(handler, "input_schema", None)
        output_contract = getattr(handler, "output_schema", None)
        if isinstance(input_contract, HandlerIOContract):
            result.output_schema_version = input_contract.schema_version
        if not result.input_summary:
            result.input_summary = self._input_summary(handler_inputs)

        result.structured_result.setdefault("input_summary", result.input_summary)
        registry_role = ""
        role_getter = getattr(self.registry, "role_for_handler", None)
        if callable(role_getter):
            registry_role = str(role_getter(getattr(handler, "name", "")) or "")
        result.structured_result.setdefault(
            "handler_role",
            str(getattr(handler, "handler_role", "") or registry_role),
        )
        result.structured_result.setdefault(
            "supported_answer_roles",
            sorted(getattr(handler, "supported_answer_roles", set()) or []),
        )
        result.structured_result.setdefault(
            "task_type",
            result.structured_result.get("task_type") or getattr(handler, "name", ""),
        )
        result.structured_result.setdefault(
            "operation",
            str(
                handler_inputs.get("operation")
                or result.structured_result.get("operation")
                or result.structured_result.get("task_type")
                or ""
            ),
        )
        result.structured_result.setdefault("output_type", result.output_type)
        result.structured_result.setdefault("semantic_role", result.semantic_role)
        result.structured_result.setdefault("supporting_inputs", list(result.supporting_inputs or []))
        result.structured_result.setdefault("calculation_trace", {})
        if isinstance(input_contract, HandlerIOContract):
            result.structured_result.setdefault(
                "input_contract",
                {
                    "schema_version": input_contract.schema_version,
                    "required_inputs": input_contract.required_input_names(),
                    "supported_attachment_types": sorted(input_contract.supported_attachment_types),
                },
            )
        if isinstance(output_contract, HandlerIOContract):
            result.structured_result.setdefault(
                "output_contract",
                {
                    "schema_version": output_contract.schema_version,
                    "required_outputs": output_contract.required_output_names(),
                },
            )

    def _input_summary(self, inputs: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, value in sorted((inputs or {}).items()):
            summary[key] = self._summarize_value(value)
        return summary

    def _summarize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            text = " ".join(value.split())
            return text[:180] + " ..." if len(text) > 180 else text
        if isinstance(value, list):
            payload: dict[str, Any] = {"type": "list", "count": len(value)}
            if value:
                payload["sample"] = self._summarize_value(value[0])
            return payload
        if isinstance(value, tuple):
            return {"type": "tuple", "count": len(value), "items": list(value[:4])}
        if isinstance(value, dict):
            keys = list(value.keys())
            return {
                "type": "dict",
                "count": len(value),
                "keys": [str(key) for key in keys[:12]],
            }
        return str(value)[:180]

    def _handler_readiness(
        self,
        handler: Any,
        handler_input: HandlerInput,
    ) -> HandlerMatch | None:
        matcher = getattr(handler, "match_input", None)
        if not callable(matcher):
            return None
        try:
            match = matcher(handler_input)
        except Exception:
            return None
        return match if isinstance(match, HandlerMatch) else None

    def _routing_text(self, handler_input: HandlerInput) -> str:
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        metadata_parts = [
            str(attachment.get("file_name", "") or ""),
            str(attachment.get("extension", "") or ""),
            str(attachment.get("file_path", "") or attachment.get("path", "") or ""),
        ]
        return normalize_text(
            "\n".join(
                part
                for part in [
                    handler_input.question,
                    " ".join(metadata_parts),
                    handler_input.attachment_result[:1600],
                    handler_input.search_result[:1600],
                ]
                if str(part or "").strip()
            )
        )

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        try:
            score = self.similarity_fn(left, right)
        except Exception:
            score = None
        if score is None:
            return self._lexical_similarity(left, right)
        return max(0.0, min(1.0, float(score)))

    def _lexical_similarity(self, left: str, right: str) -> float:
        left_terms = self._terms(left)
        right_terms = self._terms(right)
        if not left_terms or not right_terms:
            return 0.0
        overlap = left_terms & right_terms
        return len(overlap) / max(1, min(len(left_terms), len(right_terms)))

    def _handler_signal_score(self, query: str, handler: Any) -> float:
        routing_terms = {
            str(term or "").strip().lower()
            for term in getattr(handler, "routing_terms", set()) or set()
            if str(term or "").strip()
        }
        if not routing_terms:
            return 0.0
        query_terms = self._terms(query)
        overlap = query_terms & routing_terms
        if not overlap:
            return 0.0
        return min(1.0, 0.55 + 0.15 * len(overlap))

    def _terms(self, text: str) -> set[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "by",
            "from",
            "what",
            "which",
            "how",
            "many",
        }
        return {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?", text.lower())
            if len(term) > 1 and term not in stopwords
        }


__all__ = ["DeterministicHandlerRouter"]
