from __future__ import annotations

import os
import re
from typing import Any, Callable

from utils.network_utils import normalize_text, semantic_similarity_score

from .base import HandlerInput, HandlerMatch, HandlerResult, render_handler_evidence
from .capability import HandlerCapability, HandlerPreflightResult
from .input_adapters import HandlerInputAdapterRegistry
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
        input_adapter_registry: HandlerInputAdapterRegistry | None = None,
    ) -> None:
        self.registry = registry or default_deterministic_registry()
        self.threshold = (
            float(os.getenv("DETERMINISTIC_HANDLER_MATCH_THRESHOLD", "0.62"))
            if threshold is None
            else float(threshold)
        )
        self.similarity_fn = similarity_fn or semantic_similarity_score
        self.input_adapters = input_adapter_registry or HandlerInputAdapterRegistry()

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
        eligible_names = {
            str(value)
            for value in list((metadata or {}).get("eligible_handler_names") or [])
            if str(value).strip()
        }
        if eligible_names:
            matches = [match for match in matches if match.handler_name in eligible_names]
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
            if eligible_names and selected_handler_name not in eligible_names:
                return HandlerResult(
                    handler_name=selected_handler_name,
                    status="handler_unavailable",
                    error="selected handler is not eligible for the parsed attachment",
                    next_action_hint="Use only handlers exposed by the attachment capability preflight.",
                )
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

    def eligible_capabilities(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        attachment_result: str = "",
        search_result: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[HandlerCapability], list[HandlerPreflightResult]]:
        handler_input = HandlerInput(
            question=question,
            attachment=attachment or {},
            attachment_result=attachment_result,
            search_result=search_result,
            metadata=metadata or {},
        )
        capabilities: list[HandlerCapability] = []
        diagnostics: list[HandlerPreflightResult] = []
        for handler in self.registry.list_handlers():
            result = self._preflight_handler(handler, handler_input)
            diagnostics.append(result)
            attachment_profile = (metadata or {}).get("attachment_profile")
            attachment_scope = isinstance(attachment_profile, dict) and bool(attachment_profile)
            if not result.ready or (attachment_scope and not result.attachment_bound):
                continue
            capability = self.registry.capability_for(
                handler.name,
                available_inputs=result.available_inputs,
            )
            if capability is not None:
                capabilities.append(capability)
        return capabilities, diagnostics

    def preflight(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        attachment_result: str = "",
        search_result: str = "",
        metadata: dict[str, Any] | None = None,
        handler_name: str = "",
        required_handler_role: str = "",
        eligible_handler_names: list[str] | None = None,
    ) -> HandlerPreflightResult:
        handler_input = HandlerInput(
            question=question,
            attachment=attachment or {},
            attachment_result=attachment_result,
            search_result=search_result,
            metadata=metadata or {},
        )
        allowed = {str(name) for name in eligible_handler_names or [] if str(name)}
        handlers: list[Any] = []
        if handler_name:
            handler = self.registry.get(handler_name)
            if handler is not None:
                handlers = [handler]
        elif required_handler_role:
            handlers = self.registry.find_by_role(required_handler_role)
        if allowed:
            handlers = [handler for handler in handlers if handler.name in allowed]
        if not handlers:
            return HandlerPreflightResult(
                handler_name=handler_name,
                handler_role=required_handler_role,
                status="handler_unavailable",
                reason="handler is not registered or not eligible for this attachment",
            )
        results = [self._preflight_handler(handler, handler_input) for handler in handlers]
        return next((result for result in results if result.ready), results[0])

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
        adapter_result = self.input_adapters.adapt(handler_name, handler_input)
        adapted_handler_input = self._with_adapted_inputs(handler_input, adapter_result.inputs)
        try:
            handler_inputs = handler.build_input(adapted_handler_input)
        except Exception as exc:
            return HandlerResult.error_result(handler_name=handler_name, error=str(exc))
        handler_inputs = self._merge_adapted_inputs(handler_inputs, adapter_result.inputs)
        contract = getattr(handler, "input_schema", None)
        enforce_contract = bool(adapter_result.inputs) or bool(
            handler_input.metadata.get("require_attachment_provenance")
        )
        contract_missing = (
            contract.required_missing(handler_inputs)
            if enforce_contract and isinstance(contract, HandlerIOContract)
            else []
        )
        provenance_missing = self._attachment_provenance_missing(
            handler,
            handler_input,
            adapted_inputs=adapter_result.inputs,
        )
        missing_inputs = list(dict.fromkeys(provenance_missing + contract_missing))
        if missing_inputs:
            return HandlerResult.missing(
                handler_name=handler_name,
                missing_inputs=missing_inputs,
                structured_result={
                    "input_provenance": self._input_provenance(handler_input),
                    "attachment_profile": dict(
                        handler_input.metadata.get("attachment_profile") or {}
                    ),
                    "adapter": adapter_result.to_dict(),
                },
                next_action_hint=(
                    "Provide structured inputs extracted from the attachment before running "
                    f"{handler_name}."
                ),
            )
        try:
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
        result.structured_result.setdefault(
            "input_provenance", self._input_provenance(handler_input)
        )
        result.structured_result.setdefault("input_adapter", adapter_result.to_dict())
        if result.ok and not result.evidence_text:
            result.evidence_text = render_handler_evidence(result)
        return result

    def _attachment_provenance_missing(
        self,
        handler: Any,
        handler_input: HandlerInput,
        *,
        adapted_inputs: dict[str, Any] | None = None,
    ) -> list[str]:
        metadata = handler_input.metadata if isinstance(handler_input.metadata, dict) else {}
        if not metadata.get("require_attachment_provenance"):
            return []
        if getattr(handler, "uses_specialized_attachment_parser", False):
            file_path = str((adapted_inputs or {}).get("file_path") or "").strip()
            if file_path and os.path.isfile(file_path):
                return []
        profile = metadata.get("attachment_profile")
        if not isinstance(profile, dict):
            return ["attachment_profile"]
        if str(profile.get("parse_status") or "") not in {"success", "partial"}:
            return ["parsed_attachment"]

        available = {
            str(item).strip()
            for item in list(profile.get("available_inputs") or [])
            if str(item).strip()
        }
        available.update(
            key
            for key, value in (adapted_inputs or {}).items()
            if value is not None and value != "" and value != [] and value != {}
        )
        required_by_handler = {
            "table_exact_operations": {"rows"},
            "table_aggregation": {"rows"},
            "coordinate_distance": {"pairs"},
            "list_operations": {"list_items"},
            "graph_shortest_path": {"edges"},
            "boggle_dfs": {"grid"},
        }
        required = required_by_handler.get(str(getattr(handler, "name", "") or ""), set())
        return sorted(required - available)

    def _input_provenance(self, handler_input: HandlerInput) -> dict[str, Any]:
        metadata = handler_input.metadata if isinstance(handler_input.metadata, dict) else {}
        parsed_payload = metadata.get("parsed_payload")
        if not isinstance(parsed_payload, dict):
            return {}
        provenance = parsed_payload.get("provenance")
        return dict(provenance) if isinstance(provenance, dict) else {}

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
        result.operation = str(
            result.operation or result.structured_result.get("operation") or ""
        ).strip()
        result.structured_result.setdefault("output_type", result.output_type)
        result.structured_result.setdefault("semantic_role", result.semantic_role)
        result.structured_result.setdefault("supporting_inputs", list(result.supporting_inputs or []))
        result.structured_result.setdefault("calculation_trace", {})
        self._apply_derivation_metadata(result)
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

    @staticmethod
    def _apply_derivation_metadata(result: HandlerResult) -> None:
        """Normalize deterministic provenance without inventing task semantics."""

        structured = result.structured_result
        if not result.derivation_type:
            result.derivation_type = str(
                structured.get("derivation_type")
                or (
                    "deterministic_computation"
                    if result.output_type == "final_answer"
                    else "intermediate_extraction"
                )
            ).strip()
        if not result.derivation_trace:
            raw_trace = structured.get("derivation_trace")
            if isinstance(raw_trace, list):
                result.derivation_trace = [
                    dict(item) for item in raw_trace if isinstance(item, dict)
                ]
            if not result.derivation_trace:
                trace_payload = next(
                    (
                        structured.get(key)
                        for key in (
                            "calculation_trace",
                            "terms",
                            "evidence",
                            "rows",
                            "path",
                        )
                        if structured.get(key)
                    ),
                    None,
                )
                if trace_payload is not None:
                    result.derivation_trace = [
                        {
                            "operation": result.operation,
                            "inputs": list(result.supporting_inputs or []),
                            "result": result.answer,
                            "trace": trace_payload,
                        }
                    ]
        if not result.verification_payload:
            result.verification_payload = {
                "operation": result.operation,
                "answer": result.answer,
                "supporting_inputs": list(result.supporting_inputs or []),
                "derivation_type": result.derivation_type,
            }
        structured.setdefault("derivation_type", result.derivation_type)
        structured.setdefault("derivation_trace", list(result.derivation_trace))
        structured.setdefault("verification_payload", dict(result.verification_payload))

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
        adapter_result = self.input_adapters.adapt(handler.name, handler_input)
        attachment_scope = bool(
            isinstance(handler_input.metadata.get("attachment_profile"), dict)
            and handler_input.metadata.get("attachment_profile")
        )
        if not attachment_scope and not adapter_result.inputs:
            matcher = getattr(handler, "match_input", None)
            if callable(matcher):
                try:
                    match = matcher(handler_input)
                except Exception:
                    match = None
                if isinstance(match, HandlerMatch):
                    return match
            return None
        preflight = self._preflight_handler(handler, handler_input)
        return HandlerMatch(
            handler_name=handler.name,
            matched=preflight.ready,
            confidence=0.98 if preflight.ready else 0.3,
            reason=f"contract_preflight:{preflight.status}",
            handler_role=preflight.handler_role,
            missing_inputs=list(preflight.missing_inputs),
            required_inputs=list(preflight.required_inputs),
        )

    def _preflight_handler(
        self,
        handler: Any,
        handler_input: HandlerInput,
    ) -> HandlerPreflightResult:
        role = self.registry.role_for_handler(str(getattr(handler, "name", "") or ""))
        contract = getattr(handler, "input_schema", None)
        required_inputs = (
            contract.required_input_names()
            if isinstance(contract, HandlerIOContract)
            else []
        )
        extension = str(
            handler_input.attachment.get("extension")
            or os.path.splitext(
                str(
                    handler_input.attachment.get("file_path")
                    or handler_input.attachment.get("path")
                    or ""
                )
            )[1]
            or ""
        ).lower()
        supported_types = set(getattr(handler, "supported_attachment_types", set()) or set())
        if isinstance(contract, HandlerIOContract):
            supported_types.update(contract.supported_attachment_types)
        if extension and supported_types and extension not in supported_types:
            return HandlerPreflightResult(
                handler_name=handler.name,
                handler_role=role,
                status="unsupported_attachment",
                required_inputs=required_inputs,
                reason=f"{extension} is not supported",
            )

        adapter_result = self.input_adapters.adapt(handler.name, handler_input)
        adapted_handler_input = self._with_adapted_inputs(handler_input, adapter_result.inputs)
        try:
            base_inputs = handler.build_input(adapted_handler_input)
        except Exception as exc:
            return HandlerPreflightResult(
                handler_name=handler.name,
                handler_role=role,
                status="invalid_payload",
                required_inputs=required_inputs,
                reason=f"handler input build failed: {type(exc).__name__}: {exc}",
            )
        adapted_inputs = self._merge_adapted_inputs(base_inputs, adapter_result.inputs)
        attachment_scope = bool(
            isinstance(handler_input.metadata.get("attachment_profile"), dict)
            and handler_input.metadata.get("attachment_profile")
        )
        if not attachment_scope and not adapter_result.inputs:
            matcher = getattr(handler, "match_input", None)
            match = matcher(handler_input) if callable(matcher) else None
            if isinstance(match, HandlerMatch):
                missing_inputs = list(match.missing_inputs)
                matched = bool(match.matched)
            else:
                missing_inputs = (
                    contract.required_missing(adapted_inputs)
                    if isinstance(contract, HandlerIOContract)
                    else ["handler_readiness"]
                )
                matched = not missing_inputs
        else:
            missing_inputs = (
                contract.required_missing(adapted_inputs)
                if isinstance(contract, HandlerIOContract)
                else list(adapter_result.missing_inputs)
            )
            matched = not missing_inputs
        status = "ready"
        reason = adapter_result.reason or "handler input contract satisfied"
        if missing_inputs or not matched:
            status = (
                adapter_result.status
                if adapter_result.status in {"ambiguous_inputs", "invalid_payload"}
                else "missing_inputs"
            )
        return HandlerPreflightResult(
            handler_name=handler.name,
            handler_role=role,
            status=status,
            required_inputs=required_inputs,
            available_inputs=sorted(
                key
                for key, value in adapted_inputs.items()
                if value is not None and value != "" and value != [] and value != {}
            ),
            missing_inputs=missing_inputs,
            adapted_inputs=adapter_result.inputs,
            input_provenance=(
                adapter_result.input_provenance or self._input_provenance(handler_input)
            ),
            attachment_bound=bool(adapter_result.inputs),
            reason=reason,
        )

    @staticmethod
    def _merge_adapted_inputs(
        base_inputs: dict[str, Any],
        adapted_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base_inputs or {})
        for key, value in (adapted_inputs or {}).items():
            if value is not None and value != "" and value != [] and value != {}:
                merged[key] = value
        return merged

    @staticmethod
    def _with_adapted_inputs(
        handler_input: HandlerInput,
        adapted_inputs: dict[str, Any],
    ) -> HandlerInput:
        return HandlerInput(
            question=handler_input.question,
            attachment=dict(handler_input.attachment or {}),
            attachment_result=handler_input.attachment_result,
            search_result=handler_input.search_result,
            metadata={
                **dict(handler_input.metadata or {}),
                "adapted_inputs": dict(adapted_inputs or {}),
            },
        )

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
