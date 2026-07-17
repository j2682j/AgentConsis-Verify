from __future__ import annotations

import json
from typing import Any, Protocol

from tools.deterministic_solver.schemas import DeterministicSolverResult

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class SolverAlgorithm(Protocol):
    def solve(self, question: str, **kwargs: Any) -> DeterministicSolverResult:
        ...


class SolverBackedRouterHandler:
    """
    Bridge one existing deterministic algorithm into the DeterministicHandler protocol.
    """

    name = ""
    capability_description = ""
    supported_attachment_types: set[str] = set()
    missing_inputs: list[str] = ["complete_deterministic_input"]
    input_schema = io_contract(
        "solver_backed",
        [
            input_field(
                "complete_deterministic_input",
                "str",
                True,
                "Closed-world input required by the wrapped deterministic solver.",
                "question|attachment|search",
            )
        ],
        default_outputs(),
    )
    output_schema = input_schema

    def __init__(self, algorithm: SolverAlgorithm) -> None:
        self.algorithm = algorithm

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        adapted = handler_input.adapted_inputs()
        combined_text = handler_input.combined_text()
        if adapted:
            combined_text = "\n".join(
                part for part in [combined_text, json.dumps(adapted, ensure_ascii=False)] if part
            )
        return {
            "question": handler_input.question,
            "attachment_context": handler_input.attachment_result,
            "search_context": handler_input.search_result,
            "combined_text": combined_text,
            "metadata": {**dict(handler_input.metadata or {}), **adapted},
            **adapted,
        }

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        try:
            result = self.algorithm.solve(
                str(inputs.get("combined_text") or inputs.get("question") or ""),
                attachment_context=str(inputs.get("attachment_context") or ""),
                table_data=inputs.get("metadata", {}).get("table_data"),
            )
        except Exception:
            result = None
        matched = bool(result and result.used_deterministic_solver)
        return HandlerMatch(
            handler_name=self.name,
            matched=matched,
            confidence=0.96 if matched else 0.0,
            reason="solver_algorithm_input_readiness",
            missing_inputs=[] if matched else list(self.missing_inputs),
        )

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        result = self.algorithm.solve(
            str(inputs.get("combined_text") or inputs.get("question") or ""),
            attachment_context=str(inputs.get("attachment_context") or ""),
            table_data=inputs.get("metadata", {}).get("table_data"),
        )
        if not result.used_deterministic_solver:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=list(self.missing_inputs),
                structured_result=result.to_dict(),
                next_action_hint="Provide the complete deterministic input required by this handler.",
            )
        return self._to_handler_result(result)

    def _to_handler_result(self, result: DeterministicSolverResult) -> HandlerResult:
        evidence = result.evidence if isinstance(result.evidence, dict) else {}
        structured_result = {
            "task_type": result.task_type,
            "confidence": result.confidence,
            "evidence": evidence,
            "source": "deterministic_algorithm",
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=str(result.answer_text or result.answer or "").strip(),
            evidence_text=self._render_evidence(result),
            structured_result=structured_result,
            confidence=float(result.confidence or 0.0),
            output_type="final_answer",
            semantic_role=str(result.task_type or self.name),
            supporting_inputs=self._supporting_inputs(evidence),
        )

    def _render_evidence(self, result: DeterministicSolverResult) -> str:
        evidence = result.evidence if isinstance(result.evidence, dict) else {}
        lines = [
            "Deterministic handler evidence:",
            f"Handler: {self.name}",
            f"Task: {result.task_type}",
            f"Answer: {str(result.answer_text or result.answer or '').strip()}",
        ]
        if evidence:
            lines.append(f"Computation: {self._compact_evidence(evidence)}")
        lines.append(
            "Instruction: prefer this exact deterministic result for closed-world computation tasks."
        )
        return "\n".join(lines)

    def _compact_evidence(self, evidence: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in evidence.items():
            text = str(value)
            if len(text) > 400:
                text = text[:400].rstrip() + " ..."
            parts.append(f"{key}={text}")
        return "; ".join(parts)

    def _supporting_inputs(self, evidence: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for value in evidence.values():
            if isinstance(value, (list, tuple)):
                result.extend(str(item) for item in value[:8])
            elif isinstance(value, dict):
                result.extend(f"{key}={item}" for key, item in list(value.items())[:8])
            elif value is not None:
                result.append(str(value))
        return [item for item in result if item][:12]


__all__ = ["SolverBackedRouterHandler"]
