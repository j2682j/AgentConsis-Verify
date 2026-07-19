from __future__ import annotations

import re
from typing import Any

from .json_parse import try_parse_json
from .reasoning_parser import (
    ReasoningParseResult,
    prepare_reasoning_for_verifier,
)
from .stage1_output_repair import Stage1OutputRepairer
from .stage1_output_schema import Stage1StructuredOutput, ToolRequestPayload
from .stage1_output_validator import Stage1OutputValidator
from score.answer_validator import AnswerValidator
from utils.network_utils import normalize_text


class Stage1OutputParser:
    """
    Parse, repair, and validate Stage1 agent output into a stable schema.
    """

    def __init__(
        self,
        *,
        answer_validator: AnswerValidator | None = None,
        repairer: Stage1OutputRepairer | None = None,
        validator: Stage1OutputValidator | None = None,
    ) -> None:
        self.answer_validator = answer_validator or AnswerValidator()
        self.repairer = repairer or Stage1OutputRepairer(self.answer_validator)
        self.validator = validator or Stage1OutputValidator(self.answer_validator)

    def parse(self, reply: str, expected_weight_count: int = 0) -> dict[str, Any]:
        if not str(reply or "").strip():
            raise ValueError("Empty stage1 reply.")

        parsed = try_parse_json(reply)
        repair_actions: list[str] = []
        repair_applied = False
        if not isinstance(parsed, dict):
            parsed, repair_actions = self.repairer.repair(reply)
            repair_applied = bool(repair_actions)
        else:
            parsed = dict(parsed)

        output, reasoning_parse = self._coerce_structured_output(
            parsed,
            raw_reply=reply,
            expected_weight_count=expected_weight_count,
        )
        validation = self.validator.validate(output)
        payload = output.to_dict()
        payload.update(
            {
                "reasoning": output.reasoning_text(),
                "final_answer": self.answer_validator.clean(output.final_answer),
                "weights": output.weights,
                "parse_completed": validation.eligible_for_winner,
                "parse_error": None if validation.schema_valid else ";".join(validation.schema_errors),
                "structured_output": output.to_dict(),
                "schema_valid": validation.schema_valid,
                "schema_errors": validation.schema_errors,
                "repair_applied": repair_applied,
                "repair_actions": repair_actions,
                "eligible_for_winner": validation.eligible_for_winner,
                "validity_labels": validation.validity_labels,
                "reasoning_parse": reasoning_parse.to_dict(),
            }
        )
        return payload

    def _coerce_structured_output(
        self,
        parsed: dict[str, Any],
        *,
        raw_reply: str,
        expected_weight_count: int,
    ) -> tuple[Stage1StructuredOutput, ReasoningParseResult]:
        reply_type = str(parsed.get("type", "") or "").strip().lower()
        tool_request = self._coerce_tool_request(parsed)
        final_answer = "" if tool_request else self._first_present(
            parsed,
            ["final_answer", "correct_answer", "answer", "final", "result", "output"],
        )
        reasoning_parse = self._coerce_reasoning_steps(
            parsed,
            raw_reply=raw_reply,
            final_answer=str(final_answer or ""),
        )
        used_evidence_ids = self._coerce_evidence_ids(parsed, raw_reply=raw_reply)

        output = Stage1StructuredOutput(
            reasoning_steps=[
                f"step {number}. {body}"
                for number, body in reasoning_parse.steps
            ],
            final_answer=self.answer_validator.clean(
                final_answer or reasoning_parse.extracted_final_answer
            ),
            confidence=self._coerce_confidence(parsed.get("confidence")),
            used_evidence_ids=used_evidence_ids,
            answer_type=self._coerce_answer_type(parsed.get("answer_type")),
            tool_request=tool_request,
            weights=self._normalize_weights(parsed.get("weights"), expected_weight_count),
        )

        if reply_type == "tool_request" and output.tool_request is None:
            output.tool_request = ToolRequestPayload(
                tool_name=str(parsed.get("tool_name", "") or "").strip(),
                tool_args=parsed.get("tool_args") if isinstance(parsed.get("tool_args"), dict) else {},
                reasoning_step=str(parsed.get("reasoning_step", "") or "").strip(),
            )
            output.final_answer = ""
        return output, reasoning_parse

    def _coerce_tool_request(self, parsed: dict[str, Any]) -> ToolRequestPayload | None:
        tool_payload = parsed.get("tool_request")
        if isinstance(tool_payload, dict):
            return ToolRequestPayload(
                tool_name=str(tool_payload.get("tool_name") or tool_payload.get("name") or "").strip(),
                tool_args=(
                    tool_payload.get("tool_args")
                    if isinstance(tool_payload.get("tool_args"), dict)
                    else tool_payload.get("arguments")
                    if isinstance(tool_payload.get("arguments"), dict)
                    else {}
                ),
                reasoning_step=str(tool_payload.get("reasoning_step", "") or "").strip(),
            )
        if str(parsed.get("type", "") or "").strip().lower() == "tool_request":
            return ToolRequestPayload(
                tool_name=str(parsed.get("tool_name", "") or "").strip(),
                tool_args=parsed.get("tool_args") if isinstance(parsed.get("tool_args"), dict) else {},
                reasoning_step=str(parsed.get("reasoning_step", "") or "").strip(),
            )
        return None

    def _coerce_reasoning_steps(
        self,
        parsed: dict[str, Any],
        *,
        raw_reply: str,
        final_answer: str,
    ) -> ReasoningParseResult:
        source = parsed.get("reasoning_steps")
        if isinstance(source, list):
            steps = [normalize_text(item) for item in source if normalize_text(item)]
            if steps:
                return prepare_reasoning_for_verifier(
                    "",
                    final_answer=final_answer,
                    structured_steps=steps,
                )

        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, list):
            steps = [normalize_text(item) for item in reasoning if normalize_text(item)]
            if steps:
                return prepare_reasoning_for_verifier(
                    "",
                    final_answer=final_answer,
                    structured_steps=steps,
                )
        if reasoning:
            return prepare_reasoning_for_verifier(
                str(reasoning),
                final_answer=final_answer,
            )

        if parsed.get("reasoning_step"):
            return prepare_reasoning_for_verifier(
                "",
                final_answer=final_answer,
                structured_steps=[str(parsed.get("reasoning_step"))],
            )

        if isinstance(try_parse_json(raw_reply), dict):
            return prepare_reasoning_for_verifier(
                "",
                final_answer=final_answer,
            )

        return prepare_reasoning_for_verifier(
            raw_reply,
            final_answer=final_answer,
        )

    def _coerce_evidence_ids(self, parsed: dict[str, Any], *, raw_reply: str) -> list[str]:
        raw_ids = parsed.get("used_evidence_ids") or parsed.get("evidence_ids") or []
        ids: list[str] = []
        if isinstance(raw_ids, list):
            ids = [str(item).strip().upper() for item in raw_ids if str(item).strip()]
        elif isinstance(raw_ids, str):
            ids = re.findall(r"\bE\d+\b", raw_ids, re.IGNORECASE)
        if not ids:
            ids = re.findall(r"\bE\d+\b", raw_reply or "", re.IGNORECASE)

        seen: set[str] = set()
        result: list[str] = []
        for evidence_id in ids:
            match = re.fullmatch(r"E(\d+)", evidence_id.strip(), re.IGNORECASE)
            if not match:
                continue
            normalized = f"E{int(match.group(1))}"
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def _coerce_confidence(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_answer_type(self, value: Any) -> str:
        answer_type = str(value or "unknown").strip().lower()
        if "|" in answer_type:
            return "unknown"
        answer_type = answer_type.replace(" ", "_").replace("-", "_")
        return answer_type or "unknown"

    def _normalize_weights(self, weights: Any, expected_weight_count: int) -> list[int]:
        if expected_weight_count <= 0:
            return []
        if not isinstance(weights, list):
            return [3] * expected_weight_count
        normalized: list[int] = []
        for item in weights:
            try:
                value = float(item)
            except (TypeError, ValueError):
                return [3] * expected_weight_count
            mapped = int(round(1 + value * 4)) if 0.0 <= value <= 1.0 else int(round(value))
            normalized.append(max(1, min(5, mapped)))
        if len(normalized) != expected_weight_count:
            return [3] * expected_weight_count
        return normalized

    def _first_present(self, data: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None


__all__ = ["Stage1OutputParser"]
