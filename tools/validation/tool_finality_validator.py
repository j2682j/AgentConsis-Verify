from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolFinalityResult:
    """Describe whether a tool result may act as a final answer."""

    status: str
    declared_output_type: str
    effective_output_type: str
    operation_complete: bool
    scope_complete: bool
    constraints_satisfied: bool
    provenance_valid: bool
    role_match: bool = True
    missing_inputs: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    legacy_accepted: bool = False

    @property
    def final(self) -> bool:
        return self.status == "final" and self.effective_output_type == "final_answer"

    @property
    def usable_as_intermediate(self) -> bool:
        return self.status == "intermediate" and self.effective_output_type == "intermediate_value"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolFinalityValidator:
    """Validate explicit finality metadata without changing legacy tool behavior."""

    _COMPLETE = {"complete", "completed", "verified", "not_applicable"}

    def validate(
        self,
        *,
        declared_output_type: str,
        result_ok: bool,
        answer: str,
        missing_inputs: list[str] | None = None,
        finality_payload: Mapping[str, Any] | None = None,
    ) -> ToolFinalityResult:
        declared = str(declared_output_type or "intermediate_value").strip()
        missing = self._strings(missing_inputs)
        if not result_ok or not str(answer or "").strip():
            return ToolFinalityResult(
                status="invalid",
                declared_output_type=declared,
                effective_output_type="invalid",
                operation_complete=False,
                scope_complete=False,
                constraints_satisfied=False,
                provenance_valid=False,
                missing_inputs=missing,
                reasons=["tool_result_not_usable"],
            )
        if declared != "final_answer":
            return ToolFinalityResult(
                status="intermediate",
                declared_output_type=declared,
                effective_output_type="intermediate_value",
                operation_complete=True,
                scope_complete=True,
                constraints_satisfied=True,
                provenance_valid=True,
                missing_inputs=missing,
                reasons=["declared_intermediate_value"],
            )
        payload = dict(finality_payload or {})
        if not payload:
            return ToolFinalityResult(
                status="final",
                declared_output_type=declared,
                effective_output_type="final_answer",
                operation_complete=True,
                scope_complete=True,
                constraints_satisfied=True,
                provenance_valid=True,
                missing_inputs=missing,
                reasons=["legacy_finality_metadata_absent"],
                legacy_accepted=True,
            )

        required_inputs = self._strings(payload.get("required_inputs"))
        consumed_inputs = {
            item.casefold() for item in self._strings(payload.get("consumed_inputs"))
        }
        for item in required_inputs:
            if item.casefold() not in consumed_inputs and item not in missing:
                missing.append(item)
        operation_complete = bool(
            self._complete(payload.get("operation_status")) and not missing
        )
        scope_complete = self._complete(payload.get("scope_status"))
        required_constraints = self._strings(payload.get("required_constraints"))
        satisfied_constraints = {
            item.casefold() for item in self._strings(payload.get("satisfied_constraints"))
        }
        explicit_missing = self._strings(payload.get("missing_constraints"))
        missing_constraints = list(explicit_missing)
        for item in required_constraints:
            if item.casefold() not in satisfied_constraints and item not in missing_constraints:
                missing_constraints.append(item)
        constraints_satisfied = not missing_constraints
        provenance_status = str(payload.get("provenance_status") or "").strip().lower()
        provenance_ids = self._strings(payload.get("provenance_ids"))
        provenance_valid = bool(
            provenance_status == "not_applicable"
            or (
                provenance_ids
                and provenance_status not in {"invalid", "failed", "missing"}
            )
        )
        role_match = bool(payload.get("role_match", True))
        reasons: list[str] = []
        if missing:
            reasons.append("finality_missing_inputs")
        if not operation_complete:
            reasons.append("finality_operation_incomplete")
        if not scope_complete:
            reasons.append("finality_scope_incomplete")
        if not constraints_satisfied:
            reasons.append("finality_constraints_incomplete")
        if not provenance_valid:
            reasons.append("finality_provenance_invalid")
        if not role_match:
            reasons.append("finality_role_mismatch")
        final = not reasons
        invalid = not provenance_valid
        return ToolFinalityResult(
            status="final" if final else "invalid" if invalid else "intermediate",
            declared_output_type=declared,
            effective_output_type=(
                "final_answer" if final else "invalid" if invalid else "intermediate_value"
            ),
            operation_complete=operation_complete,
            scope_complete=scope_complete,
            constraints_satisfied=constraints_satisfied,
            provenance_valid=provenance_valid,
            role_match=role_match,
            missing_inputs=missing,
            missing_constraints=missing_constraints,
            reasons=reasons or ["explicit_finality_contract_verified"],
        )

    def _complete(self, value: Any) -> bool:
        return str(value or "").strip().lower() in self._COMPLETE

    @staticmethod
    def _strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


__all__ = ["ToolFinalityResult", "ToolFinalityValidator"]
