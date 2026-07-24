from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class DerivedEvidenceContract:
    """Audit the validity, scope, and constraints of one derived fact."""

    derivation_type: str
    parent_fact_ids: list[str]
    operation_status: str
    entity_binding_status: str
    record_coherence: str
    scope_contract_ids: list[str] = field(default_factory=list)
    required_constraints: list[dict[str, Any]] = field(default_factory=list)
    satisfied_constraints: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    verification_status: str = "unverified"
    scope_status: str = "not_applicable"
    reasons: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.verification_status in {"verified", "legacy_accepted"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DerivedEvidenceContract":
        payload = value or {}
        return cls(
            derivation_type=str(payload.get("derivation_type") or ""),
            parent_fact_ids=[str(item) for item in payload.get("parent_fact_ids") or []],
            operation_status=str(payload.get("operation_status") or "unverified"),
            entity_binding_status=str(payload.get("entity_binding_status") or "unverified"),
            record_coherence=str(payload.get("record_coherence") or "unverified"),
            scope_contract_ids=[str(item) for item in payload.get("scope_contract_ids") or []],
            required_constraints=[
                dict(item) for item in payload.get("required_constraints") or []
                if isinstance(item, dict)
            ],
            satisfied_constraints=[str(item) for item in payload.get("satisfied_constraints") or []],
            missing_constraints=[str(item) for item in payload.get("missing_constraints") or []],
            verification_status=str(payload.get("verification_status") or "unverified"),
            scope_status=str(payload.get("scope_status") or "not_applicable"),
            reasons=[str(item) for item in payload.get("reasons") or []],
        )


class DerivedEvidenceContractValidator:
    """Build a conservative contract for a relation-chain derivation."""

    def validate_relation_chain(
        self,
        *,
        parent_facts: list[Any],
        required_constraints: list[dict[str, Any]] | None = None,
        scope_requirement: str = "not_applicable",
        completeness_contracts: list[Any] | None = None,
    ) -> DerivedEvidenceContract:
        parents = list(parent_facts or [])
        constraints = list(required_constraints or [])
        all_grounded = bool(parents) and all(
            getattr(item, "grounding_status", "") == "grounded" for item in parents
        )
        entity_bound = bool(
            len(parents) >= 2
            and self._entity_key(getattr(parents[0], "object", ""))
            == self._entity_key(getattr(parents[1], "subject", ""))
        )
        coherent = bool(
            all_grounded
            and entity_bound
            and all(getattr(item, "fact_id", "") for item in parents)
        )
        satisfied: list[str] = []
        missing: list[str] = []
        for constraint in constraints:
            key = self._constraint_key(constraint)
            if self._constraint_satisfied(constraint, parents):
                satisfied.append(key)
            else:
                missing.append(key)

        contract_map = {
            str(getattr(item, "contract_id", "")): item
            for item in list(completeness_contracts or [])
            if str(getattr(item, "contract_id", ""))
        }
        parent_contract_ids: list[str] = []
        for fact in parents:
            qualifiers = getattr(fact, "qualifiers", {}) or {}
            raw_ids = (
                qualifiers.get("completeness_contract_ids")
                or qualifiers.get("completeness_contract_id")
                or ""
            )
            if isinstance(raw_ids, str):
                parent_contract_ids.extend(
                    item.strip() for item in raw_ids.split(",") if item.strip()
                )
            elif isinstance(raw_ids, (list, tuple, set)):
                parent_contract_ids.extend(str(item) for item in raw_ids if str(item))
        scope_required = normalize_text(scope_requirement) not in {"", "not_applicable"}
        scope_complete = bool(
            not scope_required
            or (
                parent_contract_ids
                and all(
                    contract_id in contract_map
                    and bool(getattr(contract_map[contract_id], "complete", False))
                    for contract_id in parent_contract_ids
                )
            )
        )
        explicit_checks = bool(constraints or scope_required)
        verified = bool(
            all_grounded
            and entity_bound
            and coherent
            and not missing
            and scope_complete
        )
        reasons: list[str] = []
        if not all_grounded:
            reasons.append("parent_facts_not_grounded")
        if not entity_bound:
            reasons.append("entity_binding_failed")
        if not coherent:
            reasons.append("record_coherence_failed")
        if missing:
            reasons.append("required_constraints_missing")
        if not scope_complete:
            reasons.append("required_scope_incomplete")
        return DerivedEvidenceContract(
            derivation_type="relation_chain",
            parent_fact_ids=[str(getattr(item, "fact_id", "")) for item in parents],
            operation_status="verified" if all_grounded else "unverified",
            entity_binding_status="verified" if entity_bound else "failed",
            record_coherence="verified" if coherent else "failed",
            scope_contract_ids=list(dict.fromkeys(parent_contract_ids)),
            required_constraints=constraints,
            satisfied_constraints=satisfied,
            missing_constraints=missing,
            verification_status=(
                "verified"
                if verified and explicit_checks
                else "legacy_accepted"
                if verified
                else "unverified"
            ),
            scope_status=(
                "complete"
                if scope_complete and scope_required
                else "not_applicable"
                if not scope_required
                else "incomplete"
            ),
            reasons=reasons or ["derived_relation_contract_verified"],
        )

    def _constraint_satisfied(self, constraint: dict[str, Any], facts: list[Any]) -> bool:
        field_name = normalize_text(str(constraint.get("field") or "")).casefold()
        operator = str(constraint.get("operator") or "=").strip()
        expected = constraint.get("value")
        values: list[str] = []
        for fact in facts:
            qualifiers = getattr(fact, "qualifiers", {}) or {}
            if field_name and qualifiers.get(field_name) not in (None, ""):
                values.append(str(qualifiers[field_name]))
            if field_name == "year":
                text = " ".join(
                    str(value or "")
                    for value in (
                        getattr(fact, "subject", ""),
                        getattr(fact, "relation", ""),
                        getattr(fact, "object", ""),
                        getattr(fact, "context", ""),
                    )
                )
                values.extend(re.findall(r"\b(?:18|19|20)\d{2}\b", text))
        return any(self._compare(value, expected, operator) for value in values)

    @staticmethod
    def _compare(actual: Any, expected: Any, operator: str) -> bool:
        try:
            left: Any = Decimal(str(actual).replace(",", ""))
            right: Any = Decimal(str(expected).replace(",", ""))
        except (InvalidOperation, ValueError):
            left = normalize_text(str(actual)).casefold()
            right = normalize_text(str(expected)).casefold()
        return {
            "=": left == right,
            "==": left == right,
            "!=": left != right,
            ">": left > right,
            ">=": left >= right,
            "<": left < right,
            "<=": left <= right,
        }.get(operator, False)

    @staticmethod
    def _constraint_key(value: dict[str, Any]) -> str:
        return ":".join(
            [
                str(value.get("field") or ""),
                str(value.get("operator") or "="),
                str(value.get("value") or ""),
            ]
        )

    @staticmethod
    def _entity_key(value: str) -> str:
        return re.sub(r"[^\w]+", "", normalize_text(value).casefold())


@dataclass(frozen=True)
class FactDerivation:
    """描述一次可由來源事實重現的關係推導。"""

    derivation_id: str
    derivation_type: str
    parent_fact_ids: list[str]
    result_fact_id: str
    explanation: str
    grounded: bool
    contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactDerivationResult:
    """保存新推導事實、來源鏈與推導診斷資訊。"""

    derivations: list[FactDerivation] = field(default_factory=list)
    added_fact_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivations": [item.to_dict() for item in self.derivations],
            "added_fact_ids": list(self.added_fact_ids),
            "diagnostics": dict(self.diagnostics),
        }


__all__ = [
    "DerivedEvidenceContract",
    "DerivedEvidenceContractValidator",
    "FactDerivation",
    "FactDerivationResult",
]
