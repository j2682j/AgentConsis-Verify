from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import re

from utils.network_utils import normalize_text

from .aggregation_models import AggregationDerivation
from .fact_store import TaskFactStore
from .models import EvidenceFact


class AggregationFactDeriver:
    """Build reproducible aggregate answers from grounded, complete fact groups."""

    _NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")

    def derive(
        self,
        fact_store: TaskFactStore,
        *,
        question: str = "",
        answer_requirement: str = "",
        goal_id: str = "",
        operator: str = "",
    ) -> list[AggregationDerivation]:
        selected_operator = normalize_text(operator).lower() or self.infer_operator(
            " ".join([question, answer_requirement])
        )
        if not selected_operator:
            return []
        if selected_operator == "set_difference":
            return self._set_difference(fact_store, goal_id=goal_id)

        facts = [
            fact
            for fact in fact_store.all()
            if (
                fact.grounding_status == "grounded"
                and fact.polarity == "positive"
                and not fact.parent_fact_ids
                and (not goal_id or fact.goal_id == goal_id)
            )
        ]
        groups: dict[tuple[str, str, str], list[EvidenceFact]] = defaultdict(list)
        for fact in facts:
            groups[
                (
                    normalize_text(fact.goal_id).casefold(),
                    normalize_text(fact.subject).casefold(),
                    normalize_text(fact.relation).casefold(),
                )
            ].append(fact)

        candidates: list[AggregationDerivation] = []
        for grouped in groups.values():
            contract_ids = self._complete_contracts_for(grouped, fact_store)
            if not contract_ids:
                continue
            result = self._calculate(selected_operator, grouped)
            if result is None:
                continue
            input_ids = sorted({fact.fact_id for fact in grouped if fact.fact_id})
            if not input_ids:
                continue
            candidates.append(
                AggregationDerivation(
                    derivation_id=(derivation_id := self._id(
                        selected_operator,
                        input_ids,
                        result,
                        contract_ids,
                    )),
                    operator=selected_operator,
                    input_fact_ids=input_ids,
                    result=result,
                    completeness_contract_ids=contract_ids,
                    result_fact_id=f"{derivation_id}-result",
                    grounding_status="grounded",
                    answer_bound=True,
                    goal_id=grouped[0].goal_id,
                    reason="complete_grounded_fact_group",
                    metadata={
                        "subject": grouped[0].subject,
                        "relation": grouped[0].relation,
                        "answer_requirement": normalize_text(answer_requirement),
                        "distinct_value_count": len(
                            {normalize_text(fact.object).casefold() for fact in grouped}
                        ),
                    },
                )
            )
        # Multiple unrelated groups are not a unique answer derivation.
        return candidates if len(candidates) == 1 else []

    def infer_operator(self, text: str) -> str:
        value = normalize_text(text).casefold()
        patterns = (
            ("percentage", r"\b(?:percentage|percent|proportion|rate)\b"),
            ("set_difference", r"\b(?:missing|not assigned|difference|left over)\b"),
            ("average", r"\b(?:average|mean)\b"),
            ("maximum", r"\b(?:maximum|highest|largest|most)\b"),
            ("minimum", r"\b(?:minimum|lowest|smallest|least)\b"),
            ("sum", r"\b(?:sum|total|altogether|combined)\b"),
            ("count", r"\b(?:how many|number of|count)\b"),
            ("intersection", r"\b(?:both|common to|intersection)\b"),
            ("union", r"\b(?:either|union|all distinct)\b"),
        )
        return next((name for name, pattern in patterns if re.search(pattern, value)), "")

    def _calculate(self, operator: str, facts: list[EvidenceFact]) -> str | None:
        objects = list(
            dict.fromkeys(
                normalize_text(fact.object)
                for fact in facts
                if normalize_text(fact.object)
            )
        )
        if operator == "count":
            return str(len(objects)) if objects else None
        if operator in {"intersection", "union"}:
            return None
        numbers = [self._number(value) for value in objects]
        if not numbers or any(value is None for value in numbers):
            return None
        values = [value for value in numbers if value is not None]
        if operator == "sum":
            return self._format(sum(values, Decimal(0)))
        if operator == "average":
            return self._format(sum(values, Decimal(0)) / Decimal(len(values)))
        if operator == "minimum":
            return self._format(min(values))
        if operator == "maximum":
            return self._format(max(values))
        if operator == "percentage":
            numerator = next(
                (
                    self._number(fact.object)
                    for fact in facts
                    if fact.qualifiers.get("aggregate_role") == "numerator"
                ),
                None,
            )
            denominator = next(
                (
                    self._number(fact.object)
                    for fact in facts
                    if fact.qualifiers.get("aggregate_role") == "denominator"
                ),
                None,
            )
            if numerator is None or denominator in {None, Decimal(0)}:
                return None
            return self._format((numerator / denominator) * Decimal(100)) + "%"
        return None

    def _complete_contracts_for(
        self,
        facts: list[EvidenceFact],
        fact_store: TaskFactStore,
    ) -> list[str]:
        complete = [
            contract for contract in fact_store.completeness_contracts() if contract.complete
        ]
        matching: list[str] = []
        fact_sources = {normalize_text(fact.source_id).casefold() for fact in facts}
        ref_units = {
            normalize_text(ref.unit_id).casefold()
            for fact in facts
            for ref in fact.evidence_refs
            if normalize_text(ref.unit_id)
        }
        for contract in complete:
            contract_source = normalize_text(contract.source_id).casefold()
            contract_units = {
                normalize_text(unit).casefold() for unit in contract.unit_ids
            }
            if fact_sources and fact_sources == {contract_source}:
                matching.append(contract.contract_id)
            elif ref_units and ref_units.issubset(contract_units):
                matching.append(contract.contract_id)
        return sorted(set(matching))

    def _set_difference(
        self,
        fact_store: TaskFactStore,
        *,
        goal_id: str,
    ) -> list[AggregationDerivation]:
        results: list[AggregationDerivation] = []
        complete_ids = {
            item.contract_id
            for item in fact_store.completeness_contracts()
            if item.complete
        }
        for item in fact_store.set_difference_derivations():
            if not item.missing_values:
                continue
            contract_ids = sorted(
                set(item.completeness_contract_ids) & complete_ids
            )
            if not contract_ids or len(contract_ids) != len(
                set(item.completeness_contract_ids)
            ):
                continue
            input_ids = sorted(
                set(item.universe_fact_ids + item.observed_fact_ids)
            )
            results.append(
                AggregationDerivation(
                    derivation_id=f"aggregate-{item.derivation_id}",
                    operator="set_difference",
                    input_fact_ids=input_ids,
                    result=", ".join(item.missing_values),
                    completeness_contract_ids=contract_ids,
                    result_fact_id=f"aggregate-{item.derivation_id}-result",
                    grounding_status="grounded",
                    answer_bound=True,
                    goal_id=goal_id,
                    reason="verified_set_difference",
                    metadata={"answer_requirement": "set difference"},
                )
            )
        return results

    def _number(self, value: str) -> Decimal | None:
        match = self._NUMBER_RE.search(normalize_text(value))
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    @staticmethod
    def _format(value: Decimal) -> str:
        normalized = value.normalize()
        text = format(normalized, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text

    @staticmethod
    def _id(
        operator: str,
        input_ids: list[str],
        result: str,
        contract_ids: list[str],
    ) -> str:
        payload = "\x1f".join([operator, *input_ids, result, *contract_ids])
        return "aggregate-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["AggregationFactDeriver"]
