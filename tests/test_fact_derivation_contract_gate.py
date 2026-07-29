"""Pin down what the derived-contract gate actually blocks.

RetrievalControl now runs FactDerivationEngine on the task-level fact store, so
relation chains spanning several documents can reach evidence authority. These
tests record what the `contract.verified` check inside `_compose` is worth.

The result is narrower than it looks: a contract with no explicit constraints
and no scope requirement resolves to `legacy_accepted`, which counts as
verified, so the gate only changes an outcome when the caller supplies
constraints or a scope requirement. RetrievalControl currently supplies
neither, which makes the gate a no-op on that path.
"""

from __future__ import annotations

import unittest

from tools.evidence.fact_extraction import EvidenceFact, FactDerivationEngine
from tools.evidence.fact_extraction.fact_store import TaskFactStore


MISSING_CONSTRAINT = [{"field": "year", "operator": "=", "value": "1999"}]


def _fact(fact_id: str, subject: str, relation: str, obj: str) -> EvidenceFact:
    return EvidenceFact(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object=obj,
        polarity="positive",
        role="BRIDGE",
        goal_id="G1",
        grounding_status="grounded",
        extraction_method="semantic_model",
        context=f"{subject} {relation} {obj}",
        evidence_spans=[f"{subject} {relation} {obj}"],
    )


def _chain_store() -> TaskFactStore:
    """Two grounded BRIDGE facts on one goal that compose: A->B, B->C.

    Shaped after a real level1_final_06 chain (Conduct -> Civility -> Clean
    start), which is a Wikipedia abbreviation list rather than an answer.
    """

    store = TaskFactStore()
    store.extend(
        [
            _fact("F-left", "Conduct", "stand for", "Civility"),
            _fact("F-right", "Civility", "stand for", "Clean start"),
        ]
    )
    return store


def _derive(*, gate: bool, constraints: list[dict] | None = None) -> EvidenceFact:
    store = _chain_store()
    FactDerivationEngine(enforce_contract_gate=gate).derive(
        store,
        answer_requirement="what does the abbreviation stand for",
        required_constraints=constraints,
    )
    derived = [fact for fact in store.all() if fact.parent_fact_ids]
    assert len(derived) == 1, derived
    return derived[0]


class FactDerivationContractGateTest(unittest.TestCase):
    def test_two_bridge_facts_on_one_goal_compose_into_answer_support(self) -> None:
        """The wiring's premise: BRIDGE parents can yield answer authority."""

        derived = _derive(gate=True)

        self.assertEqual(derived.role, "ANSWER_SUPPORT")
        self.assertEqual(derived.qualifiers.get("answer_binding"), "direct")
        self.assertEqual(derived.parent_fact_ids, ["F-left", "F-right"])

    def test_gate_is_a_no_op_without_constraints_or_scope(self) -> None:
        """No explicit checks -> legacy_accepted -> the gate never fires."""

        gated = _derive(gate=True)
        ungated = _derive(gate=False)

        self.assertEqual(gated.role, ungated.role)
        self.assertEqual(
            gated.qualifiers.get("answer_binding"),
            ungated.qualifiers.get("answer_binding"),
        )
        self.assertEqual(
            gated.derived_contract.get("verification_status"),
            "legacy_accepted",
        )

    def test_gate_only_bites_when_a_required_constraint_is_missing(self) -> None:
        gated = _derive(gate=True, constraints=MISSING_CONSTRAINT)

        self.assertEqual(gated.role, "BRIDGE")
        self.assertEqual(gated.qualifiers.get("answer_binding"), "bridge")
        self.assertEqual(
            gated.derived_contract.get("verification_status"), "unverified"
        )
        self.assertIn(
            "required_constraints_missing",
            gated.derived_contract.get("reasons") or [],
        )

    def test_dropping_the_gate_grants_authority_to_an_unverified_chain(self) -> None:
        """This is the risk the gate exists to prevent."""

        ungated = _derive(gate=False, constraints=MISSING_CONSTRAINT)

        self.assertEqual(ungated.role, "ANSWER_SUPPORT")
        self.assertEqual(ungated.qualifiers.get("answer_binding"), "direct")
        self.assertEqual(
            ungated.derived_contract.get("verification_status"), "unverified"
        )

    def test_gate_changes_only_the_granted_authority(self) -> None:
        gated = _derive(gate=True, constraints=MISSING_CONSTRAINT)
        ungated = _derive(gate=False, constraints=MISSING_CONSTRAINT)

        self.assertEqual(gated.subject, ungated.subject)
        self.assertEqual(gated.object, ungated.object)
        self.assertEqual(gated.parent_fact_ids, ungated.parent_fact_ids)
        self.assertNotEqual(gated.role, ungated.role)


if __name__ == "__main__":
    unittest.main()
