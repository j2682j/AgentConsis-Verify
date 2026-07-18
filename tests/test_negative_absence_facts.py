from __future__ import annotations

from types import SimpleNamespace

from score.candidate_fact_verifier import CandidateFactVerifier
from tools.evidence.fact_extraction import (
    AbsenceChecker,
    CompletenessContractBuilder,
    EvidenceFact,
    NegativeFactBuilder,
    SetDifferenceFactDeriver,
    TaskFactCollector,
    TaskFactStore,
)
from tools.search_result_builder.query.relation_plan import RelationPlan
from tools.search_result_builder.source_analyze.full_document_verifier import (
    FullDocumentVerifier,
)


def _negative_goal(scope: str = "full_document"):
    return RelationPlan.from_specs(
        [
            {
                "subject": "Article A",
                "relation": "does not mention",
                "target": "plasmons",
                "polarity": "negative",
                "verification_scope": scope,
            }
        ]
    ).goals[0]


def test_incomplete_scope_returns_unknown_and_no_negative_fact() -> None:
    contract = CompletenessContractBuilder().build(
        scope_id="article-a",
        source_id="A",
        source_type="pdf",
        expected_units=10,
        processed_units=4,
    )
    check = AbsenceChecker().check(
        contract=contract,
        target="plasmons",
        units={"p1": "An optical experiment is described."},
    )
    fact = NegativeFactBuilder().from_absence(
        check=check,
        contract=contract,
        subject="Article A",
        relation="does not mention",
    )
    assert check.status == "unknown"
    assert fact is None


def test_complete_scope_creates_verifiable_closed_world_fact() -> None:
    contract = CompletenessContractBuilder().build(
        scope_id="article-a",
        source_id="A",
        source_type="pdf",
        expected_units=2,
        processed_units=2,
    )
    check = AbsenceChecker().check(
        contract=contract,
        target="plasmons",
        units={"p1": "An optical experiment.", "p2": "The conclusion."},
    )
    fact = NegativeFactBuilder().from_absence(
        check=check,
        contract=contract,
        subject="Article A",
        relation="does not mention",
    )
    assert fact is not None
    store = TaskFactStore()
    store.add_completeness_contract(contract)
    store.add_absence_check(check)
    store.add(fact)
    verification = CandidateFactVerifier().verify(
        candidate_answer="Article A",
        fact_store=store,
    )
    assert verification.status == "supported"
    assert verification.reason == "candidate_supported_by_closed_world_absence"


def test_present_target_cannot_create_absence_fact() -> None:
    contract = CompletenessContractBuilder().build(
        scope_id="article-a",
        source_id="A",
        source_type="web",
        expected_units=1,
        processed_units=1,
    )
    check = AbsenceChecker().check(
        contract=contract,
        target="plasmons",
        units={"full": "The article explicitly studies plasmons."},
    )
    assert check.status == "present"
    assert NegativeFactBuilder().from_absence(
        check=check,
        contract=contract,
        subject="Article A",
        relation="does not mention",
    ) is None


def test_explicit_negative_requires_an_exact_negative_span() -> None:
    builder = NegativeFactBuilder()
    valid = builder.validate_explicit(
        EvidenceFact(
            fact_id="N1",
            subject="Article A",
            relation="does not mention",
            object="plasmons",
            polarity="negative",
            role="ANSWER_SUPPORT",
            evidence_spans=["Article A does not mention plasmons."],
            context="Article A does not mention plasmons.",
            source_id="A",
            grounding_status="grounded",
        )
    )
    inferred = builder.validate_explicit(
        EvidenceFact(
            fact_id="N2",
            subject="Fred",
            relation="Giftee",
            object="Rebecca",
            polarity="negative",
            role="ANSWER_SUPPORT",
            evidence_spans=["Giftee: Fred | Recipient: Rebecca"],
            context="Giftee: Fred | Recipient: Rebecca",
            source_id="table",
            grounding_status="grounded",
        )
    )
    assert valid.grounding_status == "grounded"
    assert valid.qualifiers["negation_type"] == "explicit_negative"
    assert inferred.grounding_status == "invalid"


def test_complete_set_difference_derives_missing_value_with_provenance() -> None:
    builder = CompletenessContractBuilder()
    universe_contract = builder.build(
        scope_id="employees",
        source_id="secret-santa.docx",
        source_type="docx",
        expected_units=12,
        processed_units=12,
    )
    observed_contract = builder.build(
        scope_id="gift-givers",
        source_id="secret-santa.docx",
        source_type="docx",
        expected_units=11,
        processed_units=11,
    )
    result = SetDifferenceFactDeriver().derive(
        universe_values=["Harry", "Rebecca", "Fred"],
        observed_values=["Harry", "Rebecca"],
        completeness_contracts=[universe_contract, observed_contract],
        universe_fact_ids=["U1", "U2", "U3"],
        observed_fact_ids=["O1", "O2"],
        negative_relation="did not give",
        negative_object="gift",
        answer_subject="missing gift giver",
        answer_requirement="who did not give a gift",
    )
    assert result is not None
    derivation, facts = result
    assert derivation.missing_values == ["Fred"]
    assert {fact.object for fact in facts} == {"gift", "Fred"}
    assert all(fact.parent_fact_ids == ["U1", "U2", "U3", "O1", "O2"] for fact in facts)

    store = TaskFactStore()
    store.add_completeness_contract(universe_contract)
    store.add_completeness_contract(observed_contract)
    store.add_set_difference_derivation(derivation)
    store.extend(facts)
    verification = CandidateFactVerifier().verify(
        candidate_answer="Fred",
        fact_store=store,
    )
    assert verification.status == "supported"
    assert verification.support_kind == "derived"


def test_incomplete_set_difference_does_not_derive_missing_value() -> None:
    contract = CompletenessContractBuilder().build(
        scope_id="gift-givers",
        source_id="secret-santa.docx",
        source_type="docx",
        expected_units=11,
        processed_units=7,
    )
    assert SetDifferenceFactDeriver().derive(
        universe_values=["Harry", "Fred"],
        observed_values=["Harry"],
        completeness_contracts=[contract],
    ) is None


def test_full_document_verifier_exports_contract_check_and_fact() -> None:
    result = FullDocumentVerifier().verify(
        goal=_negative_goal(),
        documents=[
            SimpleNamespace(
                document_id="D1",
                record_id="R1",
                title="Article A",
                text="The complete article discusses an optical method.",
                content_scope="full_document",
                content_complete=True,
                content_truncated=False,
            )
        ],
    )
    assert result.resolved
    assert result.completeness_contracts[0].complete
    assert result.absence_checks[0].status == "absent"
    assert result.negative_facts[0].object == "plasmons"


def test_fact_store_round_trip_preserves_absence_audit_records() -> None:
    contract = CompletenessContractBuilder().build(
        scope_id="article-a",
        source_id="A",
        source_type="pdf",
        expected_units=1,
        processed_units=0,
    )
    check = AbsenceChecker().check(
        contract=contract,
        target="plasmons",
        units={},
    )
    store = TaskFactStore()
    store.add_completeness_contract(contract)
    store.add_absence_check(check)
    restored = TaskFactStore.from_dict(store.to_dict())
    assert restored.completeness_contracts()[0].contract_id == contract.contract_id
    assert restored.absence_checks()[0].status == "unknown"
    assert restored.to_dict()["unknown_absence_count"] == 1


def test_search_collector_preserves_negative_fact_and_scope_contract() -> None:
    result = FullDocumentVerifier().verify(
        goal=_negative_goal(),
        documents=[
            SimpleNamespace(
                document_id="D1",
                record_id="R1",
                title="Article A",
                text="The complete article discusses an optical method.",
                content_scope="full_document",
                content_complete=True,
                content_truncated=False,
            )
        ],
    )
    store = TaskFactStore()
    TaskFactCollector().collect_item(
        store,
        {
            "tool_name": "search",
            "raw_result": {
                "retrieval": {
                    "semantic_facts": [
                        item.to_dict() for item in result.negative_facts
                    ],
                    "completeness_contracts": [
                        item.to_dict() for item in result.completeness_contracts
                    ],
                    "absence_checks": [
                        item.to_dict() for item in result.absence_checks
                    ],
                }
            },
        },
        question="Which article does not mention plasmons?",
        source_scope="evidence_prepare",
    )
    assert store.to_dict()["closed_world_absence_count"] == 1
    verification = CandidateFactVerifier().verify(
        candidate_answer="Article A",
        fact_store=store,
    )
    assert verification.status == "supported"
