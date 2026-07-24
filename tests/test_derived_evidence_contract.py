from score.candidate_fact_verifier import CandidateFactVerifier
from tools.evidence.fact_extraction import (
    EvidenceFact,
    FactDerivationEngine,
    TaskFactStore,
)


def _fact(
    fact_id: str,
    subject: str,
    relation: str,
    object_value: str,
    *,
    role: str = "BRIDGE",
    qualifiers: dict | None = None,
) -> EvidenceFact:
    return EvidenceFact(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object=object_value,
        qualifiers=dict(qualifiers or {}),
        role=role,
        evidence_spans=[f"{subject} {relation} {object_value}"],
        context=f"{subject} {relation} {object_value}",
        source_id=fact_id,
        source_type="web",
        grounding_status="grounded",
    )


def test_legacy_relation_derivation_remains_verifiable() -> None:
    store = TaskFactStore()
    store.add(_fact("F1", "Person", "was born in", "Country"))
    store.add(_fact("F2", "Country", "has capital", "Capital", role="ANSWER_SUPPORT"))

    FactDerivationEngine().derive(store, answer_requirement="What is the capital?")
    derived = next(fact for fact in store.all() if fact.parent_fact_ids)

    assert derived.derived_contract["verification_status"] == "legacy_accepted"
    assert derived.role == "ANSWER_SUPPORT"
    verification = CandidateFactVerifier().verify(
        candidate_answer="Capital",
        fact_store=store,
        answer_requirement="What is the capital?",
    )
    assert verification.status == "supported"


def test_missing_constraint_keeps_derived_fact_as_bridge() -> None:
    store = TaskFactStore()
    store.add(
        _fact(
            "F1",
            "Philip",
            "was nominated in",
            "1977",
            qualifiers={"year": "1977"},
        )
    )
    store.add(
        _fact(
            "F2",
            "1977",
            "belongs to country",
            "United States",
            role="ANSWER_SUPPORT",
            qualifiers={"nationality_status": "active"},
        )
    )

    FactDerivationEngine().derive(
        store,
        answer_requirement="Who was nominated after 1977 from a defunct country?",
        required_constraints=[
            {"field": "year", "operator": ">", "value": 1977},
            {"field": "nationality_status", "operator": "=", "value": "defunct"},
        ],
    )
    derived = next(fact for fact in store.all() if fact.parent_fact_ids)

    assert derived.derived_contract["verification_status"] == "unverified"
    assert derived.role == "BRIDGE"


def test_explicit_unverified_derived_fact_cannot_support_candidate() -> None:
    store = TaskFactStore()
    store.add(
        EvidenceFact(
            fact_id="D1",
            subject="requested result",
            relation="derived as",
            object="Philip",
            qualifiers={"answer_binding": "direct"},
            role="ANSWER_SUPPORT",
            evidence_spans=["Philip"],
            context="Incomplete derivation",
            source_id="derived",
            source_type="derived",
            grounding_status="grounded",
            parent_fact_ids=["F1", "F2"],
            derivation_type="relation_chain",
            derived_contract={"verification_status": "unverified"},
        )
    )

    result = CandidateFactVerifier().verify(
        candidate_answer="Philip",
        fact_store=store,
        answer_requirement="Who satisfies every constraint?",
    )

    assert result.status == "unknown"
    assert result.reason == "candidate_matches_unverified_derived_fact"
