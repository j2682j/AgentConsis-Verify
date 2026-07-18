from __future__ import annotations

import json
import unittest

from core.config import AgentReasoningSummary, EachAgentReply
from core.evidence_runner import EvidenceRunner
from score.candidate_fact_verifier import CandidateFactVerifier
from score.evidence_support_checker import EvidenceSupportChecker
from tools.evidence.fact_extraction import (
    EvidenceFact,
    FactDerivationEngine,
    TaskFactCollector,
    TaskFactStore,
)


def fact(
    fact_id: str,
    subject: str,
    relation: str,
    object_value: str,
    *,
    role: str = "BRIDGE",
    goal_id: str = "",
    source_type: str = "search",
    polarity: str = "positive",
) -> EvidenceFact:
    text = f"{subject} {relation} {object_value}"
    answer_binding = "direct" if role == "ANSWER_SUPPORT" else "bridge"
    return EvidenceFact(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object=object_value,
        qualifiers={"answer_binding": answer_binding},
        polarity=polarity,
        role=role,
        goal_id=goal_id,
        evidence_spans=[text],
        context=text,
        source_id=f"source-{fact_id}",
        source_type=source_type,
        grounding_status="grounded",
    )


class TaskFactStoreAndDerivationTests(unittest.TestCase):
    def test_evidence_runner_exposes_shared_store_and_snapshot(self) -> None:
        semantic_fact = fact(
            "F-prepared",
            "A",
            "relates to",
            "B",
            role="ANSWER_SUPPORT",
        )
        runner = EvidenceRunner(question="What relates to A?")
        bundle = runner._finalize_evidence_bundle(
            {
                "tool_usage": [
                    {
                        "tool_name": "search",
                        "raw_result": {"semantic_facts": [semantic_fact.to_dict()]},
                    }
                ]
            }
        )
        self.assertEqual(bundle["fact_store"]["fact_count"], 1)
        self.assertNotIn("_fact_store", bundle)
        json.dumps(bundle)

    def test_collector_unifies_search_and_handler_facts(self) -> None:
        store = TaskFactStore()
        collector = TaskFactCollector()
        search_fact = fact(
            "F-search",
            "KGOT",
            "has studios in",
            "Dimond Center",
        )
        collector.collect_item(
            store,
            {
                "tool_name": "search",
                "raw_result": {
                    "evidence_items": [
                        {"semantic_facts": [search_fact.to_dict()]}
                    ]
                },
            },
            question="Where are the studios?",
            source_scope="evidence_prepare",
        )
        collector.collect_item(
            store,
            {
                "tool_name": "deterministic_handler_router",
                "handler_name": "area_converter",
                "output_type": "final_answer",
                "semantic_role": "area_answer",
                "value": "67,633 square metres",
                "output_text": "Converted area: 67,633 square metres",
                "supporting_inputs": ["728,000 square feet"],
                "evidence_valid": True,
            },
            question="What is the area in square metres?",
            source_scope="evidence_prepare",
        )
        self.assertEqual(len(store.all()), 2)
        self.assertEqual(store.to_dict()["source_counts"]["search"], 1)
        self.assertEqual(store.to_dict()["source_counts"]["handler"], 1)

    def test_relation_chain_creates_traceable_answer_fact(self) -> None:
        store = TaskFactStore()
        store.extend(
            [
                fact(
                    "F1",
                    "KGOT",
                    "has studios in",
                    "Dimond Center",
                    role="BRIDGE",
                    goal_id="G-area",
                ),
                fact(
                    "F2",
                    "Dimond Center",
                    "has floor area",
                    "728,000 square feet",
                    role="BRIDGE",
                    goal_id="G-area",
                ),
            ]
        )
        result = FactDerivationEngine(max_depth=1).derive(
            store,
            answer_requirement="area of the shopping mall",
        )
        self.assertEqual(len(result.added_fact_ids), 1)
        derived = store.get(result.added_fact_ids[0])
        self.assertIsNotNone(derived)
        self.assertEqual(derived.role, "ANSWER_SUPPORT")
        self.assertEqual(derived.parent_fact_ids, ["F1", "F2"])
        verification = CandidateFactVerifier().verify(
            candidate_answer="728,000 square feet",
            fact_store=store,
            answer_requirement="area of the shopping mall",
        )
        self.assertEqual(verification.status, "supported")
        self.assertEqual(verification.support_kind, "derived")
        self.assertEqual(set(verification.derivation_chain_ids), {"F1", "F2"})

    def test_negative_answer_fact_contradicts_candidate(self) -> None:
        store = TaskFactStore()
        store.add(
            fact(
                "F-negative",
                "Article A",
                "does not mention",
                "plasmons",
                role="ANSWER_SUPPORT",
                polarity="negative",
            )
        )
        verification = CandidateFactVerifier().verify(
            candidate_answer="plasmons",
            fact_store=store,
        )
        self.assertEqual(verification.status, "contradicted")
        self.assertEqual(verification.contradicting_fact_ids, ["F-negative"])

        subject_verification = CandidateFactVerifier().verify(
            candidate_answer="Article A",
            fact_store=store,
        )
        self.assertEqual(subject_verification.status, "supported")
        self.assertEqual(
            subject_verification.reason,
            "candidate_subject_satisfies_grounded_negative_condition",
        )

    def test_support_checker_uses_derived_relation_fact(self) -> None:
        store = TaskFactStore()
        store.extend(
            [
                fact(
                    "F1",
                    "KGOT",
                    "has studios in",
                    "Dimond Center",
                    role="BRIDGE",
                    goal_id="G-area",
                ),
                fact(
                    "F2",
                    "Dimond Center",
                    "has floor area",
                    "728,000 square feet",
                    role="BRIDGE",
                    goal_id="G-area",
                ),
            ]
        )
        reasoning = "step 1. The connected mall has 728,000 square feet."
        run = EachAgentReply(
            agent_id="a1",
            model_name="test",
            run_index=1,
            raw_reply="",
            reasoning=reasoning,
            final_answer="728,000 square feet",
            tool_context="",
            parse_completed=True,
            schema_valid=True,
            eligible_for_winner=True,
        )
        summary = AgentReasoningSummary(
            agent_id="a1",
            model_name="test",
            runs=[run],
            compressed_answer="728,000 square feet",
            compressed_reasoning=reasoning,
            confidence_score=1.0,
            active=True,
            valid_run_count=1,
            eligible_run_count=1,
        )
        evidence = {
            "_fact_store": store,
            "fact_store": store.to_dict(),
            "tool_usage": [],
            "answer_requirement": "area of the shopping mall",
        }
        support = EvidenceSupportChecker().check_agent(
            target=summary,
            reasoning_steps=[(1, "The connected mall has 728,000 square feet.")],
            evidence=evidence,
            question="How large is the shopping mall?",
        )
        self.assertEqual(support.status, "derived_evidence_supported")
        verification = support.metadata["candidate_fact_verification"]
        self.assertEqual(verification["status"], "supported")
        self.assertEqual(verification["support_kind"], "derived")

    def test_store_snapshot_round_trip_keeps_derivation_fields(self) -> None:
        store = TaskFactStore()
        store.add(
            EvidenceFact(
                **{
                    **fact(
                        "D1",
                        "A",
                        "relates to",
                        "B",
                        role="ANSWER_SUPPORT",
                        source_type="derived",
                    ).to_dict(),
                    "parent_fact_ids": ["F1", "F2"],
                    "derivation_type": "relation_chain",
                }
            )
        )
        restored = TaskFactStore.from_dict(store.to_dict())
        self.assertEqual(restored.get("D1").derivation_type, "relation_chain")
        self.assertEqual(restored.to_dict()["derived_fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
