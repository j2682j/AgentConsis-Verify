from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.search_result_builder.evidence import (
    CandidateSpan,
    EvidenceConverter,
    EvidenceRoleContractBuilder,
    EvidenceSelectionContract,
    SpanRoleClassifier,
)
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore
from tools.search_result_builder.evidence.role_aware_span_finalizer import (
    RoleAwareSpanFinalizer,
)
from tools.search_result_builder.next_hop_query import RelationEvidenceBinder
from tools.search_result_builder.query import RelationPlan
from tools.search_result_builder.source_analyze.label_contract import LabelContractValidator
from tools.search_result_builder.source_analyze.rag_labeler import (
    CONTINUE_TAG,
    FINISH_TAG,
    RAGLabelResult,
)


def _relation_plan() -> RelationPlan:
    return RelationPlan.from_specs(
        [
            {
                "subject": "KGOT",
                "relation": "studio location",
                "target": "mall name",
                "source_kind": "web",
            },
            {
                "subject": "",
                "relation": "floor area",
                "target": "mall size",
                "source_kind": "web",
            },
        ]
    )


class EvidenceRoleContractTests(unittest.TestCase):
    def test_direct_precedence_prevents_dual_authority(self) -> None:
        contracts = EvidenceRoleContractBuilder().build(
            question="How large is the mall?",
            answer_requirement="area of the mall",
            answer_target="mall size",
            relation_plan=_relation_plan(),
            document_id="D1",
            source_title="Dimond Center",
            url="https://example.test",
            text="The Dimond Center has 728,000 square feet of floor area.",
            direct_spans=["728,000 square feet"],
            bridge_spans=["728,000 square feet"],
        )
        self.assertEqual(len(contracts.direct), 1)
        self.assertEqual(contracts.bridge, [])
        self.assertEqual(contracts.unsupported[0].reason, "direct_contract_precedence")

    def test_bridge_requires_active_and_next_relation_goal(self) -> None:
        contracts = EvidenceRoleContractBuilder().build(
            question="Where is KGOT located?",
            answer_requirement="location",
            answer_target="KGOT studio location",
            relation_plan=RelationPlan(),
            document_id="D1",
            source_title="KGOT",
            url="https://example.test",
            text="KGOT broadcasts from the Dimond Center.",
            direct_spans=[],
            bridge_spans=["Dimond Center"],
        )
        self.assertEqual(contracts.bridge, [])
        self.assertEqual(contracts.unsupported[0].reason, "missing_active_or_next_goal")

    def test_converter_ignores_bridge_only_document(self) -> None:
        bridge_fact = EvidenceFact(
            fact_id="F-BRIDGE",
            subject="KGOT",
            relation="broadcasts from",
            object="Dimond Center",
            qualifiers={"answer_binding": "bridge"},
            role="BRIDGE",
            evidence_spans=["KGOT broadcasts from the Dimond Center."],
            context="KGOT broadcasts from the Dimond Center.",
            source_id="D1",
            source_type="search",
            source_title="KGOT",
            grounding_status="grounded",
        )
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 1,
                        "query": "KGOT studio location",
                        "documents": [
                            {
                                "document_id": "D1",
                                "title": "KGOT",
                                "text": "KGOT broadcasts from the Dimond Center.",
                                "url": "https://example.test/kgot",
                                "bridge_spans": ["Dimond Center"],
                                "semantic_facts": [bridge_fact.to_dict()],
                                "bridge_contracts": [
                                    {
                                        "goal_id": "G1",
                                        "bridge_span": "Dimond Center",
                                        "context": "KGOT broadcasts from the Dimond Center.",
                                        "document_id": "D1",
                                        "next_goal_id": "G2",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        items = EvidenceConverter().convert_web_retrieval_output(
            output,
            contract=EvidenceSelectionContract.from_parts(
                question="How large is the mall where KGOT has studios?",
                answer_requirement="mall size",
            ),
            fact_store=(store := TaskFactStore()),
        )
        self.assertEqual(items, [])
        self.assertEqual(store.to_dict()["fact_count"], 1)
        self.assertEqual(store.to_dict()["role_counts"]["BRIDGE"], 1)

    def test_fact_store_revision_changes_only_for_new_information(self) -> None:
        store = TaskFactStore()
        fact = EvidenceFact(
            fact_id="F1",
            subject="album",
            relation="count",
            object="3",
            qualifiers={"answer_binding": "direct"},
            role="ANSWER_SUPPORT",
            evidence_spans=["The artist released three albums."],
            context="The artist released three albums.",
            source_id="D1",
            source_type="search",
            grounding_status="grounded",
        )

        self.assertEqual(store.revision, 0)
        self.assertTrue(store.add(fact))
        self.assertEqual(store.revision, 1)
        self.assertFalse(store.add(fact))
        self.assertEqual(store.revision, 1)

    def test_converter_accepts_direct_contract_only(self) -> None:
        context = "The Dimond Center has 728,000 square feet of floor area."
        output = {
            "retrieval": {
                "rounds": [
                    {
                        "round_index": 2,
                        "query": "Dimond Center floor area",
                        "documents": [
                            {
                                "document_id": "D2",
                                "title": "Dimond Center",
                                "text": context,
                                "url": "https://example.test/dimond",
                                "sequence_tag": FINISH_TAG,
                                "answer_support_spans": ["728,000 square feet"],
                                "direct_contracts": [
                                    {
                                        "goal_id": "G2",
                                        "answer_span": "728,000 square feet",
                                        "context": context,
                                        "document_id": "D2",
                                        "source_title": "Dimond Center",
                                        "url": "https://example.test/dimond",
                                        "answer_requirement": "mall size",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        items = EvidenceConverter().convert_web_retrieval_output(output)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["matched_terms"], ["728,000 square feet"])
        self.assertEqual(items[0]["bridge_spans"], [])
        self.assertFalse(items[0]["valid_for_next_hop"])
        self.assertEqual(items[0]["selection_reason"], "direct_evidence_contract")

    def test_label_sequence_tag_does_not_authorize_final_evidence(self) -> None:
        validator = LabelContractValidator()
        for tag in (CONTINUE_TAG, FINISH_TAG):
            result = validator.validate(
                RAGLabelResult(
                    label="useful",
                    kept_tokens=["Dimond Center"],
                    metadata={"sequence_tag": tag},
                )
            )
            self.assertFalse(result.valid_for_evidence)

    def test_relation_binder_ignores_uncontracted_bridge_span(self) -> None:
        binding = RelationEvidenceBinder().bind(
            plan=_relation_plan(),
            documents=[
                SimpleNamespace(
                    document_id="D1",
                    title="KGOT",
                    text="KGOT broadcasts from the Dimond Center.",
                    bridge_spans=["Dimond Center"],
                    bridge_contracts=[],
                )
            ],
        )
        self.assertEqual(binding.evidence, [])

    def test_classifier_prompt_contains_active_and_next_goal(self) -> None:
        prompt = SpanRoleClassifier()._prompt(
            question="How large is the mall where KGOT has studios?",
            answer_requirement="mall size",
            answer_target="floor area",
            active_goal="KGOT -> studio location -> mall name",
            next_goal="mall name -> floor area -> mall size",
            spans=[
                CandidateSpan(
                    id="1",
                    text="Dimond Center",
                    local_context="KGOT broadcasts from the Dimond Center.",
                    source_title="KGOT",
                )
            ],
        )
        self.assertIn("Active Goal: KGOT -> studio location -> mall name", prompt)
        self.assertIn("Next Goal: mall name -> floor area -> mall size", prompt)

    def test_classifier_keeps_only_valid_goal_assignments(self) -> None:
        classifier = SpanRoleClassifier()
        candidates = [
            CandidateSpan("1", "Dimond Center", "KGOT is in Dimond Center."),
            CandidateSpan("2", "728,000 square feet", "The area is 728,000 square feet."),
        ]
        results = classifier._normalize_results(
            [
                {"id": "1", "role": "BRIDGE", "goal_id": "G1"},
                {"id": "2", "role": "ANSWER_SUPPORT", "goal_id": "G9"},
            ],
            candidates,
            valid_goal_ids={"G1", "G2"},
        )
        self.assertEqual(results[0].goal_id, "G1")
        self.assertEqual(results[0].role, "BRIDGE")
        self.assertEqual(results[1].role, "NOISE")
        self.assertEqual(results[1].goal_id, "")

    def test_finalizer_and_contract_builder_preserve_goal_id(self) -> None:
        text = (
            "KGOT broadcasts from the Dimond Center. "
            "The Dimond Center has 728,000 square feet of floor area."
        )
        finalized = RoleAwareSpanFinalizer().finalize_batch(
            items=[
                {"id": "1", "text": "Dimond Center", "role": "BRIDGE", "goal_id": "G1"},
                {
                    "id": "2",
                    "text": "728,000 square feet",
                    "role": "ANSWER_SUPPORT",
                    "goal_id": "G2",
                },
            ],
            context=text,
            source_title="Dimond Center",
        )
        assignments = [item.to_dict() for item in finalized.finalized]
        assignments[1]["semantic_facts"] = [
            {
                "fact_id": "F-area",
                "subject": "Dimond Center",
                "relation": "has floor area",
                "object": "728,000 square feet",
                "qualifiers": {"answer_binding": "direct"},
                "polarity": "positive",
                "role": "ANSWER_SUPPORT",
                "goal_id": "G2",
                "evidence_spans": [
                    "The Dimond Center has 728,000 square feet of floor area."
                ],
                "context": text,
                "grounding_status": "grounded",
            }
        ]
        contracts = EvidenceRoleContractBuilder().build(
            question="How large is the mall where KGOT has studios?",
            answer_requirement="mall size",
            answer_target="floor area",
            relation_plan=_relation_plan(),
            document_id="D1",
            source_title="Dimond Center",
            url="https://example.test/dimond",
            text=text,
            span_assignments=assignments,
        )
        self.assertEqual(contracts.bridge[0].goal_id, "G1")
        self.assertEqual(contracts.direct[0].goal_id, "G2")
        self.assertEqual(contracts.direct[0].answer_span, "728,000 square feet")
        self.assertNotEqual(contracts.direct[0].answer_span, contracts.direct[0].context)


if __name__ == "__main__":
    unittest.main()
