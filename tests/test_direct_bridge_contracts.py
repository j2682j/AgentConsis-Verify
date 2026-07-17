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
        )
        self.assertEqual(items, [])

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


if __name__ == "__main__":
    unittest.main()
