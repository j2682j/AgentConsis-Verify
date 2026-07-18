from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from tools.evidence.fact_extraction import (
    CrossContextAssembler,
    CrossContextFactExtractor,
    AttachmentFactExtractor,
    EvidenceFact,
    FactEvidenceRef,
    FactGroundingValidator,
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactStore,
)


class FakeLLMClient:
    provider = "ollama"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def ollama_native_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=self.content,
            prompt_tokens=30,
            completion_tokens=20,
        )


def unit(
    unit_id: str,
    text: str,
    *,
    source_id: str = "https://example.test/page",
    order: int = 0,
) -> SemanticSourceUnit:
    return SemanticSourceUnit(
        unit_id=unit_id,
        text=text,
        source_id=source_id,
        source_type="web",
        source_title="Example",
        metadata={"order": order, "document_id": unit_id, "url": source_id},
    )


class CrossContextFactExtractionTests(unittest.TestCase):
    def test_assembler_builds_bounded_window_for_adjacent_shared_entity(self) -> None:
        units = [
            unit("P1", "KGOT broadcasts from studios in the Dimond Center.", order=1),
            unit("P2", "The Dimond Center has 728,000 square feet of floor area.", order=2),
            unit(
                "P3",
                "An unrelated source mentions a different subject.",
                source_id="https://other.test/page",
                order=3,
            ),
        ]
        windows = CrossContextAssembler().assemble(
            units,
            anchor_unit_ids=["P1"],
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].unit_ids, ["P1", "P2"])
        self.assertEqual(windows[0].boundary_reason, "shared_entity")
        self.assertNotIn("P3", windows[0].unit_ids)

    def test_cross_context_grounding_requires_two_valid_unit_refs(self) -> None:
        units = [
            unit("P1", "KGOT has studios in the Dimond Center.", order=1),
            unit("P2", "The Dimond Center has 728,000 square feet of floor area.", order=2),
        ]
        fact = EvidenceFact(
            fact_id="CF1",
            subject="Dimond Center",
            relation="has floor area",
            object="728,000 square feet",
            role="ANSWER_SUPPORT",
            source_id=units[0].source_id,
            evidence_refs=[
                FactEvidenceRef(
                    source_id=units[0].source_id,
                    unit_id="P1",
                    text="KGOT has studios in the Dimond Center.",
                ),
                FactEvidenceRef(
                    source_id=units[1].source_id,
                    unit_id="P2",
                    text="The Dimond Center has 728,000 square feet of floor area.",
                ),
            ],
        )
        grounded = FactGroundingValidator().validate_cross_context(
            fact,
            units=units,
        )
        self.assertEqual(grounded.grounding_status, "grounded")
        self.assertEqual(len(grounded.evidence_refs), 2)
        self.assertGreaterEqual(grounded.evidence_refs[1].start_offset, 0)

        invalid = FactGroundingValidator().validate_cross_context(
            EvidenceFact(
                fact_id="CF2",
                subject="Dimond Center",
                relation="has floor area",
                object="728,000 square feet",
                source_id=units[0].source_id,
                evidence_refs=[fact.evidence_refs[1]],
            ),
            units=units,
        )
        self.assertEqual(invalid.grounding_status, "invalid")

    def test_extractor_returns_answer_bound_cross_context_fact(self) -> None:
        response = json.dumps(
            {
                "facts": [
                    {
                        "subject": "Dimond Center",
                        "relation": "has floor area",
                        "object": "728,000 square feet",
                        "qualifiers": {},
                        "polarity": "positive",
                        "role": "ANSWER_SUPPORT",
                        "goal_id": "G2",
                        "evidence_refs": [
                            {
                                "unit_id": "P1",
                                "text": "KGOT has studios in the Dimond Center.",
                            },
                            {
                                "unit_id": "P2",
                                "text": "The Dimond Center has 728,000 square feet of floor area.",
                            },
                        ],
                    }
                ]
            }
        )
        client = FakeLLMClient(response)
        semantic_extractor = SemanticFactExtractor(llm_client=client)
        extractor = CrossContextFactExtractor(semantic_extractor=semantic_extractor)
        source_units = [
            unit("P1", "KGOT has studios in the Dimond Center.", order=1),
            unit("P2", "The Dimond Center has 728,000 square feet of floor area.", order=2),
        ]
        window = CrossContextAssembler().assemble(
            source_units,
            anchor_unit_ids=["P1"],
        )[0]
        result = extractor.extract_windows(
            question="How large is the mall where KGOT has its studios?",
            answer_requirement="floor area",
            answer_target="mall floor area",
            current_goal="Dimond Center -> floor area",
            current_goal_id="G2",
            windows=[window],
        )
        self.assertTrue(result.diagnostics["success"])
        self.assertEqual(result.facts[0].grounding_status, "grounded")
        self.assertEqual(result.facts[0].qualifiers["answer_binding"], "direct")
        self.assertEqual(len(result.facts[0].evidence_refs), 2)
        restored = EvidenceFact.from_dict(result.facts[0].to_dict())
        self.assertEqual(restored.evidence_refs[1].unit_id, "P2")
        self.assertFalse(client.calls[0]["think"])
        self.assertEqual(client.calls[0]["keep_alive"], 0)

    def test_fact_store_merges_provenance_for_same_fact(self) -> None:
        first = EvidenceFact(
            fact_id="F1",
            subject="A",
            relation="is located in",
            object="B",
            source_id="source",
            grounding_status="grounded",
            evidence_refs=[FactEvidenceRef("source", "P1", "A is located in B.")],
        )
        second = EvidenceFact(
            fact_id="F2",
            subject="A",
            relation="is located in",
            object="B",
            source_id="source",
            grounding_status="grounded",
            evidence_refs=[FactEvidenceRef("source", "P2", "The location is B.")],
        )
        store = TaskFactStore()
        self.assertTrue(store.add(first))
        self.assertFalse(store.add(second))
        self.assertEqual(len(store.all()), 1)
        self.assertEqual(len(store.all()[0].evidence_refs), 2)
        self.assertEqual(store.to_dict()["provenance_ref_count"], 2)

    def test_attachment_units_keep_table_header_and_rows_in_one_source(self) -> None:
        extractor = AttachmentFactExtractor(
            semantic_extractor=SemanticFactExtractor(
                llm_client=FakeLLMClient('{"units": []}')
            ),
            max_semantic_units=8,
        )
        units = extractor._semantic_units(
            question="Which author published the paper?",
            parsed_payload={
                "provenance": {"file_path": "paper.xlsx", "file_type": "xlsx"},
                "tables": [
                    {
                        "name": "Publications",
                        "columns": ["Title", "Author"],
                        "rows": [["A Study", "Ada"]],
                    }
                ],
            },
        )
        table_units = [item for item in units if item.unit_id.startswith("TABLE1")]
        self.assertEqual(len(table_units), 2)
        self.assertEqual({item.source_id for item in table_units}, {"paper.xlsx"})
        self.assertEqual(
            {item.metadata["table_id"] for item in table_units},
            {"TABLE1"},
        )


if __name__ == "__main__":
    unittest.main()
