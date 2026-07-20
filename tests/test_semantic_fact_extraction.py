from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from core.config import AgentReasoningSummary, EachAgentReply
from score.evidence_support_checker import EvidenceSupportChecker
from tools.evidence.fact_extraction import (
    EvidenceFact,
    FactGroundingValidator,
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactStore,
)
from tools.search_result_builder.evidence import EvidenceRoleContractBuilder
from tools.search_result_builder.evidence.span_role_classifier import (
    ANSWER_SUPPORT,
    CandidateSpan,
    SpanRoleClassifier,
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
            prompt_tokens=20,
            completion_tokens=10,
        )


def make_summary(answer: str) -> AgentReasoningSummary:
    reasoning = f"step 1. The source identifies the location as {answer}."
    run = EachAgentReply(
        agent_id="a1",
        model_name="test",
        run_index=1,
        raw_reply="",
        reasoning=reasoning,
        final_answer=answer,
        tool_context="",
        parse_completed=True,
        schema_valid=True,
        eligible_for_winner=True,
    )
    return AgentReasoningSummary(
        agent_id="a1",
        model_name="test",
        runs=[run],
        compressed_answer=answer,
        compressed_reasoning=reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )


class SemanticFactExtractionTests(unittest.TestCase):
    def test_grounding_rejects_unquoted_evidence(self) -> None:
        fact = EvidenceFact(
            fact_id="F1",
            subject="KGOT",
            relation="has studios in",
            object="Dimond Center",
            role="BRIDGE",
            evidence_spans=["KGOT is located in Anchorage."],
            source_id="D1",
        )
        result = FactGroundingValidator().validate(
            fact,
            source_text="KGOT has studios in the Dimond Center.",
        )
        self.assertEqual(result.grounding_status, "invalid")

    def test_semantic_extractor_returns_grounded_fact_and_unloads(self) -> None:
        content = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "U1",
                        "facts": [
                            {
                                "subject": "KGOT",
                                "relation": "has studios in",
                                "object": "Dimond Center",
                                "qualifiers": {},
                                "polarity": "positive",
                                "role": "BRIDGE",
                                "goal_id": "G1",
                                "evidence_spans": [
                                    "KGOT has studios in the Dimond Center."
                                ],
                            }
                        ],
                    }
                ]
            }
        )
        client = FakeLLMClient(content)
        result = SemanticFactExtractor(llm_client=client).extract_batch(
            question="Where are KGOT's studios?",
            answer_requirement="location",
            current_goal="find the studio location",
            units=[
                SemanticSourceUnit(
                    unit_id="U1",
                    text="KGOT has studios in the Dimond Center.",
                    source_id="D1",
                    source_type="web",
                )
            ],
        )
        self.assertTrue(result.diagnostics["success"])
        self.assertEqual(result.facts[0].grounding_status, "grounded")
        self.assertEqual(result.facts[0].object, "Dimond Center")
        self.assertFalse(client.calls[0]["think"])
        self.assertEqual(client.calls[0]["keep_alive"], 0)

    def test_span_classifier_only_returns_role_assignment(self) -> None:
        classifier = SpanRoleClassifier()
        results = classifier._normalize_results(
            [
                {
                    "id": "1",
                    "role": "ANSWER_SUPPORT",
                    "goal_id": "",
                }
            ],
            [
                CandidateSpan(
                    id="1",
                    text="728,000 square feet",
                    local_context=(
                        "The Dimond Center has 728,000 square feet of floor area."
                    ),
                    source_id="D2",
                )
            ],
        )
        self.assertEqual(results[0].role, ANSWER_SUPPORT)
        self.assertFalse(hasattr(results[0], "semantic_facts"))

    def test_span_classifier_uses_qwen4b_and_unloads_after_call(self) -> None:
        client = FakeLLMClient(
            json.dumps(
                {
                    "spans": [
                        {
                            "id": "1",
                            "role": "ANSWER_SUPPORT",
                            "goal_id": "",
                        }
                    ]
                }
            )
        )
        classifier = SpanRoleClassifier(
            model_name="qwen3:4b",
            llm_client=client,
        )
        classifier.classify_batch(
            question="How large is the Dimond Center?",
            answer_requirement="floor area",
            spans=[
                CandidateSpan(
                    id="1",
                    text="728,000 square feet",
                    local_context=(
                        "The Dimond Center has 728,000 square feet of floor area."
                    ),
                )
            ],
        )

        self.assertEqual(client.calls[0]["model"], "qwen3:4b")
        self.assertFalse(client.calls[0]["think"])
        self.assertEqual(client.calls[0]["keep_alive"], 0)

    def test_contract_uses_grounded_fact_object(self) -> None:
        text = "The Dimond Center has 728,000 square feet of floor area."
        contracts = EvidenceRoleContractBuilder().build(
            question="How large is the Dimond Center?",
            answer_requirement="floor area",
            answer_target="area",
            relation_plan=None,
            document_id="D2",
            source_title="Dimond Center",
            url="https://example.test",
            text=text,
            span_assignments=[
                {
                    "accepted": True,
                    "role": "ANSWER_SUPPORT",
                    "goal_id": "",
                    "original_text": text,
                    "finalized_text": text,
                    "semantic_facts": [
                        {
                            "fact_id": "F2",
                            "subject": "Dimond Center",
                            "relation": "has floor area",
                            "object": "728,000 square feet",
                            "qualifiers": {"answer_binding": "direct"},
                            "polarity": "positive",
                            "role": "ANSWER_SUPPORT",
                            "goal_id": "",
                            "evidence_spans": [text],
                            "grounding_status": "grounded",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(contracts.direct[0].answer_span, "728,000 square feet")
        self.assertEqual(contracts.direct[0].fact_id, "F2")

    def test_grounded_answer_fact_supports_agent(self) -> None:
        evidence = {
            "tool_usage": [
                {
                    "tool_name": "attachment_fact_extractor",
                    "raw_result": {
                        "semantic_facts": [
                            {
                                "fact_id": "F3",
                                "subject": "requested location",
                                "relation": "is",
                                "object": "Taipei Main Station",
                                "qualifiers": {"answer_binding": "direct"},
                                "polarity": "positive",
                                "role": "ANSWER_SUPPORT",
                                "evidence_spans": [
                                    "The requested location is Taipei Main Station."
                                ],
                                "context": (
                                    "The requested location is Taipei Main Station."
                                ),
                                "source_id": "attachment.pdf#page-1",
                                "source_type": "pdf",
                                "grounding_status": "grounded",
                            }
                        ]
                    },
                }
            ]
        }
        result = EvidenceSupportChecker().check_agent(
            target=make_summary("Taipei Main Station"),
            reasoning_steps=[
                (1, "The source identifies the location as Taipei Main Station.")
            ],
            evidence=evidence,
        )
        self.assertEqual(result.status, "attachment_evidence_supported")
        self.assertEqual(result.step_results[0].status, "supported")

    def test_fact_store_keeps_only_grounded_unique_facts(self) -> None:
        grounded = EvidenceFact(
            fact_id="F1",
            subject="A",
            relation="is",
            object="B",
            source_id="D1",
            grounding_status="grounded",
        )
        ambiguous = EvidenceFact(
            fact_id="F2",
            subject="A",
            relation="is",
            object="C",
            source_id="D1",
            grounding_status="ambiguous",
        )
        store = TaskFactStore()
        self.assertEqual(store.extend([grounded, grounded, ambiguous]), 1)
        self.assertEqual(store.to_dict()["fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
