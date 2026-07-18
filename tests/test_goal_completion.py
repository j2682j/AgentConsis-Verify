from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.search_result_builder.next_hop_query.goal_completion import (
    GoalCompletionEvaluator,
)
from tools.search_result_builder.next_hop_query.retrieval_recovery_policy import (
    RetrievalRecoveryPolicy,
)
from tools.search_result_builder.next_hop_query.relation_goal_resolver import (
    RelationGoalResolver,
)
from tools.search_result_builder.query.relation_plan import RelationPlan
from tools.search_result_builder.source_analyze.full_document_verifier import (
    FullDocumentVerifier,
)


def _negative_plan(scope: str = "full_document") -> RelationPlan:
    return RelationPlan.from_specs(
        [
            {
                "subject": "research article",
                "relation": "does not mention",
                "target": "plasmons",
                "source_kind": "academic",
                "polarity": "negative",
                "verification_scope": scope,
            }
        ]
    )


class FullDocumentVerifierTests(unittest.TestCase):
    def test_snippet_cannot_verify_absence(self) -> None:
        goal = _negative_plan().goals[0]
        result = FullDocumentVerifier().verify(
            goal=goal,
            documents=[
                SimpleNamespace(
                    document_id="D1",
                    record_id="R1",
                    title="Article A",
                    text="The abstract discusses a new optical method.",
                    content_scope="passage",
                    content_complete=False,
                    content_truncated=False,
                )
            ],
        )
        self.assertFalse(result.resolved)
        self.assertEqual(result.missing_reason, "complete_full_document_required")

    def test_complete_document_can_verify_absence(self) -> None:
        goal = _negative_plan().goals[0]
        result = FullDocumentVerifier().verify(
            goal=goal,
            documents=[
                SimpleNamespace(
                    document_id="D1",
                    record_id="R1",
                    title="Article A",
                    text="This complete article discusses a new optical method.",
                    content_scope="full_document",
                    content_complete=True,
                    content_truncated=False,
                )
            ],
        )
        self.assertTrue(result.resolved)
        self.assertEqual(result.resolved_values, ["Article A"])

    def test_absence_check_uses_all_chunks_from_the_same_document(self) -> None:
        goal = _negative_plan().goals[0]
        observed = [
            SimpleNamespace(
                document_id="D1",
                record_id="R1",
                title="Article A",
                url="https://example.test/a",
                text="The first section discusses an optical method.",
                content_scope="full_document",
                content_complete=True,
                content_truncated=False,
            )
        ]
        corpus = [
            {
                "id": "D1",
                "record_id": "R1",
                "title": "Article A",
                "url": "https://example.test/a",
                "text": "The first section discusses an optical method.",
                "content_scope": "full_document",
                "content_complete": True,
            },
            {
                "id": "D2",
                "record_id": "R1",
                "title": "Article A",
                "url": "https://example.test/a",
                "text": "The final section reports measurements of plasmons.",
                "content_scope": "full_document",
                "content_complete": True,
            },
        ]
        result = FullDocumentVerifier().verify(
            goal=goal,
            documents=observed,
            corpus_documents=corpus,
        )
        self.assertFalse(result.resolved)
        self.assertEqual(result.verifications[0].status, "present")

    def test_collection_requires_every_linked_record_to_be_fetched(self) -> None:
        goal = _negative_plan("collection").goals[0]
        corpus = [
            {"record_id": "R1", "title": "A", "content_url": "https://x/a"},
            {"record_id": "R2", "title": "B", "content_url": "https://x/b"},
        ]
        only_one = [
            SimpleNamespace(
                document_id="D1",
                record_id="R1",
                title="A",
                text="Complete article text without the target term.",
                content_scope="full_document",
                content_complete=True,
                content_truncated=False,
            )
        ]
        partial = FullDocumentVerifier().verify(
            goal=goal,
            documents=only_one,
            corpus_documents=corpus,
        )
        self.assertFalse(partial.resolved)

        complete = FullDocumentVerifier().verify(
            goal=goal,
            documents=[
                *only_one,
                SimpleNamespace(
                    document_id="D2",
                    record_id="R2",
                    title="B",
                    text="This article explicitly studies plasmons.",
                    content_scope="full_document",
                    content_complete=True,
                    content_truncated=False,
                ),
            ],
            corpus_documents=corpus,
        )
        self.assertTrue(complete.resolved)
        self.assertEqual(complete.resolved_values, ["A"])


class GoalCompletionEvaluatorTests(unittest.TestCase):
    def test_relation_completion_also_requires_direct_evidence(self) -> None:
        plan = RelationPlan.from_specs(
            [
                {
                    "subject": "Dimond Center",
                    "relation": "floor area",
                    "target": "mall size",
                }
            ]
        )
        plan = plan.replace_goal(
            plan.goals[0].replace(
                state="resolved",
                resolved_values=["728,000 square feet"],
                evidence_ids=["D1"],
            )
        )
        evaluator = GoalCompletionEvaluator()
        missing_direct = evaluator.evaluate(
            relation_plan=plan,
            documents=[],
            answer_gate_sufficient=True,
        )
        self.assertFalse(missing_direct.sufficient)

        supported = evaluator.evaluate(
            relation_plan=plan,
            documents=[
                SimpleNamespace(
                    direct_contracts=[
                        {
                            "goal_id": "G1",
                            "answer_span": "728,000 square feet",
                            "document_id": "D1",
                        }
                    ]
                )
            ],
            answer_gate_sufficient=True,
        )
        self.assertTrue(supported.sufficient)

    def test_negative_goal_cannot_be_resolved_by_a_passage_direct_contract(self) -> None:
        plan = _negative_plan()
        resolution = RelationGoalResolver().resolve_direct(
            plan,
            [
                {
                    "goal_id": "G1",
                    "answer_span": "Article A",
                    "document_id": "snippet-1",
                }
            ],
        )
        self.assertFalse(resolution.plan.complete)

    def test_later_goal_direct_contract_waits_for_active_goal(self) -> None:
        plan = RelationPlan.from_specs(
            [
                {"subject": "KGOT", "relation": "studio location", "target": "mall"},
                {"subject": "", "relation": "floor area", "target": "mall size"},
            ]
        )
        resolver = RelationGoalResolver()
        waiting = resolver.resolve_direct(
            plan,
            [
                {
                    "goal_id": "G2",
                    "answer_span": "728,000 square feet",
                    "document_id": "D2",
                }
            ],
        )
        self.assertEqual(waiting.plan.active_goal_id, "G1")
        self.assertEqual(waiting.resolved_goal_ids, [])

        first = resolver.resolve_direct(
            plan,
            [
                {
                    "goal_id": "G1",
                    "answer_span": "Dimond Center",
                    "document_id": "D1",
                },
                {
                    "goal_id": "G2",
                    "answer_span": "728,000 square feet",
                    "document_id": "D2",
                },
            ],
        )
        self.assertEqual(first.resolved_goal_ids, ["G1"])
        self.assertEqual(first.plan.active_goal_id, "G2")
        second = resolver.resolve_direct(first.plan, [
            {
                "goal_id": "G2",
                "answer_span": "728,000 square feet",
                "document_id": "D2",
            }
        ])
        self.assertTrue(second.plan.complete)
        self.assertEqual(second.plan.goals[0].resolved_values, ["Dimond Center"])
        self.assertEqual(second.plan.goals[1].resolved_values, ["728,000 square feet"])


class RetrievalRecoveryPolicyTests(unittest.TestCase):
    def test_full_document_goal_prefers_direct_fetch_then_browser(self) -> None:
        plan = _negative_plan("collection")
        corpus = [
            {
                "record_id": "R1",
                "content_url": "https://example.test/article",
                "content_scope": "collection_record",
            }
        ]
        policy = RetrievalRecoveryPolicy()
        attempted: set[str] = set()
        direct = policy.decide(
            relation_plan=plan,
            corpus_documents=corpus,
            attempted=attempted,
            top_k=8,
            candidate_pool_size=16,
            original_question="Which article does not mention plasmons?",
        )
        self.assertEqual(direct.action, "direct_fetch")
        attempted.add(direct.fingerprint)
        browser = policy.decide(
            relation_plan=plan,
            corpus_documents=corpus,
            attempted=attempted,
            top_k=8,
            candidate_pool_size=16,
            original_question="Which article does not mention plasmons?",
        )
        self.assertEqual(browser.action, "browser")

    def test_incomplete_goal_expands_when_unseen_corpus_documents_exist(self) -> None:
        policy = RetrievalRecoveryPolicy(max_top_k=16)
        decision = policy.decide(
            relation_plan=RelationPlan(),
            corpus_documents=[{"id": f"D{index}"} for index in range(5)],
            attempted=set(),
            top_k=2,
            candidate_pool_size=3,
            original_question="Who led the organization?",
        )
        self.assertEqual(decision.action, "expand_retrieval")
        self.assertEqual(decision.top_k, 4)


if __name__ == "__main__":
    unittest.main()
