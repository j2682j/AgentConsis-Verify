import unittest

from tools.search_result_builder.next_hop_query.retrieval_recovery_policy import (
    RetrievalRecoveryPolicy,
)


class RetrievalRecoveryPolicyTests(unittest.TestCase):
    def test_only_one_plain_query_expansion_is_allowed(self) -> None:
        policy = RetrievalRecoveryPolicy(max_expand_attempts=1)
        corpus = [{"id": f"D{index}"} for index in range(80)]
        attempted: set[str] = set()

        first = policy.decide(
            relation_plan=None,
            corpus_documents=corpus,
            attempted=attempted,
            top_k=10,
            candidate_pool_size=20,
            original_question="original question",
        )
        self.assertEqual(first.action, "expand_retrieval")
        attempted.add(first.fingerprint)

        second = policy.decide(
            relation_plan=None,
            corpus_documents=corpus,
            attempted=attempted,
            top_k=first.top_k,
            candidate_pool_size=first.candidate_pool_size,
            original_question="original question",
        )
        self.assertEqual(second.action, "stop")

    def test_bridge_terms_enable_second_hop_query(self) -> None:
        policy = RetrievalRecoveryPolicy(max_expand_attempts=0)
        attempted: set[str] = set()

        decision = policy.decide(
            relation_plan=None,
            corpus_documents=[],
            attempted=attempted,
            top_k=10,
            candidate_pool_size=20,
            original_question="which journal is named for one of Hreidmar's sons?",
            bridge_terms=["Fafnir", "journal of dragon studies"],
            missing_constraints=["emily midkiff", "answer_support:person"],
        )

        self.assertEqual(decision.action, "bridge_query")
        self.assertEqual(len(decision.next_queries), 1)
        query = decision.next_queries[0].casefold()
        self.assertIn("fafnir", query)
        # Colon-tagged internal markers may not leak into the web query.
        self.assertNotIn("answer_support", query)
        self.assertIn("emily midkiff", query)
        attempted.add(decision.fingerprint)

        repeat = policy.decide(
            relation_plan=None,
            corpus_documents=[],
            attempted=attempted,
            top_k=10,
            candidate_pool_size=20,
            original_question="which journal is named for one of Hreidmar's sons?",
            bridge_terms=["Fafnir", "journal of dragon studies"],
            missing_constraints=["emily midkiff"],
        )
        self.assertEqual(repeat.action, "stop")

    def test_no_bridge_terms_still_stops(self) -> None:
        policy = RetrievalRecoveryPolicy(max_expand_attempts=0)
        decision = policy.decide(
            relation_plan=None,
            corpus_documents=[],
            attempted=set(),
            top_k=10,
            candidate_pool_size=20,
            original_question="original question",
            missing_constraints=["only missing, no bridges"],
        )
        self.assertEqual(decision.action, "stop")


if __name__ == "__main__":
    unittest.main()
