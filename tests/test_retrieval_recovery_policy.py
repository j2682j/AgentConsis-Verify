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


if __name__ == "__main__":
    unittest.main()
