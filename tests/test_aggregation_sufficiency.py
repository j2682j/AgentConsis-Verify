import unittest

from tools.evidence.fact_extraction import (
    AggregationFactDeriver,
    CompletenessContractBuilder,
    EvidenceFact,
    TaskFactStore,
)
from tools.search_result_builder.next_hop_query.goal_completion import (
    GoalCompletionEvaluator,
)


class AggregationSufficiencyTests(unittest.TestCase):
    def _store(self, *, complete: bool = True) -> TaskFactStore:
        store = TaskFactStore()
        for index, title in enumerate(("Album A", "Album B", "Album C"), start=1):
            store.add(
                EvidenceFact(
                    fact_id=f"F{index}",
                    subject="Artist discography",
                    relation="contains studio album",
                    object=title,
                    role="BRIDGE",
                    source_id="discography",
                    grounding_status="grounded",
                )
            )
        store.add_completeness_contract(
            CompletenessContractBuilder().build(
                scope_id="discography",
                source_id="discography",
                source_type="collection",
                expected_units=3,
                processed_units=3,
                complete=complete,
                unit_ids=["F1", "F2", "F3"],
            )
        )
        return store

    def test_count_derivation_requires_complete_grounded_group(self) -> None:
        results = AggregationFactDeriver().derive(
            self._store(),
            question="How many studio albums are listed?",
            answer_requirement="number of studio albums",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, "3")
        self.assertEqual(results[0].operator, "count")
        store = self._store()
        self.assertTrue(store.add_aggregation_derivation(results[0]))
        self.assertEqual(store.verifiable_answer_facts()[0].object, "3")

    def test_incomplete_scope_does_not_produce_count(self) -> None:
        results = AggregationFactDeriver().derive(
            self._store(complete=False),
            question="How many studio albums are listed?",
        )
        self.assertEqual(results, [])

    def test_goal_completion_accepts_verified_aggregate(self) -> None:
        result = GoalCompletionEvaluator().evaluate(
            relation_plan=None,
            documents=[],
            fact_store=self._store(),
            question="How many studio albums are listed?",
            answer_requirement="number of studio albums",
        )
        self.assertTrue(result.sufficient)
        self.assertTrue(result.aggregate_answer_found)
        self.assertFalse(result.direct_answer_found)


if __name__ == "__main__":
    unittest.main()
