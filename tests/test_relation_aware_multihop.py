from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.search_result_builder.corpus import CorpusRecord, TaskCorpusSession
from tools.search_result_builder.next_hop_query import (
    NextHopQueryComposer,
    RelationEvidence,
    RelationEvidenceBinder,
    RelationGoalResolver,
)
from tools.search_result_builder.query import RelationPlan
from tools.search_result_builder.query import SearchIntentPlan
from tools.search_result_builder.query.mask_salience_query import MaskSalienceQueryGenerator
from tools.search_result_builder.retrieval_control import IterativeRetrievalControl
from tools.search_result_builder.source_analyze.rag_labeler import (
    CONTINUE_TAG,
    RAGLabelResult,
)


class RelationAwareMultiHopTests(unittest.TestCase):
    def _plan(self) -> RelationPlan:
        return RelationPlan.from_specs(
            [
                {
                    "subject": "KGOT",
                    "relation": "studio location",
                    "target": "shopping mall name",
                    "source_kind": "web",
                },
                {
                    "subject": "",
                    "relation": "floor area",
                    "target": "size of the mall",
                    "source_kind": "web",
                },
            ]
        )

    def test_grounded_bridge_resolves_goal_and_activates_successor(self) -> None:
        plan = self._plan()
        document = SimpleNamespace(
            document_id="page-001-000",
            title="KGOT",
            text="KGOT broadcasts from studios in the Dimond Center in Anchorage.",
            bridge_contracts=[
                {
                    "goal_id": "G1",
                    "bridge_span": "Dimond Center",
                    "context": "KGOT broadcasts from studios in the Dimond Center in Anchorage.",
                    "document_id": "page-001-000",
                    "next_goal_id": "G2",
                }
            ],
            answer_support_spans=[],
        )
        resolver = RelationGoalResolver()
        binding = RelationEvidenceBinder(resolver=resolver).bind(
            plan=plan,
            documents=[document],
        )
        self.assertEqual([item.object for item in binding.evidence], ["Dimond Center"])

        resolution = resolver.resolve(plan, binding.evidence)
        self.assertEqual(resolution.activated_goal_id, "G2")
        self.assertEqual(resolution.plan.goals[0].resolved_values, ["Dimond Center"])
        self.assertEqual(resolution.plan.active_goal_id, "G2")

    def test_composer_builds_one_branch_per_resolved_object(self) -> None:
        resolver = RelationGoalResolver()
        plan = self._plan()
        first = plan.goals[0].replace(
            state="resolved",
            resolved_values=["Dimond Center", "Alternate Center"],
        )
        second = plan.goals[1].replace(state="active")
        plan = RelationPlan(goals=[first, second], active_goal_id="G2")

        branches = NextHopQueryComposer(
            relation_resolver=resolver
        ).build_relation_requests(
            relation_plan=plan,
            constraints=["Anchorage"],
            max_requests=2,
        )
        self.assertEqual(len(branches), 2)
        self.assertIn("Dimond Center", branches[0].request.query)
        self.assertIn("floor area", branches[0].request.query)
        self.assertEqual(branches[0].request.source_requirement.source_kind, "web")

    def test_composer_rejects_original_or_previously_searched_relation_query(self) -> None:
        plan = self._plan()
        first = plan.goals[0].replace(
            state="resolved",
            resolved_values=["Dimond Center"],
        )
        second = plan.goals[1].replace(state="active")
        plan = RelationPlan(goals=[first, second], active_goal_id="G2")
        baseline = NextHopQueryComposer().build_relation_requests(
            relation_plan=plan,
            answer_requirement="mall size",
        )
        self.assertEqual(len(baseline), 1)
        expected_query = baseline[0].request.query

        original_duplicate = NextHopQueryComposer().build_relation_requests(
            relation_plan=plan,
            answer_requirement="mall size",
            original_question=expected_query,
        )
        searched_duplicate = NextHopQueryComposer().build_relation_requests(
            relation_plan=plan,
            answer_requirement="mall size",
            seen_query_keys={expected_query.casefold()},
        )

        self.assertEqual(original_duplicate, [])
        self.assertEqual(searched_duplicate, [])

    def test_resolver_activates_only_the_next_goal_after_current_goal(self) -> None:
        plan = RelationPlan.from_dict(
            {
                "goals": [
                    {
                        "goal_id": "G1",
                        "subject": "unrelated",
                        "relation": "unused relation",
                        "target": "unused target",
                        "state": "pending",
                    },
                    {
                        "goal_id": "G2",
                        "subject": "KGOT",
                        "relation": "studio location",
                        "target": "mall name",
                        "state": "active",
                    },
                    {
                        "goal_id": "G3",
                        "subject": "",
                        "relation": "floor area",
                        "target": "mall size",
                        "state": "pending",
                    },
                ],
                "active_goal_id": "G2",
            }
        )
        resolution = RelationGoalResolver().resolve(
            plan,
            [
                RelationEvidence(
                    goal_id="G2",
                    subject="KGOT",
                    relation="studio location",
                    object="Dimond Center",
                    context="KGOT has studios in the Dimond Center.",
                    document_id="D1",
                )
            ],
        )

        self.assertEqual(resolution.activated_goal_id, "G3")
        self.assertEqual(resolution.plan.goals[0].state, "pending")
        self.assertEqual(resolution.plan.goals[2].state, "active")

    def test_query_json_parses_source_requests_and_ordered_relation_goals(self) -> None:
        output = MaskSalienceQueryGenerator()._parse_query_json(
            """{
                "queries": [{
                    "query": "KGOT studio location",
                    "source_kind": "web",
                    "access_mode": "search",
                    "source_hint": ""
                }],
                "relation_goals": [
                    {"subject": "KGOT", "relation": "studio location", "target": "mall name", "source_kind": "web"},
                    {"subject": "", "relation": "floor area", "target": "mall size", "source_kind": "web"}
                ]
            }"""
        )
        self.assertEqual(output.query_requests[0].query, "KGOT studio location")
        self.assertEqual(len(output.relation_plan.goals), 2)
        self.assertEqual(output.relation_plan.active_goal_id, "G1")

    def test_relation_plan_keeps_six_goals_and_content_requirements(self) -> None:
        plan = RelationPlan.from_specs(
            [
                {
                    "subject": f"subject-{index}",
                    "relation": f"relation-{index}",
                    "target": f"target-{index}",
                    "source_kind": "academic",
                    "required_content": "pdf_text",
                }
                for index in range(1, 7)
            ]
        )

        self.assertEqual(len(plan.goals), 6)
        self.assertTrue(all(goal.required_content == "pdf_text" for goal in plan.goals))


class _FakeEmbedder:
    def prepare_query_text(self, value):
        return f"query: {value}"

    def prepare_passage_text(self, value):
        return f"passage: {value['title']}: {value['text']}"

    def embed(self, values):
        return np.ones((len(values), 3), dtype="float32")


class _FakeIndexCore:
    is_trained = True

    def __init__(self) -> None:
        self.added = []

    def add(self, values) -> None:
        self.added.extend(values.tolist())


class TaskCorpusSessionTests(unittest.TestCase):
    def test_add_records_updates_jsonl_passage_map_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            corpus_path.write_text(
                '{"id":"page-001-000","title":"old","text":"old text","url":"u","retrieved_at":"2026-01-01"}\n',
                encoding="utf-8",
            )
            retriever = SimpleNamespace(
                passage_map={
                    "page-001-000": {
                        "id": "page-001-000",
                        "title": "old",
                        "text": "old text",
                        "url": "u",
                    }
                },
                embedder=_FakeEmbedder(),
                index=SimpleNamespace(index=_FakeIndexCore(), idx2db=["page-001-000"]),
            )
            session = TaskCorpusSession(
                corpus_path=corpus_path,
                retriever=retriever,
            )
            added = session.add_records(
                [
                    CorpusRecord(
                        id="page-001-000",
                        title="Dimond Center",
                        text="The mall has 728,000 square feet of floor area.",
                        url="https://example.test/mall",
                        retrieved_at="2026-01-01",
                    )
                ]
            )
            self.assertEqual(len(added), 1)
            self.assertTrue(added[0].id.startswith("hop-01-"))
            self.assertIn(added[0].id, retriever.passage_map)
            self.assertEqual(retriever.index.idx2db[-1], added[0].id)
            self.assertEqual(len(retriever.index.index.added), 1)
            self.assertEqual(len(corpus_path.read_text(encoding="utf-8").splitlines()), 2)


class _RoundIndex:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query_vectors, top_k):
        del query_vectors, top_k
        self.calls += 1
        document_id = "D1" if self.calls == 1 else "D2"
        return [([document_id], [0.95])]


class _RoundRetriever:
    model_type = "multilingual-e5-base"

    def __init__(self) -> None:
        self.embedder = _FakeEmbedder()
        self.index = _RoundIndex()
        self.passage_map = {
            "D1": {
                "id": "D1",
                "title": "KGOT studios",
                "text": "KGOT broadcasts from studios in the Dimond Center.",
                "url": "https://example.test/kgot",
            }
        }


class _RoundLabeler:
    def label_texts(self, *, question, texts):
        del question
        return [
            RAGLabelResult(
                label="useful",
                kept_tokens=["Dimond", "Center"],
                metadata={
                    "sequence_tag": CONTINUE_TAG,
                    "continue_probability": 0.95,
                    "terminate_probability": 0.05,
                },
            )
            for _ in texts
        ]


class RelationAwareControlTests(unittest.TestCase):
    def test_resolved_goal_loads_external_branch_and_finishes_next_goal(self) -> None:
        retriever = _RoundRetriever()
        loaded_queries: list[str] = []

        def loader(requests):
            loaded_queries.extend(request.query for request in requests)
            retriever.passage_map["D2"] = {
                "id": "D2",
                "title": "Dimond Center",
                "text": "The Dimond Center has a floor area of 728,000 square feet.",
                "url": "https://example.test/dimond",
            }
            return 1

        controller = IterativeRetrievalControl(
            retriever=retriever,
            labeler=_RoundLabeler(),
            max_iter=3,
            top_k=1,
            external_source_loader=loader,
        )

        def classify(*, round_trace, question, intent_plan):
            del question, intent_plan
            for document in round_trace.documents:
                if document.document_id == "D1":
                    document.bridge_spans = ["Dimond Center"]
                    document.bridge_contracts = [
                        {
                            "goal_id": "G1",
                            "bridge_span": "Dimond Center",
                            "context": "KGOT broadcasts from studios in the Dimond Center.",
                            "document_id": "D1",
                            "next_goal_id": "G2",
                        }
                    ]
                else:
                    document.answer_support_spans = ["728,000 square feet"]
                    document.direct_contracts = [
                        {
                            "goal_id": "G2",
                            "answer_span": "728,000 square feet",
                            "fact_id": "F-area",
                            "subject": "Dimond Center",
                            "relation": "floor area",
                            "object": "728,000 square feet",
                            "grounding_status": "grounded",
                            "context": "The Dimond Center has a floor area of 728,000 square feet.",
                            "document_id": "D2",
                        }
                    ]
                document.valid_for_next_hop = True

        controller._apply_span_role_classification = classify
        plan = SearchIntentPlan(relation_plan=RelationAwareMultiHopTests()._plan())
        result = controller.run("How large is the mall where KGOT has studios?", intent_plan=plan)

        self.assertEqual(result.stop_reason, "goal_completion_sufficient")
        self.assertEqual(len(result.rounds), 2)
        self.assertTrue(loaded_queries)
        self.assertIn("Dimond Center", loaded_queries[0])
        self.assertTrue(result.relation_plan["goals"][1]["resolved_values"])


if __name__ == "__main__":
    unittest.main()
