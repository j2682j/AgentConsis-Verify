from __future__ import annotations

import unittest
from threading import Lock
import time
from types import SimpleNamespace

from tools.search_result_builder.config import EvidenceItem
from tools.search_result_builder.evidence_searcher import EvidenceSearcher
from tools.search_result_builder.next_hop_query.retrieval_controller import RetrievalDecision
from tools.search_result_builder.source_analyze import SourceUsefulnessResult


class FakeQueryPlanner:
    def plan(self, *, question: str, max_queries: int) -> dict:
        del max_queries
        return {"queries": [question], "salient_spans": ["target"]}


class FakeToolManager:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def execute_tool(self, tool_name, parameters, *, agent_id, stage):
        del tool_name, agent_id, stage
        query = parameters["input"]
        with self.lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(0.03)
        with self.lock:
            self.active_calls -= 1
        return {
            "ok": True,
            "raw_result": {
                "results": [
                    {
                        "title": f"Title for {query}",
                        "url": f"https://example.com/{query.replace(' ', '-')}",
                        "content": f"Evidence returned for {query}.",
                    }
                ]
            },
        }


class FakeSourceAnalysis:
    def __init__(
        self,
        *,
        repeat_evidence: bool = False,
        duplicate_follow_up: bool = False,
    ) -> None:
        self.repeat_evidence = repeat_evidence
        self.duplicate_follow_up = duplicate_follow_up
        self.calls = 0
        self.last_blocked_sources = []
        self.last_evidence_items = []
        self.last_diagnostics = {}

    def build(self, **kwargs):
        self.calls += 1
        sources = kwargs["sources"]
        if self.repeat_evidence:
            source = sources[0]
            self.last_evidence_items = [
                EvidenceItem(
                    evidence_id="E1",
                    source_id=source.source_id,
                    query_id=source.query_id,
                    title=source.title,
                    text="Repeated evidence",
                )
            ]
        elif self.calls == 1:
            source = sources[0]
            self.last_evidence_items = [
                EvidenceItem(
                    evidence_id=f"E{index}",
                    source_id=source.source_id,
                    query_id=source.query_id,
                    title=source.title,
                    text=f"Initial evidence {index}",
                )
                for index in range(1, 3)
            ]
        else:
            self.last_evidence_items = [
                EvidenceItem(
                    evidence_id=f"E{index}",
                    source_id=source.source_id,
                    query_id=source.query_id,
                    title=source.title,
                    text=(
                        "Duplicate follow-up evidence"
                        if self.duplicate_follow_up
                        else f"Follow-up evidence {index}"
                    ),
                )
                for index, source in enumerate(sources, 1)
            ]
        self.last_blocked_sources = []
        self.last_diagnostics = {"call": self.calls}
        return SourceUsefulnessResult(sources=sources, fetched_pages=0)


class FakeRetrievalController:
    def __init__(self, *, sufficient_at: int) -> None:
        self.sufficient_at = sufficient_at
        self.calls = 0

    def assess(self, *, evidence_items, **kwargs):
        del kwargs
        self.calls += 1
        sufficient = len(evidence_items) >= self.sufficient_at
        return RetrievalDecision(
            need_next_hop=not sufficient,
            reason="sufficient_evidence" if sufficient else "insufficient_evidence",
            confidence=min(1.0, len(evidence_items) / self.sufficient_at),
        )

    def rank_evidence(self, *, question, evidence_items):
        del question
        ranked = list(reversed(evidence_items))
        diagnostics = []
        for index, item in enumerate(ranked):
            item.evidence_quality = round(0.9 - index * 0.1, 2)
            diagnostics.append(
                {
                    "query_id": item.query_id,
                    "source_id": item.source_id,
                    "score": item.evidence_quality,
                }
            )
        return ranked, diagnostics


class FakeRagFilter:
    def build_query(self, *, question, evidence_items):
        del question
        return SimpleNamespace(query=f"follow up {evidence_items[0].text}")


class FakeRenderer:
    def render(self, output):
        return "\n".join(item.text for item in output.evidence_items)


class EvidenceSearcherNextHopTests(unittest.TestCase):
    def build_searcher(self, *, source_analysis, controller, max_queries=2):
        tool_manager = FakeToolManager()
        searcher = EvidenceSearcher(
            tool_manager=tool_manager,
            query_planner=FakeQueryPlanner(),
            source_analysis=source_analysis,
            retrieval_controller=controller,
            rag_filter=FakeRagFilter(),
            renderer=FakeRenderer(),
            max_parallel_next_hop_queries=max_queries,
        )
        return searcher, tool_manager

    def test_searches_h1_and_h2_in_parallel_then_reassesses_once(self):
        controller = FakeRetrievalController(sufficient_at=4)
        searcher, tool_manager = self.build_searcher(
            source_analysis=FakeSourceAnalysis(),
            controller=controller,
        )

        output = searcher.search("original question", max_full_page_results=0)
        diagnostics = output.diagnostics["evidence_driven_search"]

        self.assertEqual(controller.calls, 2)
        self.assertEqual(
            output.diagnostics["sufficiency_method"]["signals"],
            [
                "spacy_ner_coverage",
                "encoder_semantic_relevance",
                "constraint_coverage",
            ],
        )
        self.assertEqual(diagnostics["mode"], "parallel_filter_queries")
        self.assertEqual(diagnostics["parallel_query_count"], 2)
        self.assertEqual(diagnostics["stop_reason"], "sufficient_evidence")
        self.assertEqual([query.query_id for query in output.queries], ["Q1", "H1", "H2"])
        self.assertEqual(tool_manager.max_active_calls, 2)
        self.assertEqual(len(output.evidence_items), 4)
        self.assertEqual(
            [item.query_id for item in output.evidence_items[2:]],
            ["H2", "H1"],
        )
        self.assertEqual(
            diagnostics["cross_hop_consolidation"]["method"],
            "ngram_dedup_then_three_signal_rerank",
        )
        self.assertEqual(
            diagnostics["cross_hop_consolidation"]["ranking"][0]["query_id"],
            "H2",
        )

    def test_stops_when_follow_up_adds_no_new_evidence(self):
        controller = FakeRetrievalController(sufficient_at=3)
        searcher, _ = self.build_searcher(
            source_analysis=FakeSourceAnalysis(repeat_evidence=True),
            controller=controller,
            max_queries=2,
        )

        output = searcher.search("original question", max_full_page_results=0)
        diagnostics = output.diagnostics["evidence_driven_search"]

        self.assertEqual(controller.calls, 2)
        self.assertEqual(diagnostics["parallel_query_count"], 1)
        self.assertEqual(diagnostics["stop_reason"], "no_new_evidence")
        self.assertEqual(diagnostics["evidence_gain"], 0)
        self.assertEqual(len(output.evidence_items), 1)

    def test_deduplicates_equivalent_h1_h2_evidence_before_reranking(self):
        controller = FakeRetrievalController(sufficient_at=3)
        searcher, _ = self.build_searcher(
            source_analysis=FakeSourceAnalysis(duplicate_follow_up=True),
            controller=controller,
        )

        output = searcher.search("original question", max_full_page_results=0)
        diagnostics = output.diagnostics["evidence_driven_search"]
        consolidation = diagnostics["cross_hop_consolidation"]

        self.assertEqual(diagnostics["parallel_query_count"], 2)
        self.assertEqual(consolidation["input_count"], 2)
        self.assertEqual(consolidation["deduplicated_count"], 1)
        self.assertEqual(consolidation["removed_count"], 1)
        self.assertEqual(len(output.evidence_items), 3)
        self.assertGreater(output.evidence_items[-1].evidence_quality, 0)


if __name__ == "__main__":
    unittest.main()
