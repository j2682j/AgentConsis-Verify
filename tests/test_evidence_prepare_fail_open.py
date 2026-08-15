"""Injected failures must cost their own branch and nothing more.

Three defects reached Evidence Prepare during this investigation and each one
emptied a whole task's search evidence: `not vectors` on a numpy array in corpus
enrichment, `payload.get` on a string return in video acquisition, and a `<`
between two dicts that has never reproduced. `level1_final_18` scored 20/53 --
above its baseline -- with the prepared pipeline dead on 25 of 53 tasks, because
Stage 1's own search compensated and nothing in the record said otherwise.

The fail-open work was shipped and then, in `level1_final_21`, never exercised:
zero acquisition failures, zero enrichment failures, every task `complete`. Code
that only runs when something breaks cannot be validated by a run where nothing
breaks, so the failures are injected here.

Each case asserts the same three things: the failing branch is dropped, what
survives is still delivered, and the status says `partial_failure` rather than
`failed`.
"""

from __future__ import annotations

import unittest

from core.evidence_runner import EvidenceRunner
from tools.search_result_builder.query import SearchQueryRequest, SourceRequirement
from tools.search_result_builder.source_acquisition import SourceAcquisitionRouter

STATUS = EvidenceRunner._evidence_pipeline_status
PREPARE_STATUS = EvidenceRunner._evidence_prepare_status


class _Search:
    def __init__(self, results):
        self.results = results

    def run(self, parameters):
        return {"backend": "fake", "results": list(self.results), "notices": []}


class _Video:
    @staticmethod
    def extract_url(text):
        return ""

    def run(self, parameters):
        return {"ok": False, "error_message": "unused"}


def _requests(count: int) -> list[SearchQueryRequest]:
    return [
        SearchQueryRequest(
            query=f"query {index}",
            source_requirement=SourceRequirement(source_kind="web", access_mode="search"),
        )
        for index in range(1, count + 1)
    ]


class SingleAcquisitionFailureTest(unittest.TestCase):
    def test_one_source_fails_and_the_others_still_arrive(self) -> None:
        search = _Search(
            [{"title": "Kept", "url": "https://example.org/a", "content": "surviving text"}]
        )
        router = SourceAcquisitionRouter(search_tool=search, video_tool=_Video())
        original = router.acquire
        calls = {"n": 0}

        def failing(request, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise TypeError("'<' not supported between instances of 'dict' and 'dict'")
            return original(request, **kwargs)

        router.acquire = failing
        sources, traces = router.acquire_many(
            _requests(3), question="anything", max_results=3
        )

        self.assertEqual(calls["n"], 3)
        failed = [t for t in traces if t.actual_acquirer == "acquisition_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("TypeError", failed[0].notices[0])
        self.assertIn("source_acquisition.py", failed[0].error_traceback)
        # Two branches survived; the failure cost only its own.
        self.assertEqual(len(sources), 2)

    def test_the_status_reads_partial_failure_not_failed(self) -> None:
        output = {
            "web_searches": [
                {"actual_acquirer": "acquisition_failed"},
                {"actual_acquirer": "search"},
            ],
            "diagnostics": {},
        }

        self.assertEqual(
            STATUS(
                output_dict=output,
                evidence_items=[{"relaxed": False}],
                references=[{"reference_id": "R1"}],
                strict_evidence_count=1,
                source_count=2,
            ),
            ("partial_failure", "strict"),
        )


class EnrichmentFailureTest(unittest.TestCase):
    def test_the_corpus_and_references_outlive_a_failed_enrichment(self) -> None:
        output = {
            "web_searches": [{"actual_acquirer": "search"}],
            "diagnostics": {
                "collection_link_enrichment": {
                    "added_record_count": 0,
                    "error_traceback": (
                        "ValueError: The truth value of an array with more than "
                        "one element is ambiguous"
                    ),
                }
            },
        }

        pipeline, evidence = STATUS(
            output_dict=output,
            evidence_items=[],
            references=[{"reference_id": "R1"}, {"reference_id": "R2"}],
            strict_evidence_count=0,
            source_count=5,
        )

        self.assertEqual(pipeline, "partial_failure")
        self.assertEqual(evidence, "unverified_only")


class TopLevelFailureTest(unittest.TestCase):
    def test_a_search_build_that_raised_reads_failed(self) -> None:
        bundle = {
            "tool_usage": [
                {
                    "tool_name": "search",
                    "ok": False,
                    "error": "boom",
                    "raw_result": {
                        "diagnostics": {
                            "pipeline_status": "failed",
                            "evidence_status": "empty",
                            "failure_scope": "search_evidence_build",
                        }
                    },
                }
            ]
        }

        self.assertEqual(PREPARE_STATUS(bundle), ("failed", "empty"))

    def test_a_search_entry_with_no_diagnostics_still_reads_failed(self) -> None:
        bundle = {"tool_usage": [{"tool_name": "search", "ok": False, "raw_result": None}]}

        self.assertEqual(PREPARE_STATUS(bundle), ("failed", "empty"))


class NotRunTest(unittest.TestCase):
    def test_a_task_routing_never_sent_to_search_is_not_run(self) -> None:
        """24 of 53 tasks take this path, and used to carry no status at all."""

        bundle = {
            "tool_usage": [
                {"tool_name": "attachment_reader", "ok": True},
                {"tool_name": "deterministic_handler_router", "ok": True},
            ]
        }

        self.assertEqual(PREPARE_STATUS(bundle), ("not_run", "not_applicable"))

    def test_not_run_is_distinct_from_finding_nothing(self) -> None:
        searched_and_empty = {
            "tool_usage": [
                {
                    "tool_name": "search",
                    "ok": True,
                    "raw_result": {
                        "diagnostics": {
                            "pipeline_status": "complete",
                            "evidence_status": "empty",
                        }
                    },
                }
            ]
        }

        self.assertEqual(PREPARE_STATUS(searched_and_empty), ("complete", "empty"))
        self.assertNotEqual(
            PREPARE_STATUS(searched_and_empty), PREPARE_STATUS({"tool_usage": []})
        )


if __name__ == "__main__":
    unittest.main()
