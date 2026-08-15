from __future__ import annotations

import unittest

from tools.search_result_builder.query import SearchQueryRequest, SourceRequirement
from tools.search_result_builder.source_acquisition import SourceAcquisitionRouter


class FakeSearchTool:
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def run(self, parameters):
        self.calls.append(dict(parameters))
        return {
            "backend": "fake-search",
            "results": list(self.results),
            "notices": [],
        }


class FakeVideoTool:
    def __init__(self, *, ok=True):
        self.calls = []
        self.ok = ok

    @staticmethod
    def extract_url(text):
        for item in str(text or "").split():
            if "youtube.com/watch" in item:
                return item
        return ""

    def run(self, parameters):
        self.calls.append(dict(parameters))
        if not self.ok:
            return {"ok": False, "error_message": "video failed"}
        return {
            "ok": True,
            "output_text": "Frame evidence contains the requested object.",
            "raw_result": {"title": "Test video"},
        }


class FakeTranscriptTool(FakeVideoTool):
    def run(self, parameters):
        self.calls.append(dict(parameters))
        return {
            "ok": True,
            "output_text": "Video transcript source: youtube captions\n[00:12] target phrase",
            "raw_result": {"title": "Transcript"},
        }


class SourceAcquisitionRouterTests(unittest.TestCase):
    def test_invalid_source_values_fall_back_to_web_search(self):
        request = SearchQueryRequest.from_dict(
            {
                "query": "target fact",
                "source_kind": "unknown_backend",
                "access_mode": "magic",
                "source_hint": "",
            }
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.source_requirement.source_kind, "web")
        self.assertEqual(request.source_requirement.access_mode, "search")

    def test_direct_fetch_without_url_or_hint_is_repaired_to_search(self):
        request = SearchQueryRequest.from_dict(
            {
                "query": "target catalog",
                "source_kind": "collection",
                "access_mode": "direct_fetch",
                "source_hint": "",
            }
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.source_requirement.access_mode, "search")

    def test_web_query_uses_search_and_preserves_source_metadata(self):
        search = FakeSearchTool(
            [
                {
                    "title": "Paper",
                    "url": "https://example.org/paper",
                    "content": "Publication metadata",
                }
            ]
        )
        router = SourceAcquisitionRouter(search_tool=search, video_tool=FakeVideoTool())
        request = SearchQueryRequest(
            query="paper publication metadata",
            source_requirement=SourceRequirement(
                source_kind="academic",
                access_mode="search",
                source_hint="example.org",
                required_content="pdf_text",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="Who wrote the paper?",
            max_results=3,
        )

        self.assertEqual(len(search.calls), 1)
        self.assertEqual(sources[0].source_kind, "academic")
        self.assertEqual(sources[0].source_hint, "example.org")
        self.assertEqual(sources[0].required_content, "pdf_text")
        self.assertEqual(traces[0].required_content, "pdf_text")
        self.assertEqual(traces[0].actual_acquirer, "fake-search")

    def test_direct_browser_request_skips_search(self):
        search = FakeSearchTool()
        router = SourceAcquisitionRouter(search_tool=search, video_tool=FakeVideoTool())
        request = SearchQueryRequest(
            query="https://catalog.example.org/search",
            source_requirement=SourceRequirement(
                source_kind="collection",
                access_mode="browser",
                source_hint="catalog.example.org",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="Find the matching record.",
            max_results=3,
        )

        self.assertEqual(search.calls, [])
        self.assertEqual(sources[0].access_mode, "browser")
        self.assertTrue(sources[0].should_fetch_full_page)
        self.assertEqual(traces[0].actual_acquirer, "playwright")

    def test_direct_video_url_uses_video_tool(self):
        search = FakeSearchTool()
        video = FakeVideoTool()
        router = SourceAcquisitionRouter(search_tool=search, video_tool=video)
        request = SearchQueryRequest(
            query="https://youtube.com/watch?v=abcdefghijk",
            source_requirement=SourceRequirement(
                source_kind="video",
                access_mode="direct_fetch",
                source_hint="youtube.com",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="What appears in the video?",
            max_results=3,
        )

        self.assertEqual(search.calls, [])
        self.assertEqual(len(video.calls), 1)
        self.assertTrue(sources[0].fetched)
        self.assertIn("Frame evidence", sources[0].raw_content)
        self.assertEqual(traces[0].actual_acquirer, "video_evidence")

    def test_duplicate_direct_video_url_is_acquired_once(self):
        search = FakeSearchTool()
        video = FakeVideoTool()
        router = SourceAcquisitionRouter(search_tool=search, video_tool=video)
        requirement = SourceRequirement(
            source_kind="video",
            access_mode="direct_fetch",
            source_hint="https://youtube.com/watch?v=abcdefghijk",
        )
        requests = [
            SearchQueryRequest(query="first video query", source_requirement=requirement),
            SearchQueryRequest(query="second video query", source_requirement=requirement),
        ]

        sources, traces = router.acquire_many(
            requests,
            question="What appears in the video?",
            max_results=3,
        )

        self.assertEqual(len(video.calls), 1)
        self.assertEqual(len(sources), 1)
        self.assertEqual(traces[1].actual_acquirer, "duplicate_direct_source_skipped")

    def test_video_failure_falls_back_to_search_results(self):
        search = FakeSearchTool(
            [
                {
                    "title": "Video",
                    "url": "https://youtube.com/watch?v=abcdefghijk",
                    "content": "Video search result",
                }
            ]
        )
        video = FakeVideoTool(ok=False)
        router = SourceAcquisitionRouter(search_tool=search, video_tool=video)
        request = SearchQueryRequest(
            query="target video",
            source_requirement=SourceRequirement(
                source_kind="video",
                access_mode="search",
                source_hint="youtube.com",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="What appears in the video?",
            max_results=3,
        )

        self.assertEqual(len(search.calls), 1)
        self.assertEqual(len(video.calls), 1)
        self.assertEqual(len(sources), 1)
        self.assertTrue(traces[0].fallback_used)
        self.assertIn("video failed", traces[0].notices)

    def test_transcript_requirement_uses_transcript_acquirer(self):
        search = FakeSearchTool()
        video = FakeVideoTool()
        transcript = FakeTranscriptTool()
        router = SourceAcquisitionRouter(
            search_tool=search,
            video_tool=video,
            transcript_tool=transcript,
        )
        request = SearchQueryRequest(
            query="https://youtube.com/watch?v=abcdefghijk",
            source_requirement=SourceRequirement(
                source_kind="video",
                access_mode="direct_fetch",
                source_hint="youtube.com",
                required_content="transcript",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="What is said at 00:12?",
            max_results=3,
        )

        self.assertEqual(video.calls, [])
        self.assertEqual(len(transcript.calls), 1)
        self.assertEqual(traces[0].actual_acquirer, "video_transcript")
        self.assertEqual(sources[0].required_content, "transcript")
        self.assertTrue(sources[0].requirement_met)

    def test_a_string_payload_falls_back_instead_of_raising(self):
        """The video tools do not always return a mapping.

        Every fake above returns a dict, so the branch that reads
        `payload.get("ok")` was never given anything else. In production the
        tool returns a plain string on some failure paths, and the resulting
        AttributeError was not caught here -- it escaped the router and killed
        the entire search evidence build. Task 034 produced `search_used=False`
        and zero facts in every recorded run, level1_final_16 and _18 alike,
        for exactly this reason.
        """

        class _StringReturningTool(FakeVideoTool):
            def run(self, parameters):
                self.calls.append(dict(parameters))
                return "transcript unavailable"

        search = FakeSearchTool(
            [
                {
                    "title": "Video",
                    "url": "https://youtube.com/watch?v=abcdefghijk",
                    "content": "Video search result",
                }
            ]
        )
        video = _StringReturningTool()
        router = SourceAcquisitionRouter(search_tool=search, video_tool=video)
        request = SearchQueryRequest(
            query="target video",
            source_requirement=SourceRequirement(
                source_kind="video",
                access_mode="search",
                source_hint="youtube.com",
            ),
        )

        sources, traces = router.acquire_many(
            [request],
            question="What appears in the video?",
            max_results=3,
        )

        self.assertEqual(len(video.calls), 1)
        self.assertEqual(len(sources), 1)
        self.assertTrue(traces[0].fallback_used)
        self.assertIn("transcript unavailable", traces[0].notices)


class AcquisitionFailsOpenTest(unittest.TestCase):
    """One source raising must not cost the task every other source.

    Both defects found in this router escaped a single request and unwound the
    whole of `RetrievalControl.run`, so the handler above it recorded an empty
    `search_result` and the task ran with no prepared evidence while the
    benchmark still reported a normal score. Task 034 produced zero facts in
    every recorded run because of one `payload.get` on a string.
    """

    def _requests(self) -> list[SearchQueryRequest]:
        return [
            SearchQueryRequest(
                query="first query",
                source_requirement=SourceRequirement(
                    source_kind="web", access_mode="search"
                ),
            ),
            SearchQueryRequest(
                query="second query",
                source_requirement=SourceRequirement(
                    source_kind="web", access_mode="search"
                ),
            ),
        ]

    def _router_that_fails_the_first_request(self, error: Exception):
        """`_search_sources` already guards the search tool, so the failure is
        injected at `acquire` -- which is where both real defects escaped from,
        one in `_acquire_video`'s payload handling and one below it in corpus
        enrichment. What is under test is that `acquire_many` isolates it.
        """

        search = FakeSearchTool(
            [
                {
                    "title": "Second",
                    "url": "https://example.org/second",
                    "content": "The surviving source.",
                }
            ]
        )
        router = SourceAcquisitionRouter(search_tool=search, video_tool=FakeVideoTool())
        original = router.acquire
        calls = {"n": 0}

        def failing_acquire(request, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise error
            return original(request, **kwargs)

        router.acquire = failing_acquire
        return router, calls

    def test_a_raising_source_drops_only_its_own_branch(self):
        router, calls = self._router_that_fails_the_first_request(
            TypeError("'<' not supported between instances of 'dict' and 'dict'")
        )

        sources, traces = router.acquire_many(
            self._requests(), question="anything", max_results=3
        )

        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].actual_acquirer, "acquisition_failed")
        self.assertTrue(traces[0].fallback_used)
        self.assertIn("TypeError", traces[0].notices[0])
        # The point of the change: the second source still arrives.
        self.assertEqual(len(sources), 1)
        self.assertIn("surviving", sources[0].snippet)

    def test_the_failure_keeps_its_traceback(self):
        """Recovering the first two defects meant instrumenting and rerunning."""

        router, _calls = self._router_that_fails_the_first_request(
            ValueError("The truth value of an array with more than one element is ambiguous")
        )

        _sources, traces = router.acquire_many(
            self._requests()[:1], question="anything", max_results=3
        )

        self.assertIn("ValueError", traces[0].error_traceback)
        self.assertIn("ambiguous", traces[0].error_traceback)


if __name__ == "__main__":
    unittest.main()
