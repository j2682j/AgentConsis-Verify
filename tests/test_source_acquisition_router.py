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


if __name__ == "__main__":
    unittest.main()
