from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
    PageContentFetcher,
    fetch_page_content_result,
)


class FakeResponse:
    def __init__(
        self,
        text: str,
        *,
        content_type: str = "text/html; charset=utf-8",
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class PageContentFetcherTests(unittest.TestCase):
    def test_fetch_page_content_extracts_body_text_without_navigation_noise(self):
        html = """
        <html>
          <body>
            <nav>Home Search Login</nav>
            <script>window.secret = "noise";</script>
            <article>
              <h1>Useful Article</h1>
              <p>This paragraph is the useful evidence about the target answer.</p>
              <p>It should remain available for downstream chunking.</p>
            </article>
            <footer>Cookie policy and unrelated footer links.</footer>
          </body>
        </html>
        """

        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(html),
        ) as mock_get:
            result = fetch_page_content_result(
                "https://example.com/article",
                max_tokens=200,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Useful Article", result.content)
        self.assertIn("target answer", result.content)
        self.assertNotIn("Home Search Login", result.content)
        self.assertNotIn("Cookie policy", result.content)
        self.assertTrue(
            result.method.startswith(
                ("beautifulsoup", "trafilatura", "readability", "structured_html")
            )
        )
        mock_get.assert_called_once()
        self.assertIn("User-Agent", mock_get.call_args.kwargs["headers"])

    def test_page_content_fetcher_writes_raw_content_and_method_reason(self):
        html = """
        <html>
          <body>
            <main>
              <p>This is a sufficiently long fetched article body with useful
              factual content for the downstream retrieval corpus builder.</p>
              <p>It has enough characters to pass the minimum content check.</p>
            </main>
          </body>
        </html>
        """
        source = SearchSourceCandidate(
            source_id="S1",
            query_id="Q1",
            title="Example",
            url="https://example.com/article",
            snippet="short snippet",
            should_fetch_full_page=True,
        )

        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(html),
        ):
            fetched = PageContentFetcher(
                max_workers=1,
                min_content_chars=80,
            ).fetch_sources([source], max_pages=1)

        self.assertEqual(fetched, 1)
        self.assertTrue(source.fetched)
        self.assertFalse(source.should_fetch_full_page)
        self.assertIn("factual content", source.raw_content)
        self.assertIn("full_page_fetched", source.filter_reasons)
        self.assertTrue(
            any(reason.startswith("fetch_method:") for reason in source.filter_reasons)
        )
        self.assertTrue(
            any(reason.startswith("fetch_quality:") for reason in source.filter_reasons)
        )
        self.assertTrue(
            any(reason.startswith("fetch_trace:") for reason in source.filter_reasons)
        )

    def test_fetch_page_content_includes_metadata_tables_and_captions(self):
        html = """
        <html>
          <head>
            <title>Target Page</title>
            <meta name="description" content="Metadata answer description">
            <script type="application/ld+json">
              {"headline": "Structured headline", "description": "Structured answer detail"}
            </script>
          </head>
          <body>
            <main>
              <h1>Main Heading</h1>
              <p>The article body contains the central answer sentence.</p>
              <table>
                <tr><th>Name</th><th>Value</th></tr>
                <tr><td>Target</td><td>42</td></tr>
              </table>
              <figure>
                <img src="x.png" alt="Alt text answer clue">
                <figcaption>Caption answer clue</figcaption>
              </figure>
            </main>
          </body>
        </html>
        """

        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(html),
        ):
            result = fetch_page_content_result(
                "https://example.com/structured",
                max_tokens=500,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Metadata answer description", result.content)
        self.assertIn("Structured answer detail", result.content)
        self.assertIn("Main Heading", result.content)
        self.assertIn("Target | 42", result.content)
        self.assertIn("Caption answer clue", result.content)
        self.assertIn("Alt text answer clue", result.content)
        self.assertEqual(result.quality_status, "ok")
        self.assertTrue(any(item.startswith("tables:") for item in result.trace))

    def test_js_shell_uses_playwright_fallback_when_static_text_is_sparse(self):
        html = """
        <html>
          <body>
            <div id="root">Loading...</div>
            <script>window.__APP_STATE__ = {"ready": false};</script>
          </body>
        </html>
        """

        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(html),
        ), patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher._fetch_with_playwright",
            return_value=("Rendered answer content from the browser runtime.", "playwright_body_text"),
        ) as mock_browser:
            result = fetch_page_content_result(
                "https://example.com/app",
                max_tokens=200,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.method, "playwright_body_text")
        self.assertIn("Rendered answer content", result.content)
        mock_browser.assert_called_once_with("https://example.com/app")

    def test_http_403_can_use_playwright_fallback(self):
        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(
                "Access denied",
                status_code=403,
            ),
        ), patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher._fetch_with_playwright",
            return_value=("Rendered page text with enough useful evidence after browser fallback.", "playwright_body_text"),
        ) as mock_browser:
            result = fetch_page_content_result(
                "https://example.com/blocked",
                max_tokens=200,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status_code, 403)
        self.assertEqual(result.method, "playwright_body_text")
        self.assertIn("playwright_fallback:http_error", result.trace)
        mock_browser.assert_called_once_with("https://example.com/blocked")

    def test_regular_html_does_not_use_playwright_fallback(self):
        html = """
        <html>
          <body>
            <article>
              <p>This static article contains enough useful evidence for the
              normal extraction path, so browser rendering is unnecessary.</p>
              <p>The content is available directly from the HTTP response.</p>
            </article>
          </body>
        </html>
        """

        with patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher.requests.get",
            return_value=FakeResponse(html),
        ), patch(
            "tools.search_result_builder.source_analyze.seer.page_content_fetcher._fetch_with_playwright",
        ) as mock_browser:
            result = fetch_page_content_result(
                "https://example.com/static",
                max_tokens=200,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.method, "playwright_body_text")
        self.assertIn("static article", result.content)
        mock_browser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
