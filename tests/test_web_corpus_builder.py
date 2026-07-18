from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.corpus import (
    DocumentChunker,
    DocumentCleaner,
    WebCorpusBuilder,
)
from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
    PageFetchResult,
)


class WebCorpusBuilderTests(unittest.TestCase):
    def test_document_cleaner_removes_html_shell_and_preserves_chinese(self):
        cleaner = DocumentCleaner()
        content = """
        <html>
          <head><style>.hidden { display:none }</style></head>
          <body>
            <nav>首頁 登入 訂閱</nav>
            <main>
              <h1>臺北市人口</h1>
              <p>截至2025年，臺北市人口為約250萬人。</p>
              <p>截至2025年，臺北市人口為約250萬人。</p>
            </main>
            <footer>All rights reserved.</footer>
          </body>
        </html>
        """

        cleaned = cleaner.clean(content)

        self.assertIn("臺北市人口", cleaned)
        self.assertIn("截至2025年", cleaned)
        self.assertNotIn("首頁 登入 訂閱", cleaned)
        self.assertNotIn("All rights reserved", cleaned)
        self.assertEqual(cleaned.count("截至2025年"), 1)

    def test_document_chunker_respects_size_and_order(self):
        chunker = DocumentChunker(
            max_chars=120,
            overlap_chars=20,
            min_chars=20,
        )
        text = "\n\n".join(
            [
                "第一段說明臺北市人口統計資料與資料來源。" * 4,
                "第二段說明不同年度的人口變化趨勢。" * 4,
            ]
        )

        chunks = chunker.chunk(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertIn("第一段", chunks[0])
        self.assertTrue(any("第二段" in chunk for chunk in chunks))

    def test_builder_fetches_cleans_deduplicates_and_exports_exact_schema(self):
        fetch_calls = []

        def fake_fetcher(url: str, *, max_tokens: int):
            fetch_calls.append((url, max_tokens))
            return """
            <html><body><main>
            <h1>臺北市人口</h1>
            <p>截至2025年，臺北市人口為約250萬人，資料來自官方統計。</p>
            </main></body></html>
            """

        builder = WebCorpusBuilder(
            chunker=DocumentChunker(
                max_chars=300,
                overlap_chars=0,
                min_chars=10,
            ),
            page_fetcher=fake_fetcher,
        )
        sources = [
            SearchSourceCandidate(
                source_id="S1",
                query_id="Q1",
                title="臺北市人口",
                url="https://example.com/taipei",
                snippet="搜尋摘要",
            ),
            {
                "title": "重複頁面",
                "url": "https://example.com/duplicate",
                "raw_content": (
                    "臺北市人口\n"
                    "截至2025年，臺北市人口為約250萬人，資料來自官方統計。"
                ),
            },
            {
                "title": "Blocked",
                "url": "https://example.com/blocked",
                "raw_content": "不應輸出的內容。",
                "blocked": True,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "corpus.jsonl"
            count = builder.build_jsonl(
                sources,
                output_path,
                retrieved_at="2026-06-25",
                fetch_missing=True,
                max_pages_to_fetch=1,
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(fetch_calls, [("https://example.com/taipei", 8000)])
        self.assertEqual(count, 1)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(
            list(payload)[:5],
            ["id", "title", "text", "url", "retrieved_at"],
        )
        self.assertEqual(payload["record_type"], "passage")
        self.assertIn("record_id", payload)
        self.assertIn("content_url", payload)
        self.assertEqual(payload["id"], "page-001-000")
        self.assertEqual(payload["title"], "臺北市人口")
        self.assertEqual(payload["url"], "https://example.com/taipei")
        self.assertEqual(payload["retrieved_at"], "2026-06-25")
        self.assertIn("約250萬人", payload["text"])

    def test_builder_uses_snippet_when_fetch_is_disabled(self):
        builder = WebCorpusBuilder(
            chunker=DocumentChunker(
                max_chars=300,
                overlap_chars=0,
                min_chars=5,
            )
        )

        records = builder.build_records(
            [
                {
                    "title": "人口摘要",
                    "url": "https://example.com/summary",
                    "snippet": "這是一段可建立 corpus 的搜尋摘要。",
                }
            ],
            retrieved_at="2026-06-25",
            fetch_missing=False,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "這是一段可建立 corpus 的搜尋摘要。")


    def test_omitted_chunks_cannot_be_marked_as_a_complete_document(self):
        long_text = "First complete sentence about optics. " * 20

        def fetcher(url: str, *, max_tokens: int):
            del max_tokens
            return PageFetchResult(
                content=long_text,
                method="test",
                quality_status="ok",
                is_complete=True,
                original_char_count=len(long_text),
                final_url=url,
            )

        builder = WebCorpusBuilder(
            chunker=DocumentChunker(max_chars=120, overlap_chars=0, min_chars=10),
            page_fetcher=fetcher,
        )
        records = builder.build_records(
            [{"title": "Long article", "url": "https://example.com/long"}],
            max_chunks_per_url=1,
        )

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].content_complete)
        self.assertTrue(records[0].content_truncated)


if __name__ == "__main__":
    unittest.main()
