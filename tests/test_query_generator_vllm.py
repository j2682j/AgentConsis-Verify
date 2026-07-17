from __future__ import annotations

import unittest

from core.llm_client import LLMChatResult
from tools.search_result_builder.query.mask_salience_query import (
    MaskSalienceQueryGenerator,
)
from tools.search_result_builder.query.span_repair import SalientSpan


class FakeLLMClient:
    def __init__(self, provider="vllm"):
        self.calls = []
        self.native_calls = []
        self.provider = provider

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMChatResult(
            content=(
                '{"queries": ['
                '{"query": "Moon minimum perigee", "source_kind": "academic", '
                '"access_mode": "search", "source_hint": "NASA"}, '
                '{"query": "Eliud Kipchoge marathon pace", "source_kind": "video", '
                '"access_mode": "search", "source_hint": "youtube.com"}'
                ']}'
            ),
            prompt_tokens=20,
            completion_tokens=12,
        )

    def ollama_native_chat(self, **kwargs):
        self.native_calls.append(kwargs)
        return LLMChatResult(
            content=(
                '{"queries": ['
                '{"query": "Moon minimum perigee", "source_kind": "academic", '
                '"access_mode": "search", "source_hint": "NASA"}, '
                '{"query": "Eliud Kipchoge marathon pace", "source_kind": "video", '
                '"access_mode": "search", "source_hint": "youtube.com"}'
                ']}'
            ),
            prompt_tokens=20,
            completion_tokens=12,
        )


class QueryGeneratorVLLMTests(unittest.TestCase):
    def test_explicit_video_url_repairs_source_type_and_preserves_url(self):
        generator = MaskSalienceQueryGenerator(
            query_model_name="qwen3:4b",
            llm_client=FakeLLMClient(),
        )
        original_url = "https://www.youtube.com/watch?v=L1vXCYZAYYM"
        requests = generator._parse_query_json(
            '{"queries": [{"query": "video https://www.youtube.com/watch?v=L1:1vXCYZAYYM", '
            '"source_kind": "web", "access_mode": "search", '
            '"source_hint": "YouTube page"}]}',
            question=f"In the video {original_url}, what is shown?",
        )

        self.assertEqual(len(requests), 1)
        self.assertIn(original_url, requests[0].query)
        self.assertNotIn("L1:1vXCYZAYYM", requests[0].query)
        self.assertEqual(requests[0].source_requirement.source_kind, "video")
        self.assertEqual(requests[0].source_requirement.access_mode, "direct_fetch")
        self.assertEqual(requests[0].source_requirement.source_hint, original_url)

    def test_query_generator_uses_injected_openai_compatible_client(self):
        client = FakeLLMClient()
        generator = MaskSalienceQueryGenerator(
            query_model_name="qwen3:4b",
            llm_client=client,
        )
        spans = [
            SalientSpan(
                text="Moon minimum perigee",
                start=0,
                end=20,
                score=1.0,
                tokens=["Moon", "minimum", "perigee"],
                token_indices=[0],
            )
        ]

        requests = generator.generate_queries_with_model(
            "How many hours would the journey take?",
            spans,
            num_candidates=2,
        )

        self.assertEqual(
            [request.query for request in requests],
            ["Moon minimum perigee", "Eliud Kipchoge marathon pace"],
        )
        self.assertEqual(requests[0].source_requirement.source_kind, "academic")
        self.assertEqual(requests[1].source_requirement.source_kind, "video")
        self.assertEqual(requests[1].source_requirement.source_hint, "youtube.com")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["model"], "qwen3:4b")
        self.assertEqual(client.calls[0]["max_tokens"], 768)
        self.assertFalse(client.calls[0]["enable_thinking"])

    def test_query_generator_uses_ollama_native_chat_without_thinking(self):
        client = FakeLLMClient(provider="ollama")
        generator = MaskSalienceQueryGenerator(
            query_model_name="qwen3:4b",
            llm_client=client,
        )
        spans = [
            SalientSpan(
                text="Moon minimum perigee",
                start=0,
                end=20,
                score=1.0,
                tokens=["Moon", "minimum", "perigee"],
                token_indices=[0],
            )
        ]

        requests = generator.generate_queries_with_model(
            "How many hours would the journey take?",
            spans,
            num_candidates=2,
        )

        self.assertEqual(
            [request.query for request in requests],
            ["Moon minimum perigee", "Eliud Kipchoge marathon pace"],
        )
        self.assertEqual(requests[0].source_requirement.access_mode, "search")
        self.assertEqual(requests[1].source_requirement.source_kind, "video")
        self.assertEqual(client.calls, [])
        self.assertEqual(len(client.native_calls), 1)
        self.assertFalse(client.native_calls[0]["think"])
        self.assertEqual(client.native_calls[0]["keep_alive"], 0)
        self.assertEqual(
            client.native_calls[0]["json_format"],
            generator.QUERY_JSON_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
