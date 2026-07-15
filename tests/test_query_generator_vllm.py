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
            content='{"queries": ["Moon minimum perigee", "Eliud Kipchoge marathon pace"]}',
            prompt_tokens=20,
            completion_tokens=12,
        )

    def ollama_native_chat(self, **kwargs):
        self.native_calls.append(kwargs)
        return LLMChatResult(
            content='{"queries": ["Moon minimum perigee", "Eliud Kipchoge marathon pace"]}',
            prompt_tokens=20,
            completion_tokens=12,
        )


class QueryGeneratorVLLMTests(unittest.TestCase):
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

        queries = generator.generate_queries_with_model(
            "How many hours would the journey take?",
            spans,
            num_candidates=2,
        )

        self.assertEqual(
            queries,
            ["Moon minimum perigee", "Eliud Kipchoge marathon pace"],
        )
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

        queries = generator.generate_queries_with_model(
            "How many hours would the journey take?",
            spans,
            num_candidates=2,
        )

        self.assertEqual(
            queries,
            ["Moon minimum perigee", "Eliud Kipchoge marathon pace"],
        )
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
