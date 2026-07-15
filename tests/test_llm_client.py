from __future__ import annotations

import unittest
import json
from types import SimpleNamespace

from core.llm_client import LLMClient


class FakeCompletions:
    def __init__(self, response=None, stream_response=None):
        self.kwargs = {}
        self.response = response or SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"queries": ["query one"]}',
                        reasoning="",
                        tool_calls=[],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
        )
        self.stream_response = stream_response or []

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return self.stream_response
        return self.response


class FakeOpenAIClient:
    def __init__(self, response=None, stream_response=None):
        self.completions = FakeCompletions(response, stream_response)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeNativeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LLMClientTests(unittest.TestCase):
    def test_vllm_chat_uses_openai_endpoint_options(self):
        openai_client = FakeOpenAIClient()
        client = LLMClient(provider="vllm", client=openai_client)

        result = client.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Generate queries"}],
            temperature=0.1,
            max_tokens=768,
            enable_thinking=False,
        )

        sent = openai_client.completions.kwargs
        self.assertEqual(sent["model"], "qwen3:4b")
        self.assertEqual(sent["max_tokens"], 768)
        self.assertEqual(sent["temperature"], 0.1)
        self.assertFalse(
            sent["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(result.content, '{"queries": ["query one"]}')

    def test_ollama_adds_no_think_and_uses_reasoning_fallback(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        reasoning='{"queries": ["reasoning query"]}',
                        tool_calls=[],
                    )
                )
            ],
            usage=None,
        )
        openai_client = FakeOpenAIClient(response=response)
        client = LLMClient(provider="ollama", client=openai_client)

        result = client.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Generate queries"}],
            enable_thinking=False,
        )

        sent = openai_client.completions.kwargs
        self.assertTrue(sent["messages"][0]["content"].endswith("/no_think"))
        self.assertNotIn("extra_body", sent)
        self.assertEqual(result.content, '{"queries": ["reasoning query"]}')
        self.assertEqual(result.reasoning, result.content)

    def test_tool_calls_are_normalized(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        reasoning="step 1. Search for the result.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="search",
                                    arguments='{"query": "Moon perigee"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=None,
        )
        client = LLMClient(
            provider="ollama",
            client=FakeOpenAIClient(response=response),
        )

        result = client.chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Find the answer"}],
        )

        self.assertEqual(
            result.tool_calls,
            [
                {
                    "id": "call-1",
                    "name": "search",
                    "arguments": {"query": "Moon perigee"},
                }
            ],
        )

    def test_stream_yields_content_and_reasoning_chunks(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="hello"))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="", reasoning=" world")
                    )
                ]
            ),
        ]
        client = LLMClient(
            provider="ollama",
            client=FakeOpenAIClient(stream_response=chunks),
        )

        output = list(
            client.stream(
                model="qwen3:4b",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        self.assertEqual(output, ["hello", " world"])

    def test_ollama_native_chat_disables_thinking_and_requests_json(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeNativeResponse(
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"queries": ["Moon minimum perigee"]}',
                        "thinking": "",
                    },
                    "prompt_eval_count": 21,
                    "eval_count": 14,
                }
            )

        client = LLMClient(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            client=FakeOpenAIClient(),
            native_urlopen=fake_urlopen,
        )
        result = client.ollama_native_chat(
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Generate queries"}],
            temperature=0.1,
            max_tokens=256,
            think=False,
            json_format=True,
            keep_alive=0,
        )

        request, timeout = calls[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertEqual(payload["options"]["temperature"], 0.1)
        self.assertEqual(payload["keep_alive"], 0)
        self.assertEqual(timeout, client.timeout)
        self.assertEqual(
            result.content,
            '{"queries": ["Moon minimum perigee"]}',
        )
        self.assertEqual(result.prompt_tokens, 21)
        self.assertEqual(result.completion_tokens, 14)


if __name__ == "__main__":
    unittest.main()
