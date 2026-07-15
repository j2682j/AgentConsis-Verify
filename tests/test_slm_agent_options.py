from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from core.llm_client import LLMChatResult
from core.slm_agent import SLM_Agent


class FakeLLMClient:
    def __init__(self, result=None, provider="fake"):
        self.calls = []
        self.native_calls = []
        self.stream_calls = []
        self.provider = provider
        self.result = result or LLMChatResult(
            content="FINAL_ANSWER=OK",
            prompt_tokens=10,
            completion_tokens=3,
            raw_response=object(),
        )

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def ollama_native_chat(self, **kwargs):
        self.native_calls.append(kwargs)
        return self.result

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield "one"
        yield " two"


class SLMAgentOptionTests(unittest.TestCase):
    def test_agent_resolves_model_and_delegates_options(self):
        client = FakeLLMClient()
        with patch.dict(
            "os.environ",
            {"Qwen_MODEL_ID": "served-qwen", "LLM_ENABLE_THINKING": "false"},
            clear=False,
        ):
            agent = SLM_Agent(
                model_name="qwen3:4b",
                temperature=0.2,
                max_tokens=128,
                llm_client=client,
            )
            content = agent.invoke([{"role": "user", "content": "Hello"}])

        self.assertEqual(content, "FINAL_ANSWER=OK")
        self.assertEqual(client.calls[0]["model"], "served-qwen")
        self.assertEqual(client.calls[0]["temperature"], 0.2)
        self.assertEqual(client.calls[0]["max_tokens"], 128)
        self.assertFalse(client.calls[0]["enable_thinking"])

    def test_explicit_thinking_setting_is_preserved(self):
        client = FakeLLMClient()
        agent = SLM_Agent(
            model_name="qwen3:4b",
            enable_thinking=True,
            llm_client=client,
        )

        agent.invoke([{"role": "user", "content": "Hello"}])

        self.assertTrue(client.calls[0]["enable_thinking"])

    def test_native_tool_call_becomes_stage1_tool_request(self):
        client = FakeLLMClient(
            LLMChatResult(
                content="",
                reasoning="step 1. Search for the answer.",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search",
                        "arguments": {"query": "Moon perigee"},
                    }
                ],
            )
        )
        agent = SLM_Agent(model_name="qwen3:4b", llm_client=client)

        content = agent.invoke([{"role": "user", "content": "Find it"}])
        parsed = json.loads(content)

        self.assertEqual(parsed["type"], "tool_request")
        self.assertEqual(parsed["tool_name"], "search")
        self.assertEqual(parsed["tool_args"], {"query": "Moon perigee"})

    def test_usage_fallback_and_stream_delegate(self):
        client = FakeLLMClient(
            LLMChatResult(content="answer", prompt_tokens=0, completion_tokens=0)
        )
        agent = SLM_Agent(model_name="qwen3:4b", llm_client=client)
        messages = [{"role": "user", "content": "Question"}]

        content, prompt_tokens, completion_tokens = agent.invoke_with_usage(
            messages
        )
        chunks = list(agent.stream_invoke(messages))

        self.assertEqual(content, "answer")
        self.assertGreater(prompt_tokens, 0)
        self.assertGreater(completion_tokens, 0)
        self.assertEqual(chunks, ["one", " two"])

    def test_ollama_provider_uses_native_chat_and_keep_alive_unload(self):
        client = FakeLLMClient(provider="ollama")
        agent = SLM_Agent(
            model_name="qwen3:4b",
            temperature=0.3,
            max_tokens=64,
            llm_client=client,
        )

        content, prompt_tokens, completion_tokens = agent.invoke_with_usage(
            [{"role": "user", "content": "Hello"}],
            unload_after_call=True,
        )

        self.assertEqual(content, "FINAL_ANSWER=OK")
        self.assertEqual(client.calls, [])
        self.assertEqual(len(client.native_calls), 1)
        self.assertEqual(client.native_calls[0]["model"], "qwen3:4b")
        self.assertEqual(client.native_calls[0]["temperature"], 0.3)
        self.assertEqual(client.native_calls[0]["max_tokens"], 64)
        self.assertFalse(client.native_calls[0]["think"])
        self.assertEqual(client.native_calls[0]["keep_alive"], 0)
        self.assertEqual(prompt_tokens, 10)
        self.assertEqual(completion_tokens, 3)


if __name__ == "__main__":
    unittest.main()
