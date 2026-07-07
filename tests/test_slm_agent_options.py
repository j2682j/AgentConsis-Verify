from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from core.llm_client import LLMChatResult
from core.slm_agent import SLM_Agent


class FakeLLMClient:
    def __init__(self, result=None):
        self.calls = []
        self.stream_calls = []
        self.result = result or LLMChatResult(
            content="FINAL_ANSWER=OK",
            prompt_tokens=10,
            completion_tokens=3,
            raw_response=object(),
        )

    def chat(self, **kwargs):
        self.calls.append(kwargs)
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


if __name__ == "__main__":
    unittest.main()
