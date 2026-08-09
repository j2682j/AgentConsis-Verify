"""Pin the properties a comparison seed has to have to be useful.

level1_final_06, _07 and _08 each scored 19 of 53 while 8 tasks flipped between
consecutive runs, and _08 against _10 -- two configurations with identical
run-level accuracy -- still differed on 6 tasks. So the empirical noise floor is
about +/-3 tasks and a change worth one or two is unreadable without a pinned
seed. Pinning makes a rerun of unchanged code reproduce its result -- but only
if the seed is stable across processes and still differs per run, since three
identical runs would make their aggregation meaningless.

Pinned is the default, and free sampling has to be asked for, because the
failure mode being guarded against is an unpinned run that nobody notices until
its numbers get compared against something.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from core.sampling_seed import (
    DEFAULT_SEED,
    ENV_VAR,
    base_seed,
    describe,
    run_seed,
    sampling_overrides,
)

ROOT = Path(__file__).resolve().parents[1]


class SamplingSeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.get(ENV_VAR)
        os.environ[ENV_VAR] = "42"

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = self._previous

    def test_unset_is_pinned_not_free(self) -> None:
        """Forgetting to set it must not silently produce an incomparable run."""

        os.environ.pop(ENV_VAR, None)

        self.assertEqual(base_seed(), DEFAULT_SEED)
        self.assertIn("seed", sampling_overrides(agent_id="qwen", run_index=1))

    def test_free_sampling_has_to_be_asked_for(self) -> None:
        for value in ("off", "OFF", "none", "free", "random"):
            with self.subTest(value=value):
                os.environ[ENV_VAR] = value

                self.assertIsNone(base_seed())
                self.assertIsNone(run_seed(agent_id="qwen", run_index=1))
                self.assertEqual(sampling_overrides(agent_id="qwen", run_index=1), {})

    def test_an_unreadable_value_raises_rather_than_unpinning(self) -> None:
        os.environ[ENV_VAR] = "not-a-number"

        with self.assertRaises(ValueError):
            base_seed()

    def test_describe_names_both_states(self) -> None:
        os.environ[ENV_VAR] = "7"
        self.assertEqual(describe(), "7")

        os.environ[ENV_VAR] = "off"
        self.assertIn("free", describe())

    def test_same_run_is_repeatable(self) -> None:
        first = run_seed(agent_id="qwen", run_index=2)
        second = run_seed(agent_id="qwen", run_index=2)

        self.assertEqual(first, second)

    def test_runs_of_one_agent_stay_distinct(self) -> None:
        seeds = {run_seed(agent_id="qwen", run_index=index) for index in (1, 2, 3)}

        self.assertEqual(len(seeds), 3)

    def test_agents_do_not_share_a_seed(self) -> None:
        qwen = {run_seed(agent_id="qwen", run_index=i) for i in (1, 2, 3)}
        gemma = {run_seed(agent_id="gemma", run_index=i) for i in (1, 2, 3)}

        self.assertEqual(qwen & gemma, set())

    def test_turns_within_a_run_stay_distinct(self) -> None:
        self.assertNotEqual(
            run_seed(agent_id="qwen", run_index=1, turn=0),
            run_seed(agent_id="qwen", run_index=1, turn=1),
        )

    def test_a_different_base_seed_gives_a_different_schedule(self) -> None:
        first = run_seed(agent_id="qwen", run_index=1)
        os.environ[ENV_VAR] = "43"
        second = run_seed(agent_id="qwen", run_index=1)

        self.assertNotEqual(first, second)

    def test_seed_is_stable_across_processes(self) -> None:
        """The whole point: str hash() is salted per process, blake2s is not."""

        script = (
            "import sys; sys.path.insert(0, r'%s');"
            "from core.sampling_seed import run_seed;"
            "print(run_seed(agent_id='qwen', run_index=1))" % ROOT
        )
        env = {**os.environ, ENV_VAR: "42", "PYTHONHASHSEED": "0"}
        first = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        ).stdout.strip()
        env["PYTHONHASHSEED"] = "1"
        second = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        ).stdout.strip()

        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertEqual(int(first), run_seed(agent_id="qwen", run_index=1))

    def test_overrides_carry_the_seed_for_the_completion_call(self) -> None:
        overrides = sampling_overrides(agent_id="qwen", run_index=1)

        self.assertEqual(
            overrides, {"seed": run_seed(agent_id="qwen", run_index=1)}
        )


class SeedReachesTheAgentTest(unittest.TestCase):
    """Stage1 has to actually pass the seed down, not just be able to derive one.

    Pinning by default made every scripted test double start receiving a `seed`
    keyword, which is the visible sign of this hop working. Guard it directly so
    the wiring cannot be dropped while `sampling_overrides` keeps passing.
    """

    def setUp(self) -> None:
        self._previous = os.environ.get(ENV_VAR)

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = self._previous

    def _run(self) -> list[dict]:
        import json

        from core.config import AgentConfig
        from core.stage1_tool_use_runner import Stage1ToolUseRunner
        from tools.tool_manager import ToolManager

        captured: list[dict] = []

        class _Agent:
            def invoke_with_usage(self, messages, **overrides):
                captured.append(overrides)
                return (
                    json.dumps(
                        {
                            "type": "final_answer",
                            "reasoning_steps": ["step 1. Answer."],
                            "final_answer": "ok",
                            "confidence": 0.5,
                        }
                    ),
                    10,
                    5,
                )

        Stage1ToolUseRunner(tool_manager=ToolManager(), max_tool_turns=1).run(
            config=AgentConfig(agent_id="a1", model_name="fake"),
            agent=_Agent(),
            question="Find the answer.",
            evidence_packets=[],
            run_index=1,
        )
        return captured

    def test_a_pinned_seed_reaches_the_agent_call(self) -> None:
        os.environ[ENV_VAR] = "42"

        captured = self._run()

        self.assertTrue(captured)
        # The runner varies the seed per turn as well, and turns are 1-based.
        self.assertEqual(
            captured[0].get("seed"),
            run_seed(agent_id="a1", run_index=1, turn=1),
        )

    def test_free_sampling_sends_no_seed(self) -> None:
        os.environ[ENV_VAR] = "off"

        captured = self._run()

        self.assertTrue(captured)
        self.assertNotIn("seed", captured[0])


class SeedReachesTheModelTest(unittest.TestCase):
    """The seed is only worth anything if it survives to the provider call.

    `_completion_options` and `_ollama_native_chat` pop several keys out of the
    options dict on the way down, so an override can be silently dropped.
    """

    def _agent_with_capture(self):
        from core.slm_agent import SLM_Agent

        captured: dict = {}

        class _Client:
            provider = "ollama"

            def ollama_native_chat(self, **kwargs):
                captured.update(kwargs)

                class _Result:
                    content = "ok"
                    reasoning = ""
                    prompt_tokens = 1
                    completion_tokens = 1
                    tool_calls = None

                return _Result()

        agent = SLM_Agent.__new__(SLM_Agent)
        agent.llm_client = _Client()
        agent.model = "qwen3:4b"
        agent.temperature = 0.5
        agent.max_tokens = 512
        agent.enable_thinking = False
        agent.reasoning_effort = ""
        agent.kwargs = {}
        agent.use_ollama_native = True
        return agent, captured

    def test_seed_override_reaches_the_provider(self) -> None:
        agent, captured = self._agent_with_capture()

        options = agent._completion_options({"seed": 12345})
        agent._ollama_native_chat([{"role": "user", "content": "hi"}], dict(options))

        self.assertEqual(captured.get("seed"), 12345)

    def test_no_seed_is_sent_when_none_is_configured(self) -> None:
        agent, captured = self._agent_with_capture()

        options = agent._completion_options({})
        agent._ollama_native_chat([{"role": "user", "content": "hi"}], dict(options))

        self.assertNotIn("seed", captured)


if __name__ == "__main__":
    unittest.main()
