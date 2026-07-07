from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tools.tool_cache import ToolCache
from tools.tool_manager import ToolManager
from tools.video_transcript_tool import VideoTranscriptTool


class FakeAttachmentReader:
    name = "attachment_reader"

    def run(self, parameters):
        return {
            "used": True,
            "context": f"fake attachment context: {parameters.get('file_path')}",
            "metadata": {"reader": "fake_reader"},
            "tool_usage": [],
        }


class CountingToolManager:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.lock = Lock()

    def execute_tool(self, tool_name, tool_args, *, agent_id, stage):
        del tool_name, tool_args, agent_id, stage
        with self.lock:
            self.calls += 1
        time.sleep(0.04)
        return dict(self.result)


class RaisingToolManager:
    def __init__(self):
        self.calls = 0

    def execute_tool(self, tool_name, tool_args, *, agent_id, stage):
        del tool_name, tool_args, agent_id, stage
        self.calls += 1
        raise RuntimeError("backend unavailable")


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.manager = ToolManager()

    def test_attachment_nested_failure_is_not_reported_as_success(self):
        result = self.manager.normalize_result(
            "attachment_reader",
            {
                "used": True,
                "context": "Extracted content:\nNone",
                "metadata": {"reader": "error_reader"},
                "tool_usage": [{"ok": False, "error": "vision context exceeded"}],
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "retryable_failure")
        self.assertEqual(result["error_code"], "attachment_read_failed")
        self.assertFalse(result["evidence_valid"])

    def test_unsupported_deterministic_result_is_explicit(self):
        result = self.manager.normalize_result(
            "deterministic_solver",
            {
                "used_deterministic_solver": False,
                "error": "no deterministic handler matched",
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["error_code"], "deterministic_handler_not_found")

    def test_deterministic_solver_gap_is_preserved(self):
        result = self.manager.normalize_result(
            "deterministic_solver",
            {
                "used_deterministic_solver": False,
                "error": "missing required deterministic handler inputs",
                "missing_inputs": ["table_rows"],
                "next_action_hint": "Use attachment_reader or provide CSV rows.",
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing_inputs"], ["table_rows"])
        self.assertEqual(
            result["next_action_hint"],
            "Use attachment_reader or provide CSV rows.",
        )
        self.assertEqual(
            result["retry_hint"],
            "Use attachment_reader or provide CSV rows.",
        )

    def test_empty_search_result_is_partial_without_evidence(self):
        result = self.manager.normalize_result(
            "search",
            {"results": [], "backend": "searxng", "notices": []},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(result["retryable"])

    def test_video_transcript_rejects_local_media_path(self):
        result = VideoTranscriptTool().run({"url": "sample.mp3"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["error_code"], "local_media_requires_attachment_reader")

    def test_manager_reroutes_local_media_video_request_to_attachment_reader(self):
        self.manager.tools["attachment_reader"] = FakeAttachmentReader()

        result = self.manager.execute_tool(
            "video_transcript",
            {
                "input": "question focus",
                "attachment": {
                    "file_path": "C:\\SCP\\data\\gaia\\sample.mp3",
                    "file_name": "sample.mp3",
                    "extension": ".mp3",
                },
            },
            agent_id="a1",
            stage="stage1",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool_name"], "attachment_reader")
        self.assertIn("sample.mp3", result["output_text"])

    def test_single_flight_executes_identical_parallel_request_once(self):
        manager = CountingToolManager(self._success_result())
        cache = ToolCache()

        def execute(index):
            return cache.get_or_execute(
                tool_manager=manager,
                tool_name="search",
                tool_args={"input": "same query"},
                agent_id=f"a{index}",
                stage="stage1",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(execute, range(8)))

        self.assertEqual(manager.calls, 1)
        self.assertEqual(sum(not item["cache_hit"] for item in results), 1)
        self.assertEqual(sum(item["duplicate_request"] for item in results), 7)
        self.assertTrue(
            all(item["status"] in {"success", "already_available"} for item in results)
        )

    def test_identical_failed_request_is_blocked(self):
        manager = CountingToolManager(self._failed_result())
        cache = ToolCache()
        kwargs = {
            "tool_manager": manager,
            "tool_name": "deterministic_solver",
            "tool_args": {"input": "solve graph"},
            "agent_id": "a1",
            "stage": "stage1",
        }

        first = cache.get_or_execute(**kwargs)
        second = cache.get_or_execute(**kwargs)

        self.assertEqual(manager.calls, 1)
        self.assertEqual(first["status"], "unsupported")
        self.assertEqual(second["status"], "duplicate_blocked")
        self.assertEqual(second["error_code"], "duplicate_failed_request")

    def test_single_flight_releases_waiters_when_manager_raises(self):
        manager = RaisingToolManager()
        cache = ToolCache()
        kwargs = {
            "tool_manager": manager,
            "tool_name": "search",
            "tool_args": {"input": "same query"},
            "agent_id": "a1",
            "stage": "stage1",
        }

        first = cache.get_or_execute(**kwargs)
        second = cache.get_or_execute(**kwargs)

        self.assertEqual(manager.calls, 1)
        self.assertEqual(first["status"], "retryable_failure")
        self.assertEqual(first["error_code"], "tool_manager_exception")
        self.assertEqual(second["status"], "duplicate_blocked")

    def _success_result(self):
        return {
            "ok": True,
            "tool_name": "search",
            "status": "success",
            "output_text": "result",
            "raw_result": {"results": [{"title": "x"}]},
            "error": None,
            "error_code": "",
            "error_message": "",
            "retryable": False,
            "retry_hint": "",
            "evidence_valid": True,
            "cache_hit": False,
            "duplicate_request": False,
        }

    def _failed_result(self):
        return {
            "ok": False,
            "tool_name": "deterministic_solver",
            "status": "unsupported",
            "output_text": "",
            "raw_result": None,
            "error": "missing capability",
            "error_code": "deterministic_handler_not_found",
            "error_message": "missing capability",
            "retryable": False,
            "retry_hint": "",
            "evidence_valid": False,
            "cache_hit": False,
            "duplicate_request": False,
        }


if __name__ == "__main__":
    unittest.main()
