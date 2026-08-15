"""The funnel replay must not be fed its own output.

A recorded search result carries the retrieval it was built from *and* what the
funnel made of it -- `evidence_items`, `unverified_references`, `summary`. Handing
those back to the rebuilder would reproduce them exactly while testing nothing,
which is the failure that made a requirement-gate repair measure as a no-op
across five runs: the replay restored the gates' conclusions and then reported
that changing the gates changed nothing.

`retrieval_input` is the boundary. These tests hold it, and hold the rebuilt
funnel to reproducing the recorded render byte for byte -- a funnel that cannot
reproduce the baseline cannot be trusted to say where an answer died.
"""

from __future__ import annotations

import glob
import json
import os
import unittest

from scripts.replay.evidence_funnel_rebuilder import (
    OUTPUT_FIELDS,
    digest,
    rebuild,
    retrieval_input,
)

RUN = "level_1_final_20"
TASKS = ("004", "013")


def _load(task_number: str) -> dict:
    matches = glob.glob(f"c:/SCP/outputs/{RUN}/tasks/{task_number}_*.json")
    if not matches:
        raise unittest.SkipTest(f"{RUN}/{task_number} not recorded")
    return json.loads(open(matches[0], encoding="utf-8").read())


def _recorded_search(task: dict) -> tuple[dict, str]:
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    for item in meta.get("tool_usage") or []:
        if isinstance(item, dict) and item.get("tool_name") == "search":
            raw = item.get("raw_result")
            if isinstance(raw, dict):
                return raw, str(item.get("output_text") or "")
    raise unittest.SkipTest("no recorded search result")


class RetrievalInputBoundaryTest(unittest.TestCase):
    def test_every_output_field_is_stripped(self) -> None:
        task = _load("004")
        raw, _summary = _recorded_search(task)

        stripped = retrieval_input(raw)

        for field in OUTPUT_FIELDS:
            with self.subTest(field=field):
                self.assertNotIn(field, stripped)

    def test_the_recording_really_did_carry_them(self) -> None:
        """Otherwise the test above passes for the wrong reason."""

        task = _load("004")
        raw, _summary = _recorded_search(task)

        self.assertTrue(
            any(field in raw for field in OUTPUT_FIELDS),
            "the recording should contain funnel output to strip",
        )

    def test_the_retrieval_the_funnel_needs_survives(self) -> None:
        task = _load("004")
        raw, _summary = _recorded_search(task)

        stripped = retrieval_input(raw)

        self.assertTrue((stripped.get("retrieval") or {}).get("rounds"))
        self.assertTrue((stripped.get("diagnostics") or {}).get("query_plan"))


class BaselineFidelityTest(unittest.TestCase):
    def test_the_rebuilt_render_matches_the_recorded_one(self) -> None:
        """Byte for byte, on both tasks P2 is aimed at."""

        for task_number in TASKS:
            with self.subTest(task=task_number):
                task = _load(task_number)
                _raw, summary = _recorded_search(task)
                replay = rebuild(task, task_id=task_number)

                self.assertEqual(
                    digest(replay.rendered_search_context),
                    digest(summary),
                    "rebuilt search context differs from the recorded one",
                )

    def test_reference_counts_match_the_recording(self) -> None:
        for task_number in TASKS:
            with self.subTest(task=task_number):
                task = _load(task_number)
                raw, _summary = _recorded_search(task)
                replay = rebuild(task, task_id=task_number)

                self.assertEqual(
                    len(replay.relaxed_reference_ids),
                    len(raw.get("unverified_references") or []),
                )

    def test_every_stage_reports_a_fidelity(self) -> None:
        replay = rebuild(_load("004"), task_id="004")

        names = [stage.name for stage in replay.stages]
        self.assertEqual(
            names, ["documents", "contract", "conversion", "reference", "render", "context"]
        )
        for stage in replay.stages:
            with self.subTest(stage=stage.name):
                self.assertIn(
                    stage.fidelity,
                    {"exact", "content_equivalent", "approximate", "unsupported"},
                )


if __name__ == "__main__":
    unittest.main()
