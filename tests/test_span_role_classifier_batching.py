"""Pin that an over-sized candidate set is classified, not silently truncated.

`max_spans_per_call` is sized against `max_tokens`: the model emits one role per
span, so a larger slice would truncate the reply rather than classify more. The
cap therefore has to bound a single prompt, not the batch -- it previously
bounded the batch with `spans[: self.max_spans_per_call]`, which dropped every
span past the cap without recording a reason.

That truncation was latent while an upstream budget held rounds to ten spans.
Raising that budget makes it load-bearing, so these tests hold the property
directly: every candidate comes back classified, and it takes more than one
bounded call to do it.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from tools.search_result_builder.evidence.span_role_classifier import (
    CandidateSpan,
    SpanRoleClassifier,
)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt_tokens = 10
        self.completion_tokens = 5


class _RecordingClient:
    """Answer BRIDGE for whatever ids a call asks about, and count the calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def ollama_native_chat(self, **kwargs: Any) -> _Response:
        schema = kwargs.get("json_format") or {}
        ids = _requested_ids(schema)
        self.calls.append(ids)
        payload = [{"id": span_id, "role": "BRIDGE", "goal_id": ""} for span_id in ids]
        return _Response(json.dumps(payload))


def _requested_ids(schema: Any) -> list[str]:
    """Pull the candidate ids the classifier constrained this call to.

    The schema carries a second enum for the role field, so read the id
    property directly rather than collecting every enum in the tree.
    """

    properties = (schema.get("items") or {}).get("properties") or {}
    return list((properties.get("id") or {}).get("enum") or [])


def _spans(count: int) -> list[CandidateSpan]:
    return [
        CandidateSpan(
            id=str(index),
            text=f"Distinct evidence sentence number {index}.",
            local_context=f"Context around sentence number {index}.",
            source_title="Source",
        )
        for index in range(1, count + 1)
    ]


class SpanRoleClassifierBatchingTest(unittest.TestCase):
    def _classifier(self, client: _RecordingClient) -> SpanRoleClassifier:
        return SpanRoleClassifier(llm_client=client, max_spans_per_call=5)

    def test_every_candidate_past_the_cap_is_still_classified(self) -> None:
        client = _RecordingClient()

        result = self._classifier(client).classify_batch(
            question="Who won?",
            spans=_spans(12),
        )

        self.assertEqual(len(result.results), 12)
        self.assertEqual(
            [item.id for item in result.results],
            [str(index) for index in range(1, 13)],
        )

    def test_oversized_batch_is_split_into_bounded_calls(self) -> None:
        client = _RecordingClient()

        result = self._classifier(client).classify_batch(
            question="Who won?",
            spans=_spans(12),
        )

        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(len(call) <= 5 for call in client.calls))
        self.assertEqual(result.diagnostics["chunk_count"], 3)
        self.assertTrue(result.diagnostics["success"])

    def test_batch_within_the_cap_still_takes_a_single_call(self) -> None:
        client = _RecordingClient()

        result = self._classifier(client).classify_batch(
            question="Who won?",
            spans=_spans(4),
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(result.results), 4)
        self.assertNotIn("chunk_count", result.diagnostics)

    def test_diagnostics_report_the_whole_candidate_set(self) -> None:
        client = _RecordingClient()

        result = self._classifier(client).classify_batch(
            question="Who won?",
            spans=_spans(12),
        )

        self.assertEqual(result.diagnostics["candidate_count"], 12)
        self.assertEqual(result.diagnostics["bridge_count"], 12)


if __name__ == "__main__":
    unittest.main()
