"""Separate a broken Evidence Prepare from one that simply found nothing.

Both used to surface the same way -- an empty `search_result` -- and the
confusion cost a full benchmark run. A `not vectors` check on a numpy array
raised inside corpus enrichment, escaped, and emptied the prepared evidence on
25 of 53 tasks in level1_final_18; the run reported 20/53, above the 18/53
baseline, and nothing in the record distinguished it from tasks where retrieval
had legitimately converted no evidence.

`pipeline_status` answers whether anything raised. `evidence_status` answers
what survived. They are independent: a task can lose one source and still reach
strict evidence, and a task with no failures at all can still end up empty.
"""

from __future__ import annotations

import unittest

from core.evidence_runner import EvidenceRunner

STATUS = EvidenceRunner._evidence_pipeline_status


def _output(*, failed_sources: int = 0, ok_sources: int = 0, enrichment_error: str = "") -> dict:
    searches = [{"actual_acquirer": "acquisition_failed"} for _ in range(failed_sources)]
    searches += [{"actual_acquirer": "search"} for _ in range(ok_sources)]
    return {
        "web_searches": searches,
        "diagnostics": {"collection_link_enrichment": {"error_traceback": enrichment_error}},
    }


def _evidence(strict: int = 0, relaxed: int = 0) -> list[dict]:
    return [{"relaxed": False}] * strict + [{"relaxed": True}] * relaxed


def _references(count: int) -> list[dict]:
    return [{"reference_id": f"R{i}"} for i in range(1, count + 1)]


class EvidencePipelineStatusTest(unittest.TestCase):
    def test_one_source_fails_and_the_rest_survive(self) -> None:
        pipeline, evidence = STATUS(
            output_dict=_output(failed_sources=1, ok_sources=1),
            evidence_items=_evidence(strict=2),
            references=_references(3),
            strict_evidence_count=2,
            source_count=4,
        )

        self.assertEqual(pipeline, "partial_failure")
        self.assertEqual(evidence, "strict")

    def test_enrichment_fails_but_the_corpus_stands(self) -> None:
        pipeline, evidence = STATUS(
            output_dict=_output(ok_sources=2, enrichment_error="ValueError: ..."),
            evidence_items=[],
            references=_references(4),
            strict_evidence_count=0,
            source_count=6,
        )

        self.assertEqual(pipeline, "partial_failure")
        self.assertEqual(evidence, "unverified_only")

    def test_every_acquisition_fails(self) -> None:
        pipeline, evidence = STATUS(
            output_dict=_output(failed_sources=3),
            evidence_items=[],
            references=[],
            strict_evidence_count=0,
            source_count=0,
        )

        self.assertEqual(pipeline, "failed")
        self.assertEqual(evidence, "empty")

    def test_nothing_raised_and_nothing_converted(self) -> None:
        """The case that must not read as a defect: retrieval simply found none."""

        pipeline, evidence = STATUS(
            output_dict=_output(ok_sources=5),
            evidence_items=[],
            references=[],
            strict_evidence_count=0,
            source_count=9,
        )

        self.assertEqual(pipeline, "complete")
        self.assertEqual(evidence, "empty")

    def test_relaxed_evidence_alone_is_not_strict(self) -> None:
        pipeline, evidence = STATUS(
            output_dict=_output(ok_sources=2),
            evidence_items=_evidence(relaxed=3),
            references=_references(1),
            strict_evidence_count=0,
            source_count=3,
        )

        self.assertEqual(pipeline, "complete")
        self.assertEqual(evidence, "unverified_only")

    def test_every_reachable_combination_is_reachable(self) -> None:
        """The matrix used to read an accuracy drop, so hold every cell open.

        `failed` implies `empty`: the status is only `failed` when nothing
        survived, which is what makes it different from `partial_failure`.
        """

        matrix = {
            ("complete", "strict"): (0, 2, 2, 4),
            ("complete", "unverified_only"): (0, 0, 2, 4),
            ("complete", "empty"): (0, 0, 0, 4),
            ("partial_failure", "strict"): (1, 2, 2, 4),
            ("partial_failure", "unverified_only"): (1, 0, 2, 4),
            ("partial_failure", "empty"): (1, 0, 0, 4),
            ("failed", "empty"): (1, 0, 0, 0),
        }
        for expected, (failures, strict, refs, sources) in matrix.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    STATUS(
                        output_dict=_output(
                            failed_sources=failures,
                            ok_sources=0 if not sources else 2,
                        ),
                        evidence_items=_evidence(strict=strict),
                        references=_references(refs),
                        strict_evidence_count=strict,
                        source_count=sources,
                    ),
                    expected,
                )

    def test_the_two_axes_do_not_constrain_each_other(self) -> None:
        for failures, strict, expected in (
            (0, 2, ("complete", "strict")),
            (1, 2, ("partial_failure", "strict")),
            (0, 0, ("complete", "unverified_only")),
            (1, 0, ("partial_failure", "unverified_only")),
        ):
            with self.subTest(failures=failures, strict=strict):
                self.assertEqual(
                    STATUS(
                        output_dict=_output(failed_sources=failures, ok_sources=2),
                        evidence_items=_evidence(strict=strict),
                        references=_references(2),
                        strict_evidence_count=strict,
                        source_count=4,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
