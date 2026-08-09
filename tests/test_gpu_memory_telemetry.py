"""Pin the VRAM telemetry that a GPU fault has to be readable from.

level1_final_11 lost 50 of 53 tasks to `CUDA error: unknown error` and left no
record of how full the card was. The surviving timing showed the collapse was
gradual -- 473 ms/step on the task before the fault against a 24 ms median for
comparable tasks -- so the interesting number is the trend across tasks, not
the value at the moment it died. These tests hold the two properties that
makes possible: the snapshot separates whole-card use from this process's use,
and a run without a GPU degrades to empty rather than raising.
"""

from __future__ import annotations

import unittest
from unittest import mock

from benchmark.gaia import gpu_memory

_MB = 1024 * 1024


class _FakeCuda:
    def __init__(self, *, available=True, free=6 * _MB, total=16 * _MB, raises=None):
        self._available = available
        self._free = free
        self._total = total
        self._raises = raises
        self.peak_resets = 0

    def is_available(self):
        return self._available

    def mem_get_info(self):
        if self._raises:
            raise self._raises
        return self._free, self._total

    def memory_allocated(self):
        return 2 * _MB

    def memory_reserved(self):
        return 3 * _MB

    def max_memory_reserved(self):
        return 5 * _MB

    def reset_peak_memory_stats(self):
        self.peak_resets += 1


def _patched(cuda):
    return mock.patch.object(gpu_memory, "torch", mock.Mock(cuda=cuda))


class SnapshotTest(unittest.TestCase):
    def test_reports_whole_card_and_this_process_separately(self) -> None:
        """The point of two figures: it says whether torch is the one filling it."""

        with _patched(_FakeCuda(free=6 * _MB, total=16 * _MB)):
            reading = gpu_memory.snapshot()

        self.assertEqual(reading["device_used_mb"], 10)
        self.assertEqual(reading["device_free_mb"], 6)
        self.assertEqual(reading["device_total_mb"], 16)
        self.assertEqual(reading["torch_allocated_mb"], 2)
        self.assertEqual(reading["torch_peak_reserved_mb"], 5)

    def test_no_gpu_gives_an_empty_reading_not_an_error(self) -> None:
        with _patched(_FakeCuda(available=False)):
            self.assertEqual(gpu_memory.snapshot(), {})
            self.assertFalse(gpu_memory.available())

    def test_missing_torch_gives_an_empty_reading(self) -> None:
        with mock.patch.object(gpu_memory, "torch", None):
            self.assertEqual(gpu_memory.snapshot(), {})
            self.assertFalse(gpu_memory.available())

    def test_a_failing_driver_is_reported_not_raised(self) -> None:
        """A run must not die because its telemetry could not be read."""

        with _patched(_FakeCuda(raises=RuntimeError("CUDA error: unknown error"))):
            reading = gpu_memory.snapshot()

        self.assertIn("error", reading)
        self.assertIn("unknown error", reading["error"])


class PerTaskTest(unittest.TestCase):
    def test_begin_task_resets_the_process_peak(self) -> None:
        """Otherwise every task reports the peak of the whole run so far."""

        cuda = _FakeCuda()
        with _patched(cuda):
            before = gpu_memory.begin_task()

        self.assertEqual(cuda.peak_resets, 1)
        self.assertEqual(before["device_used_mb"], 10)

    def test_begin_task_without_a_gpu_does_nothing(self) -> None:
        cuda = _FakeCuda(available=False)
        with _patched(cuda):
            self.assertEqual(gpu_memory.begin_task(), {})

        self.assertEqual(cuda.peak_resets, 0)


class SummaryLineTest(unittest.TestCase):
    def test_line_shows_the_change_across_the_task(self) -> None:
        before = {"device_used_mb": 4000, "device_total_mb": 16000}
        after = {"device_used_mb": 9000, "device_total_mb": 16000, "torch_peak_reserved_mb": 6000}

        line = gpu_memory.summarize(before, after)

        self.assertIn("9000/16000MB", line)
        self.assertIn("+5000MB", line)
        self.assertIn("torch_peak=6000MB", line)

    def test_nothing_to_report_gives_an_empty_line(self) -> None:
        self.assertEqual(gpu_memory.summarize({}, {}), "")
        self.assertEqual(gpu_memory.summarize({"device_used_mb": 1}, {"error": "boom"}), "")


if __name__ == "__main__":
    unittest.main()
