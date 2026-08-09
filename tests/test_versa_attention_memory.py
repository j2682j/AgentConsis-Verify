"""Pin the SDPA override that keeps VersaPRM inside the GPU.

VersaPRM's base model is grouped-query: 24 query heads against 8 KV heads.
Transformers asks `scaled_dot_product_attention` for GQA broadcasting whenever
torch is new enough and no attention mask is present, which is exactly how the
scorer calls it. No fused kernel on this build accepts that combination, so
SDPA falls back to the math backend and materialises `[1, 24, seq, seq]` in
float32 -- quadratic in the reasoning length.

Measured on the RTX 4080 at 8k tokens: 13.55 GB and 0.414 s that way, against
0.05 GB and 0.012 s once the KV heads are expanded instead. End to end through
the scorer, a 4k chain went from 10.05 GB to 7.45 GB and an 8k chain from
19.80 GB and 66 s to 8.86 GB and 0.99 s; 12k used to raise
`CUDA out of memory` and now 24k fits in 14.48 GB.

That quadratic growth is what ended level1_final_11 -- `CUDA error: unknown
error` on task 4 of 53, then three hours of empty results -- and what pushed
three tasks in level1_final_12 past the 16 GB card.

The override picks the other branch transformers already implements, so the
arithmetic is the same; only the kernel differs. Float16 kernels accumulate in
different orders, so scores move slightly: on a real 283-step chain the
per-step probabilities differ by at most 9.8e-4, and the averaged
`verifier_score` the winner selector reads by at most 3.7e-4 against a 0.83 to
0.97 range. Runs stay deterministic, but they are not bit-comparable with
level1_final_12 and earlier.
"""

from __future__ import annotations

import unittest
from unittest import mock

from score.versa_prm_scorer import _prefer_repeated_kv_attention


class _FakeModule:
    def __init__(self):
        self.use_gqa_in_sdpa = lambda attention_mask, key: True


class PreferRepeatedKvTest(unittest.TestCase):
    def _patched(self, module):
        integrations = mock.Mock(sdpa_attention=module)
        return mock.patch.dict(
            "sys.modules",
            {"transformers.integrations": integrations},
        )

    def test_gqa_broadcasting_is_turned_off(self) -> None:
        """The whole fix: transformers must expand KV instead of asking for GQA."""

        module = _FakeModule()
        with self._patched(module):
            applied = _prefer_repeated_kv_attention()

        self.assertTrue(applied)
        self.assertFalse(module.use_gqa_in_sdpa(None, None))

    def test_applying_it_twice_is_harmless(self) -> None:
        """`load()` runs per scorer instance and several share a process."""

        module = _FakeModule()
        with self._patched(module):
            self.assertTrue(_prefer_repeated_kv_attention())
            self.assertTrue(_prefer_repeated_kv_attention())

        self.assertFalse(module.use_gqa_in_sdpa(None, None))

    def test_a_transformers_without_the_hook_is_left_alone(self) -> None:
        """A layout change must not stop the scorer from loading."""

        module = mock.Mock(spec=[])
        with self._patched(module):
            self.assertFalse(_prefer_repeated_kv_attention())

    def test_an_unimportable_transformers_is_reported_not_raised(self) -> None:
        with mock.patch.dict("sys.modules", {"transformers.integrations": None}):
            self.assertFalse(_prefer_repeated_kv_attention())


if __name__ == "__main__":
    unittest.main()
