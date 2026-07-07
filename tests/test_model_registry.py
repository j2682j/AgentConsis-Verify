from __future__ import annotations

import unittest
from unittest.mock import patch

from core.model_registry import resolve_model_id


class ModelRegistryTests(unittest.TestCase):
    def test_known_alias_uses_environment_mapping(self):
        with patch.dict(
            "os.environ",
            {"Qwen_MODEL_ID": "served-qwen"},
            clear=False,
        ):
            self.assertEqual(resolve_model_id("qwen3:4b"), "served-qwen")

    def test_unknown_or_unconfigured_alias_is_preserved(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_model_id("custom-model"), "custom-model")
            self.assertEqual(resolve_model_id("qwen3:4b"), "qwen3:4b")


if __name__ == "__main__":
    unittest.main()
