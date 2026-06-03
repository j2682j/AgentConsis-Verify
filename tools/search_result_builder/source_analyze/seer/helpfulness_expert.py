"""
Helpfulness Expert for SEER.

Helpfulness scoring:
    - 可使用 google/flan-t5-small 作為輕量 helpfulness expert。
    - f(a | q)：評估答案 a 是否能回應問題 q。
    - f(a | q, e)：評估 evidence e 是否有助於支持答案 a。
    - 透過 log / sigmoid 將模型分數壓到 0 到 1 區間，得到 helpfulness score。
"""

from __future__ import annotations

import math
import re
from typing import Any

from utils.network_utils import normalize_text


MODEL_NAME = "google/flan-t5-small"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class HelpfulnessExpert:
    """
    評估 evidence helpfulness。
        - 輸入模型：
            huggingface google/flan-t5-small。

        - 評估 evidence 是否有助於回答問題。

        - 將模型回覆轉成 sigmoid / 0-1 分數。

    """

    def __init__(self, *, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self):
        """
        Lazy-load helpfulness expert model。

        Args:
            - None.

        Returns:
            - None.
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (self.model_name, device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self.tokenizer, self.model, self.device = cached
            return

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        _MODEL_CACHE[cache_key] = (self.tokenizer, self.model, self.device)

    def score(self, *, question: str, answer: str = "", evidence: str = "") -> float | None:
        """
        使用模型估計 evidence 是否有助於回答問題。

        Args:
            - question: 原始問題。
            - answer: 候選答案，可為空字串。
            - evidence: 要評估的 evidence 文字。

        Returns:
            - float | None: helpfulness 分數，模型不可用時回傳 None。
        """
        try:
            self.load()
        except Exception:
            return None
        if self.tokenizer is None or self.model is None:
            return None

        import torch

        prompt = (
            "Rate whether the evidence helps answer the question. "
            "Return only yes or no.\n"
            f"Question: {question}\n"
            f"Answer: {answer or 'unknown'}\n"
            f"Evidence: {evidence}\n"
            "Helpful:"
        )
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                output = self.model.generate(**encoded, max_new_tokens=2, do_sample=False)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        except Exception:
            return None

        if decoded.startswith("yes"):
            return self._sigmoid(2.0)
        if decoded.startswith("no"):
            return self._sigmoid(-2.0)
        return 0.5

    def _sigmoid(self, value: float) -> float:
        """
        將 logit 轉成 0 到 1 的分數。

        Args:
            - value: logit 數值。

        Returns:
            - float: sigmoid 後的分數。
        """
        return 1.0 / (1.0 + math.exp(-value))



