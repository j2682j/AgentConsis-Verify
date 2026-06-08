"""
Helpfulness Expert for SEER.

This module uses google/flan-t5-small only as a probability model. It does not
ask the model to judge, classify, or follow a scoring instruction.

Formula:
    - f(a | q) is the sequence log probability of the reference answer a when
      the model is conditioned only on the question q.
    - f(a | q, e) is the sequence log probability of the same answer a when
      the model is conditioned on the question q plus evidence e.
    - helpfulness = sigmoid(f(a | q, e) - f(a | q)).

If the reference answer is not available, score() returns None so the caller can
use a fallback or skip model-based helpfulness.
"""

from __future__ import annotations

import math
from typing import Any

from utils.network_utils import normalize_text


MODEL_NAME = "google/flan-t5-small"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class HelpfulnessExpert:
    """
    使用 flan-t5-small 計算標準答案的條件對數機率變化。

    Args:
        - model_name: HuggingFace seq2seq 模型名稱。
        - device: 執行裝置；未指定時自動使用 cuda 或 cpu。

    Returns:
        - HelpfulnessExpert: 純 log probability helpfulness 計算器。
    """

    def __init__(self, *, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype).to(device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        _MODEL_CACHE[cache_key] = (tokenizer, model, device)

    def score(self, *, question: str, answer: str = "", evidence: str = "") -> float | None:
        """
        根據標準答案 log probability delta 計算 evidence helpfulness。

        此函式不使用評分 prompt。它只做兩次條件機率計算：

        1. baseline: f(answer | question)
        2. with evidence: f(answer | question + evidence)

        最後回傳 sigmoid(with_evidence_logprob - baseline_logprob)。

        Args:
            - question: 原始問題。
            - answer: 標準答案；若未提供，無法進行純機率計算並回傳 None。
            - evidence: 待評估的證據文字。

        Returns:
            - float | None: 0 到 1 的 helpfulness 分數；輸入不足或模型不可用時回傳 None。
        """
        question_text = normalize_text(question)
        answer_text = normalize_text(answer)
        evidence_text = normalize_text(evidence)
        if not question_text or not answer_text or not evidence_text:
            return None

        try:
            self.load()
            assert self.tokenizer is not None
            assert self.model is not None
            assert self.device is not None

            baseline_logprob = self._sequence_logprob(
                condition=self._condition_text(question=question_text),
                target=answer_text,
            )
            evidence_logprob = self._sequence_logprob(
                condition=self._condition_text(question=question_text, evidence=evidence_text),
                target=answer_text,
            )
            return max(0.0, min(self._sigmoid(evidence_logprob - baseline_logprob), 1.0))
        except Exception:
            return None

    def logprob(self, *, question: str, answer: str, evidence: str = "") -> float | None:
        """
        直接回傳標準答案在指定條件下的 sequence log probability。

        Args:
            - question: 原始問題。
            - answer: 標準答案。
            - evidence: 可選 evidence；若提供，條件為 question + evidence。

        Returns:
            - float | None: answer 的 sequence log probability；模型不可用時回傳 None。
        """
        question_text = normalize_text(question)
        answer_text = normalize_text(answer)
        evidence_text = normalize_text(evidence)
        if not question_text or not answer_text:
            return None
        try:
            self.load()
            return self._sequence_logprob(
                condition=self._condition_text(question=question_text, evidence=evidence_text),
                target=answer_text,
            )
        except Exception:
            return None

    def _condition_text(self, *, question: str, evidence: str = "") -> str:
        if evidence:
            return normalize_text(f"{question}\n{evidence}")
        return normalize_text(question)

    def _sequence_logprob(self, *, condition: str, target: str) -> float:
        import torch

        assert self.tokenizer is not None
        assert self.model is not None
        assert self.device is not None

        encoded_condition = self.tokenizer(
            condition,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        encoded_target = self.tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=64,
        )
        labels = encoded_target["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_condition, labels=labels)

        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            mask = torch.ones_like(labels, dtype=token_log_probs.dtype)
        else:
            mask = (labels != pad_token_id).to(token_log_probs.dtype)
        return float((token_log_probs * mask).sum().detach().cpu().item())

    def _sigmoid(self, value: float) -> float:
        """
        將 log probability delta 轉成 0 到 1 的分數。

        Args:
            - value: f(a | q, e) - f(a | q)。

        Returns:
            - float: sigmoid 後的 helpfulness score。
        """
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)


__all__ = ["HelpfulnessExpert"]
