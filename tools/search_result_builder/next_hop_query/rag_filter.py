from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.network_utils import normalize_text

from ..config import EvidenceItem
from .filter_input_builder import FilterInputBuilder, QueryInfoTokenRecord

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_FILTER_CHECKPOINT = PROJECT_ROOT / "models" / "filter_mixed_v1" / "filter_mixed_v1"
FILTER_MAX_LENGTH = 128
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


def _resolve_filter_checkpoint(path: str | Path | None = None) -> Path:
    checkpoint = Path(path) if path else PROJECT_FILTER_CHECKPOINT
    if (checkpoint / "config.json").exists():
        return checkpoint

    nested = checkpoint / checkpoint.name
    if (nested / "config.json").exists():
        return nested

    return checkpoint


@dataclass
class RAGFilterResult:
    """
    儲存 EfficientRAG filter 產生 next-hop query 的結果。

    Args:
        - query: 產生出的 next-hop query。
        - kept_question_tokens: 從原始問題保留下來的 tokens。
        - kept_evidence_tokens: 從 useful evidence 保留下來的 tokens。
        - fallback_used: 是否使用 deterministic fallback。
        - metadata: 模型、輸入與錯誤等診斷資訊。

    Returns:
        - RAGFilterResult: next-hop query filter 結果。
    """

    query: str
    kept_question_tokens: list[str] = field(default_factory=list)
    kept_evidence_tokens: list[str] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class EfficientRAGFilterAdapter:
    """
    使用 EfficientRAG filter checkpoint 產生 next-hop query，失敗時退回 keyword fallback。

    Args:
        - max_question_tokens: fallback query 最多保留的 question tokens。
        - max_evidence_tokens: fallback query 最多保留的 evidence tokens。
        - filter_checkpoint: EfficientRAG filter checkpoint 路徑。
        - device: 模型執行裝置，未指定時自動選 cuda / cpu。
        - max_filter_info_items: 傳給 filter model 的 evidence item 數量上限。
        - max_filter_info_chars: 每個 evidence item 傳給 filter model 的字元上限。

    Returns:
        - EfficientRAGFilterAdapter: next-hop query filter。
    """

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "answer",
        "question",
    }

    def __init__(
        self,
        *,
        max_question_tokens: int = 6,
        max_evidence_tokens: int = 8,
        filter_checkpoint: str | None = None,
        device: str | None = None,
        max_filter_info_items: int = 4,
        max_filter_info_chars: int = 240,
        filter_input_builder: FilterInputBuilder | None = None,
    ) -> None:
        self.max_question_tokens = max_question_tokens
        self.max_evidence_tokens = max_evidence_tokens
        self.filter_checkpoint = str(
            _resolve_filter_checkpoint(filter_checkpoint)
        )
        self.device = device or os.getenv("SEARCH_FILTER_DEVICE", "cpu")
        self.max_filter_info_items = max(1, max_filter_info_items)
        self.max_filter_info_chars = max(80, max_filter_info_chars)
        self.filter_input_builder = filter_input_builder or FilterInputBuilder()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._model_device: str | None = None

    def build_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> RAGFilterResult:
        """
        根據原始問題與 useful evidence 產生 next-hop query。

        Args:
            - question: 原始問題。
            - evidence_items: evidence conversion 產生的 useful evidence items。

        Returns:
            - RAGFilterResult: next-hop query 結果。
        """
        model_result = self._build_model_query(question=question, evidence_items=evidence_items)
        if model_result is not None and model_result.query:
            return model_result
        return self._build_fallback_query(question=question, evidence_items=evidence_items)

    def _build_model_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> RAGFilterResult | None:
        info_list = self._evidence_info_list(evidence_items)
        if not info_list:
            return None
        record = self.filter_input_builder.build(
            query=question,
            info_items=info_list,
        )
        if not record.info_tokens:
            return None

        try:
            self._load_filter_model()
            tokenized = self._build_filter_input(record.query_info_tokens)
            selected_tokens, predicted_labels = self._predict_filtered_query(tokenized)
            query = self._compose_query_from_tokens(selected_tokens)
        except Exception as exc:
            fallback = self._build_fallback_query(question=question, evidence_items=evidence_items)
            fallback.fallback_used = True
            fallback.metadata = {
                **fallback.metadata,
                "filter_checkpoint": self.filter_checkpoint,
                "model_error": f"{type(exc).__name__}: {exc}",
                "query_info_tokens": record.query_info_tokens,
            }
            return fallback

        cleaned_query = self._clean_query(query)
        if not cleaned_query:
            fallback = self._build_fallback_query(question=question, evidence_items=evidence_items)
            fallback.fallback_used = True
            fallback.metadata = {
                **fallback.metadata,
                "method": "efficientrag_filter_model_empty_fallback",
                "filter_checkpoint": self.filter_checkpoint,
                "query_info_tokens": record.query_info_tokens,
                "predicted_query_info_labels": predicted_labels,
            }
            return fallback

        kept_tokens = self._ordered_keywords(cleaned_query)
        question_token_set = set(self._ordered_keywords(question))
        evidence_token_set: set[str] = set()
        for info in info_list:
            evidence_token_set.update(self._ordered_keywords(info))

        return RAGFilterResult(
            query=cleaned_query,
            kept_question_tokens=[token for token in kept_tokens if token in question_token_set],
            kept_evidence_tokens=[token for token in kept_tokens if token in evidence_token_set],
            fallback_used=False,
            metadata={
                "method": "efficientrag_filter_model",
                "filter_checkpoint": self.filter_checkpoint,
                "query_info_tokens": record.query_info_tokens,
                "predicted_query_info_labels": predicted_labels,
                "selected_query_info_tokens": selected_tokens,
                "filter_input": self._record_to_text(record),
                "device": self._model_device or "",
            },
        )

    def _build_fallback_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> RAGFilterResult:
        info_list = self._evidence_info_list(evidence_items)
        record = self.filter_input_builder.build(
            query=question,
            info_items=info_list,
        )
        question_tokens = self._ordered_keywords(question)[: self.max_question_tokens]
        evidence_tokens: list[str] = []
        for item in evidence_items:
            evidence_tokens.extend(item.matched_terms)
            evidence_tokens.extend(self._ordered_keywords(item.text)[:4])
            evidence_tokens = self._dedupe(evidence_tokens)
            if len(evidence_tokens) >= self.max_evidence_tokens:
                break
        evidence_tokens = evidence_tokens[: self.max_evidence_tokens]
        parts = self._dedupe(evidence_tokens + question_tokens)
        query = normalize_text(" ".join(parts))[:300]
        fallback_used = False
        if not query:
            query = normalize_text(question)
            fallback_used = True
        return RAGFilterResult(
            query=query,
            kept_question_tokens=question_tokens,
            kept_evidence_tokens=evidence_tokens,
            fallback_used=fallback_used,
            metadata={
                "method": "efficientrag_filter_fallback",
                "query_info_tokens": record.query_info_tokens,
                "filter_input": self._record_to_text(record),
            },
        )

    def _load_filter_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        checkpoint = str(Path(self.filter_checkpoint))
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"EfficientRAG filter checkpoint not found: {checkpoint}")

        try:
            import sentencepiece  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "EfficientRAG filter requires SentencePiece. "
                "Install it in the active Python environment with "
                "`python -m pip install sentencepiece`, then restart the process."
            ) from exc

        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (checkpoint, device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._tokenizer, self._model, self._model_device = cached
            return

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._model_device = device
        _MODEL_CACHE[cache_key] = (tokenizer, model, device)

    def _evidence_info_list(self, evidence_items: list[EvidenceItem]) -> list[str]:
        infos: list[str] = []
        for item in evidence_items[: self.max_filter_info_items]:
            text = normalize_text(" ".join(item.matched_terms or []) or item.text)
            if text:
                infos.append(text[: self.max_filter_info_chars])
        return infos

    def _record_to_text(self, record: QueryInfoTokenRecord) -> str:
        return self._compose_query_from_tokens(record.query_info_tokens)

    def _build_filter_input(self, query_info_tokens: list[str]) -> dict[str, Any]:
        assert self._tokenizer is not None
        words = [CLS_TOKEN] + list(query_info_tokens) + [SEP_TOKEN]
        tokens: list[str] = []
        word_ranges: list[tuple[int, int]] = []
        for word in words:
            start = len(tokens)
            pieces = self._tokenizer.tokenize(word)
            tokens.extend(pieces)
            word_ranges.append((start, len(tokens)))
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens[:FILTER_MAX_LENGTH])
        model_inputs = self._tokenizer.pad(
            {"input_ids": [input_ids]},
            max_length=FILTER_MAX_LENGTH,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "model_inputs": model_inputs,
            "word_tokens": words,
            "word_ranges": word_ranges,
            "query_info_tokens": list(query_info_tokens),
        }

    def _spacify(self, text: str) -> list[str]:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            return [word.text for word in nlp(text) if word.lemma_ != ","]
        except Exception:
            return re.findall(r"[A-Za-z0-9_.-]+|[^\w\s]", normalize_text(text))

    def _predict_filtered_query(self, tokenized: dict[str, Any]) -> tuple[list[str], list[bool]]:
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._model_device is not None

        model_inputs = tokenized["model_inputs"]
        tokenized_on_device = {
            key: value.to(self._model_device)
            for key, value in model_inputs.items()
        }
        with torch.no_grad():
            outputs = self._model(**tokenized_on_device)
        labels = outputs.logits.argmax(dim=-1).detach().cpu()[0]
        attention_mask = model_inputs["attention_mask"][0]
        word_tokens: list[str] = tokenized["word_tokens"]
        word_ranges: list[tuple[int, int]] = tokenized["word_ranges"]
        query_info_tokens: list[str] = tokenized["query_info_tokens"]

        selected_tokens: list[str] = []
        predicted_labels: list[bool] = []
        for word_index, (word, token_range) in enumerate(zip(word_tokens, word_ranges)):
            if word_index == 0 or word_index == len(word_tokens) - 1:
                continue
            start, end = token_range
            if start >= FILTER_MAX_LENGTH:
                predicted_labels.append(False)
                continue
            end = min(end, FILTER_MAX_LENGTH)
            active_positions = [
                position
                for position in range(start, end)
                if int(attention_mask[position].item()) == 1
            ]
            keep = bool(
                active_positions
                and any(int(labels[position].item()) == 1 for position in active_positions)
            )
            predicted_labels.append(keep)
            if keep and not self._is_control_token(word):
                selected_tokens.append(word)

        if len(predicted_labels) < len(query_info_tokens):
            predicted_labels.extend([False] * (len(query_info_tokens) - len(predicted_labels)))
        return selected_tokens, predicted_labels[: len(query_info_tokens)]

    def _compose_query_from_tokens(self, tokens: list[str]) -> str:
        text = " ".join(token for token in tokens if normalize_text(token))
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return normalize_text(text)

    def _is_control_token(self, token: str) -> bool:
        cleaned = normalize_text(token)
        if cleaned in {CLS_TOKEN, SEP_TOKEN, "Query", "Info", ":"}:
            return True
        return bool(re.fullmatch(r"[^\w\s]", cleaned))

    def _clean_query(self, query: str) -> str:
        cleaned = normalize_text(query)
        cleaned = re.sub(r"\b(?:Query|Info)\s*:\s*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
        return cleaned[:300]

    def _ordered_keywords(self, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for token in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalize_text(text).lower()):
            if token in self.STOPWORDS or len(token) <= 2 or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = normalize_text(value).lower()
            if key and key not in seen:
                result.append(value)
                seen.add(key)
        return result


__all__ = ["EfficientRAGFilterAdapter", "RAGFilterResult"]
