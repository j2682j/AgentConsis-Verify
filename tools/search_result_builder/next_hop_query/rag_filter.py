from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.network_utils import normalize_text

from ..config import EvidenceItem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_FILTER_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "filter"
    / "hotpotQA"
    / "filter_20260601_211420"
    / "checkpoint-36303"
)
FILTER_MAX_LENGTH = 128
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


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
    ) -> None:
        self.max_question_tokens = max_question_tokens
        self.max_evidence_tokens = max_evidence_tokens
        self.filter_checkpoint = filter_checkpoint or str(PROJECT_FILTER_CHECKPOINT)
        self.device = device
        self.max_filter_info_items = max(1, max_filter_info_items)
        self.max_filter_info_chars = max(80, max_filter_info_chars)
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
            - evidence_items: SourceAnalysis 產生的 useful evidence items。

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

        try:
            self._load_filter_model()
            sentence = self._build_query_info_sentence(info_list=info_list, query=question)
            tokenized = self._build_filter_input(sentence)
            query = self._predict_filtered_query(tokenized)
        except Exception as exc:
            fallback = self._build_fallback_query(question=question, evidence_items=evidence_items)
            fallback.fallback_used = True
            fallback.metadata = {
                **fallback.metadata,
                "filter_checkpoint": self.filter_checkpoint,
                "model_error": f"{type(exc).__name__}: {exc}",
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
                "filter_input": sentence,
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
                "filter_input": sentence,
                "device": self._model_device or "",
            },
        )

    def _build_fallback_query(
        self,
        *,
        question: str,
        evidence_items: list[EvidenceItem],
    ) -> RAGFilterResult:
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
            metadata={"method": "efficientrag_filter_fallback"},
        )

    def _load_filter_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        checkpoint = str(Path(self.filter_checkpoint))
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"EfficientRAG filter checkpoint not found: {checkpoint}")

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
            text = normalize_text(item.text)
            if text:
                infos.append(text[: self.max_filter_info_chars])
        return infos

    def _build_query_info_sentence(self, *, info_list: list[str], query: str) -> str:
        info_str = "; ".join(f"Info: {normalize_text(info)}" for info in info_list if normalize_text(info))
        return normalize_text(f"Query: {query} {info_str}")

    def _build_filter_input(self, sentence: str) -> dict[str, Any]:
        assert self._tokenizer is not None
        words = [CLS_TOKEN] + self._spacify(sentence) + [SEP_TOKEN]
        tokens: list[str] = []
        for word in words:
            tokens.extend(self._tokenizer.tokenize(word))
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens[:FILTER_MAX_LENGTH])
        return self._tokenizer.pad(
            {"input_ids": [input_ids]},
            max_length=FILTER_MAX_LENGTH,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )

    def _spacify(self, text: str) -> list[str]:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            return [word.text for word in nlp(text) if word.lemma_ != ","]
        except Exception:
            return re.findall(r"[A-Za-z0-9_.-]+|[^\w\s]", normalize_text(text))

    def _predict_filtered_query(self, tokenized: dict[str, Any]) -> str:
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._model_device is not None

        tokenized_on_device = {
            key: value.to(self._model_device)
            for key, value in tokenized.items()
        }
        with torch.no_grad():
            outputs = self._model(**tokenized_on_device)
        labels = outputs.logits.argmax(dim=-1).detach().cpu()[0]
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        selected_ids = input_ids[(labels == 1) & (attention_mask == 1)]
        return self._tokenizer.decode(selected_ids, skip_special_tokens=True)

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
