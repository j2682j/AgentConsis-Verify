from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from utils.network_utils import normalize_text


DEFAULT_HF_MODEL_NAME = "BAAI/bge-m3"
_HF_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


@dataclass
class TokenSalient:
    """
    儲存 tokenizer 單一 token 的 salience 分數。

    Args:
        - token_index: token 在 tokenizer input 中的位置。
        - token: tokenizer 原始 token 字串。
        - text: token 對應到原始問題的文字。
        - start: token 在原始問題中的起始字元位置。
        - end: token 在原始問題中的結束字元位置。
        - embedding_similarity: 刪除 token 後與原問題 embedding 的 cosine similarity。
        - embedding_delta: 1 - embedding_similarity，表示 token 對語意表示的影響量。
        - score: 加上輕量 bonus / penalty 後的 salience 分數。
        - keep: 是否通過 stopword、標點與長度過濾。
        - reason: 過濾或保留原因。

    Returns:
        - TokenSalient: token-level salience 分析結果。
    """

    token_index: int
    token: str
    text: str
    start: int
    end: int
    embedding_similarity: float
    embedding_delta: float
    score: float
    keep: bool
    reason: str = ""


class SemanticImpactScorer:
    """
    使用 encoder embedding deletion delta 計算每個 token 對問題語意表示的影響。

    Args:
        - hf_model_name: HuggingFace encoder / embedding model 名稱。
        - max_input_tokens: tokenizer 與 encoder 的最大輸入 token 數。
        - max_salient_tokens: 過濾後最多保留的 token 數。
        - min_token_chars: token 原文最小字元長度。
        - device: 執行裝置，未指定時自動選 cuda / cpu。

    Returns:
        - SemanticImpactScorer: 可重用的 semantic impact scorer。
    """

    STOPWORDS = {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "give",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "must",
        "not",
        "of",
        "on",
        "or",
        "please",
        "provide",
        "question",
        "result",
        "should",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "use",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
    GENERIC_QUERY_TERMS = {
        "answer",
        "article",
        "base",
        "candidate",
        "data",
        "day",
        "doesn",
        "document",
        "each",
        "example",
        "exchange",
        "find",
        "here",
        "information",
        "line",
        "page",
        "question",
        "result",
        "search",
        "selected",
        "source",
        "studie",
        "title",
        "unknown",
    }
    PUNCTUATION_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

    def __init__(
        self,
        *,
        hf_model_name: str | None = None,
        max_input_tokens: int = 256,
        max_salient_tokens: int = 12,
        min_token_chars: int = 2,
        device: str | None = None,
    ) -> None:
        self.hf_model_name = hf_model_name or os.getenv("SEARCH_SALIENCE_HF_MODEL", DEFAULT_HF_MODEL_NAME)
        self.max_input_tokens = max_input_tokens
        self.max_salient_tokens = max_salient_tokens
        self.min_token_chars = min_token_chars
        self.device = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def score_tokens(self, question: str) -> list[TokenSalient]:
        """
        使用 HF encoder / embedding model 計算刪除每個 token 後的 embedding delta。

        Args:
            - question: 原始問題。

        Returns:
            - list[TokenSalient]: 每個 token 的 salience 分析結果。
        """
        self._load_hf_model()
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.device is not None

        encoded = self.tokenizer(
            question,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        offsets = encoded["offset_mapping"][0].tolist()
        input_ids = encoded["input_ids"][0].tolist()
        baseline_embedding = self._embed_texts([question])[0]
        perturbed_embeddings = self._embed_texts(
            [
                self._delete_span(question, int(start), int(end))
                for start, end in offsets
            ]
        )

        token_salience: list[TokenSalient] = []
        for index, token_id in enumerate(input_ids):
            start, end = offsets[index]
            token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            original_text = question[int(start) : int(end)]
            similarity = float((baseline_embedding * perturbed_embeddings[index]).sum().detach().cpu().item())
            embedding_delta = max(0.0, 1.0 - similarity)
            keep, reason = self._token_keep_reason(original_text)
            score = self._token_score(original_text, embedding_delta, keep=keep)
            token_salience.append(
                TokenSalient(
                    token=str(token),
                    token_index=index,
                    text=normalize_text(original_text),
                    start=int(start),
                    end=int(end),
                    embedding_similarity=round(similarity, 6),
                    embedding_delta=round(embedding_delta, 6),
                    score=round(score, 6),
                    keep=keep,
                    reason=reason,
                )
            )
        return token_salience

    def filter_tokens(self, tokens: list[TokenSalient]) -> list[TokenSalient]:
        """
        過濾 stopword、標點、過短 token，並保留最高 salience token。

        Args:
            - tokens: score_tokens() 產生的 token salience。

        Returns:
            - list[TokenSalient]: 可進入 span repair 的 token。
        """
        kept = [token for token in tokens if token.keep and token.score > 0]
        kept.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        return kept[: self.max_salient_tokens]

    def semantic_similarities(self, reference: str, texts: list[str]) -> list[float]:
        """
        計算參考文字與多段文字的 encoder cosine similarity。

        Args:
            - reference: 作為比較基準的文字。
            - texts: 要與基準比較的文字列表。

        Returns:
            - list[float]: 與 texts 順序相同的 cosine similarity。
        """
        if not texts:
            return []
        self._load_hf_model()
        embeddings = self._embed_texts([reference, *texts])
        reference_embedding = embeddings[0]
        return [
            float((reference_embedding * embedding).sum().detach().cpu().item())
            for embedding in embeddings[1:]
        ]

    def _load_hf_model(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (self.hf_model_name, device)
        cached = _HF_MODEL_CACHE.get(cache_key)
        if cached is not None:
            self.tokenizer, self.model, self.device = cached
            return

        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, use_fast=True)
        model = AutoModel.from_pretrained(self.hf_model_name, torch_dtype=dtype).to(device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        _HF_MODEL_CACHE[cache_key] = (tokenizer, model, device)

    def _token_keep_reason(self, text: str) -> tuple[bool, str]:
        cleaned = normalize_text(text).strip()
        lowered = cleaned.lower()
        if not cleaned:
            return False, "empty"
        if len(cleaned) < self.min_token_chars and not any(char.isdigit() for char in cleaned):
            return False, "too_short"
        if lowered in self.STOPWORDS:
            return False, "stopword"
        if self.PUNCTUATION_RE.fullmatch(cleaned):
            return False, "punctuation"
        if lowered in self.GENERIC_QUERY_TERMS:
            return False, "generic"
        return True, "kept"

    def _token_score(self, text: str, embedding_delta: float, *, keep: bool) -> float:
        """
        對包含數字、大寫字母、較長表面形式的 token 給予輕量 search-oriented bonus。

        Args:
            - text: token 對應的原始文字。
            - embedding_delta: 刪除 token 後的 embedding delta。
            - keep: 是否通過過濾。

        Returns:
            - float: token 最終 salience 分數。
        """
        if not keep:
            return 0.0
        score = embedding_delta
        if any(char.isdigit() for char in text):
            score += 0.02
        if re.search(r"[A-Z]", text):
            score += 0.015
        if len(text) >= 8:
            score += 0.005
        return score

    def _embed_texts(self, texts: list[str]):
        import torch

        assert self.tokenizer is not None
        assert self.model is not None
        assert self.device is not None

        normalized_texts = [normalize_text(text) or " " for text in texts]
        encoded = self.tokenizer(
            normalized_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _delete_span(self, question: str, start: int, end: int) -> str:
        if start < 0 or end <= start:
            return question
        deleted = normalize_text(f"{question[:start]} {question[end:]}")
        return deleted or question


__all__ = ["DEFAULT_HF_MODEL_NAME", "SemanticImpactScorer", "TokenSalient"]
