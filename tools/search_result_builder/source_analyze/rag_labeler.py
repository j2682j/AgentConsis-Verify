from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.network_utils import normalize_text


LABELER_MAX_LENGTH = 512
CONTINUE_TAG = "<CONTINUE>"
FINISH_TAG = "<FINISH>"
TERMINATE_TAG = "<TERMINATE>"
SEQUENCE_LABELS_TWO = {0: CONTINUE_TAG, 1: TERMINATE_TAG}
SEQUENCE_LABELS_THREE = {0: CONTINUE_TAG, 1: TERMINATE_TAG, 2: FINISH_TAG}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_LABELER_CHECKPOINT = PROJECT_ROOT / "models" / "labeler_v2"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}
_SPACY_CACHE: dict[str, Any] = {}


def _default_labeler_checkpoint() -> Path:
    """
    回傳專案內固定的 EfficientRAG Labeler checkpoint。

    Args:
        - 無。

    Returns:
        - Path: `<repo>/models/labeler_v2` 絕對路徑。
    """
    return PROJECT_LABELER_CHECKPOINT


@dataclass
class RAGLabelResult:
    """
    保存 EfficientRAG labeler 對單一 chunk 的判斷結果。

    Args:
        - label: useful 或 useless。
        - kept_tokens: Token head 抽取的 useful tokens。
        - dropped_tokens: 未被抽取的 chunk tokens。
        - metadata: 模型、sequence tag、機率與 fallback 資訊。

    Returns:
        - RAGLabelResult: Labeler 的結構化結果。
    """

    label: str
    kept_tokens: list[str] = field(default_factory=list)
    dropped_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class EfficientRAGLabelerAdapter:
    """
    使用 EfficientRAG 預訓練 DeBERTa 雙頭模型標註 retrieved chunks。

    Args:
        - labeler_checkpoint: EfficientRAG labeler checkpoint 路徑。
        - device: 推論裝置；預設讀取 SEARCH_LABELER_DEVICE，否則使用 CPU。
        - max_length: Labeler 最大輸入 token 數。
        - batch_size: 單次推論的 chunk 數量。
        - spacy_model: 切分原始文字單位使用的 spaCy model。

    Returns:
        - EfficientRAGLabelerAdapter: 支援 batch label 與 fallback 的 adapter。
    """

    STOPWORDS = {
        "the", "and", "for", "with", "from", "what", "which", "who",
        "when", "where", "why", "how", "answer", "question", "this",
        "that", "are", "was", "were",
    }

    def __init__(
        self,
        *,
        labeler_checkpoint: str | None = None,
        device: str | None = None,
        max_length: int = LABELER_MAX_LENGTH,
        batch_size: int = 8,
        spacy_model: str = "en_core_web_sm",
        max_question_tokens: int = 128,
        min_passage_tokens: int = 320,
    ) -> None:
        self.labeler_checkpoint = str(
            Path(labeler_checkpoint)
            if labeler_checkpoint
            else _default_labeler_checkpoint()
        )
        self.device = device or os.getenv("SEARCH_LABELER_DEVICE", "cpu")
        self.max_length = max(32, max_length)
        self.batch_size = max(1, batch_size)
        self.spacy_model = spacy_model
        self.max_question_tokens = max(16, max_question_tokens)
        self.min_passage_tokens = max(32, min_passage_tokens)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._model_device: str | None = None
        self._sequence_label_count: int = 3

    def label_text(
        self,
        *,
        question: str,
        text: str,
        useful_probability: float | None = None,
        threshold: float | None = None,
    ) -> RAGLabelResult:
        """標註單一 evidence chunk。"""
        return self.label_texts(
            question=question,
            texts=[text],
            useful_probability=useful_probability,
            threshold=threshold,
        )[0]

    def label_texts(
        self,
        *,
        question: str,
        texts: list[str],
        useful_probability: float | None = None,
        threshold: float | None = None,
    ) -> list[RAGLabelResult]:
        """以 batch 方式標註多個 evidence chunks。"""
        if not texts:
            return []
        try:
            self._load_model()
            results: list[RAGLabelResult] = []
            for start in range(0, len(texts), self.batch_size):
                results.extend(
                    self._predict_batch(
                        question=question,
                        texts=texts[start : start + self.batch_size],
                    )
                )
            return results
        except Exception as exc:
            return [
                self._fallback_label(
                    question=question,
                    text=text,
                    useful_probability=useful_probability,
                    threshold=threshold,
                    model_error=f"{type(exc).__name__}: {exc}",
                )
                for text in texts
            ]

    def _load_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        checkpoint = str(Path(self.labeler_checkpoint))
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"EfficientRAG labeler checkpoint not found: {checkpoint}"
            )
        try:
            import sentencepiece  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "EfficientRAG labeler requires SentencePiece. "
                "Install it with `python -m pip install sentencepiece`."
            ) from exc

        import torch
        from transformers import DebertaV2Tokenizer

        from .labeler_model import DebertaForSequenceTokenClassification

        requested_device = self.device.strip().lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        sequence_label_count = self._checkpoint_sequence_label_count(checkpoint)
        cache_key = (checkpoint, requested_device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._tokenizer, self._model, self._model_device = cached
            self._sequence_label_count = getattr(
                self._model,
                "sequence_labels",
                sequence_label_count,
            )
            return
        tokenizer = DebertaV2Tokenizer.from_pretrained(checkpoint)
        model = DebertaForSequenceTokenClassification.from_pretrained(
            checkpoint,
            token_labels=2,
            sequence_labels=sequence_label_count,
        ).to(requested_device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._model_device = requested_device
        self._sequence_label_count = sequence_label_count
        _MODEL_CACHE[cache_key] = (tokenizer, model, requested_device)

    def _checkpoint_sequence_label_count(self, checkpoint: str) -> int:
        """
        從 checkpoint classifier shape 判斷 sequence label 類別數。

        Args:
            - checkpoint: Labeler checkpoint 目錄。

        Returns:
            - int: sequence classifier 類別數，預設為 3。
        """
        try:
            from safetensors import safe_open

            model_path = Path(checkpoint) / "model.safetensors"
            if model_path.exists():
                with safe_open(model_path, framework="pt", device="cpu") as file:
                    if "sequence_classifier.bias" in file.keys():
                        return int(file.get_tensor("sequence_classifier.bias").shape[0])
                    if "sequence_classifier.weight" in file.keys():
                        return int(file.get_tensor("sequence_classifier.weight").shape[0])
        except Exception:
            return 3
        return 3

    def _predict_batch(
        self,
        *,
        question: str,
        texts: list[str],
    ) -> list[RAGLabelResult]:
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._model_device is not None
        tokenized = self._build_inputs(question=question, texts=texts)
        model_inputs = {
            key: value.to(self._model_device)
            for key, value in tokenized.items()
        }
        with torch.no_grad():
            outputs = self._model(**model_inputs)
        probabilities = torch.softmax(
            outputs.sequence_logits,
            dim=-1,
        ).detach().cpu()
        sequence_labels = probabilities.argmax(dim=-1)
        token_labels = outputs.token_logits.argmax(dim=-1).detach().cpu()
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        results: list[RAGLabelResult] = []
        for index, text in enumerate(texts):
            kept_tokens, span_metadata = self._extract_useful_spans(
                text=text,
                input_ids=input_ids[index],
                token_labels=token_labels[index],
                attention_mask=attention_mask[index],
            )
            kept_tokens = kept_tokens[:12]
            text_tokens = self._ordered_keywords(text)
            kept_set = {
                token
                for span in kept_tokens
                for token in self._ordered_keywords(span)
            }
            sequence_id = int(sequence_labels[index].item())
            sequence_label_map = (
                SEQUENCE_LABELS_THREE
                if self._sequence_label_count == 3
                else SEQUENCE_LABELS_TWO
            )
            sequence_tag = sequence_label_map.get(sequence_id, TERMINATE_TAG)
            is_useful_sequence = sequence_tag in {CONTINUE_TAG, FINISH_TAG}
            passage_start, passage_end = self._passage_token_bounds(
                input_ids=input_ids[index],
                attention_mask=attention_mask[index],
            )
            question_token_count = max(0, passage_start - 2)
            passage_token_count = max(0, passage_end - passage_start)
            original_question_tokens = len(
                self._tokenize_words(self._spacify(question))
            )
            original_passage_tokens = len(
                self._tokenize_words(self._spacify(text))
            )
            results.append(
                RAGLabelResult(
                    label=(
                        "useful"
                        if is_useful_sequence
                        else "useless"
                    ),
                    kept_tokens=kept_tokens,
                    dropped_tokens=[
                        token for token in text_tokens if token not in kept_set
                    ][:20],
                    metadata={
                        "method": "efficientrag_labeler_model",
                        "checkpoint": self.labeler_checkpoint,
                        "device": self._model_device,
                        "sequence_tag": sequence_tag,
                        "continue_probability": round(
                            float(probabilities[index][0].item()), 6
                        ),
                        "terminate_probability": round(
                            float(probabilities[index][1].item()), 6
                        ),
                        "finish_probability": (
                            round(float(probabilities[index][2].item()), 6)
                            if probabilities.shape[-1] > 2
                            else 0.0
                        ),
                        "question_token_count": question_token_count,
                        "passage_token_count": passage_token_count,
                        "question_truncated": (
                            original_question_tokens > question_token_count
                        ),
                        "passage_truncated": (
                            original_passage_tokens > passage_token_count
                        ),
                        **span_metadata,
                    },
                )
            )
        return results

    def _extract_useful_spans(
        self,
        *,
        text: str,
        input_ids: Any,
        token_labels: Any,
        attention_mask: Any,
    ) -> tuple[list[str], dict[str, object]]:
        """
        從 passage 區域的連續 positive tokens 還原完整原文 spans。

        Args:
            - text: Labeler 收到的原始 passage。
            - input_ids: 單筆 tokenizer input IDs。
            - token_labels: Token head 的 argmax labels。
            - attention_mask: 單筆 attention mask。

        Returns:
            - list[str]: 能在原始 passage 對齊的完整 word/span。
            - dict[str, object]: decoded、repaired 與 rejected span 紀錄。
        """
        assert self._tokenizer is not None
        passage_start, passage_end = self._passage_token_bounds(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        positive_positions = [
            position
            for position in range(passage_start, passage_end)
            if (
                int(attention_mask[position].item()) == 1
                and int(token_labels[position].item()) == 1
            )
        ]
        ranges = self._contiguous_ranges(positive_positions)
        decoded_spans: list[str] = []
        repaired_spans: list[str] = []
        rejected_spans: list[str] = []
        kept_spans: list[str] = []
        seen: set[str] = set()

        for start, end in ranges:
            decoded = normalize_text(
                self._tokenizer.decode(
                    input_ids[start:end],
                    skip_special_tokens=True,
                )
            ).strip(" \t\r\n.,;:!?\"'`()[]{}")
            if not decoded:
                continue
            decoded_spans.append(decoded)
            aligned, repaired = self._align_span_to_text(
                decoded_span=decoded,
                text=text,
            )
            if not aligned:
                rejected_spans.append(decoded)
                continue
            normalized = normalize_text(aligned).casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            kept_spans.append(normalized)
            if repaired:
                repaired_spans.append(
                    f"{decoded} -> {normalize_text(aligned)}"
                )

        return kept_spans, {
            "decoded_useful_spans": decoded_spans,
            "repaired_useful_spans": repaired_spans,
            "rejected_useful_spans": rejected_spans,
            "passage_token_start": passage_start,
            "passage_token_end": passage_end,
            "positive_token_count": len(positive_positions),
        }

    def _passage_token_bounds(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
    ) -> tuple[int, int]:
        assert self._tokenizer is not None
        active_length = int(attention_mask.sum().item())
        sep_token_id = int(self._tokenizer.sep_token_id)
        separator_positions = [
            position
            for position in range(active_length)
            if int(input_ids[position].item()) == sep_token_id
        ]
        if len(separator_positions) < 2:
            return 0, active_length
        return separator_positions[0] + 1, separator_positions[-1]

    def _contiguous_ranges(
        self,
        positions: list[int],
    ) -> list[tuple[int, int]]:
        if not positions:
            return []
        ranges: list[tuple[int, int]] = []
        start = positions[0]
        previous = positions[0]
        for position in positions[1:]:
            if position == previous + 1:
                previous = position
                continue
            ranges.append((start, previous + 1))
            start = position
            previous = position
        ranges.append((start, previous + 1))
        return ranges

    def _align_span_to_text(
        self,
        *,
        decoded_span: str,
        text: str,
    ) -> tuple[str, bool]:
        """
        將 decoded span 對齊原文；必要時執行唯一且可信的前綴修復。

        Args:
            - decoded_span: SentencePiece tokens 解碼後的片段。
            - text: 原始 passage。

        Returns:
            - str: 對齊後的完整原文 word/span，失敗時為空字串。
            - bool: 是否使用前綴修復。
        """
        fragment = normalize_text(decoded_span).strip()
        source = normalize_text(text)
        if not fragment or not source:
            return "", False

        exact_pattern = re.compile(
            rf"(?<!\w){re.escape(fragment)}(?!\w)",
            flags=re.IGNORECASE,
        )
        exact_matches = list(exact_pattern.finditer(source))
        if exact_matches:
            return exact_matches[0].group(0), False

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._'-]*", fragment):
            return "", False
        if len(fragment) < 4:
            return "", False

        candidates: dict[str, str] = {}
        fragment_key = fragment.casefold()
        for match in re.finditer(
            r"(?<!\w)[A-Za-z0-9][A-Za-z0-9._'-]*(?!\w)",
            source,
        ):
            word = match.group(0)
            word_key = word.casefold()
            if not word_key.startswith(fragment_key):
                continue
            if len(fragment_key) / len(word_key) < 0.6:
                continue
            candidates.setdefault(word_key, word)

        if len(candidates) != 1:
            return "", False
        return next(iter(candidates.values())), True

    def _build_inputs(
        self,
        *,
        question: str,
        texts: list[str],
    ) -> dict[str, Any]:
        assert self._tokenizer is not None
        question_tokens = self._tokenize_words(self._spacify(question))
        cls_tokens = self._tokenizer.tokenize("[CLS]")
        sep_tokens = self._tokenizer.tokenize("[SEP]")
        special_token_count = len(cls_tokens) + 2 * len(sep_tokens)
        available_tokens = max(0, self.max_length - special_token_count)
        max_question_tokens = min(
            self.max_question_tokens,
            max(0, available_tokens - self.min_passage_tokens),
        )
        question_tokens = self._truncate_question_tokens(
            question_tokens,
            max_question_tokens,
        )
        passage_budget = max(
            0,
            self.max_length - special_token_count - len(question_tokens),
        )

        input_ids: list[list[int]] = []
        for text in texts:
            passage_tokens = self._tokenize_words(
                self._spacify(text)
            )[:passage_budget]
            tokens = (
                cls_tokens
                + question_tokens
                + sep_tokens
                + passage_tokens
                + sep_tokens
            )
            input_ids.append(
                self._tokenizer.convert_tokens_to_ids(tokens)
            )
        return self._tokenizer.pad(
            {"input_ids": input_ids},
            max_length=self.max_length,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )

    def _truncate_question_tokens(
        self,
        tokens: list[str],
        max_tokens: int,
    ) -> list[str]:
        if max_tokens <= 0:
            return []
        if len(tokens) <= max_tokens:
            return tokens
        head_count = max(1, max_tokens * 2 // 3)
        tail_count = max_tokens - head_count
        if tail_count <= 0:
            return tokens[:max_tokens]
        return [*tokens[:head_count], *tokens[-tail_count:]]

    def _tokenize_words(self, words: list[str]) -> list[str]:
        assert self._tokenizer is not None
        tokens: list[str] = []
        for word in words:
            tokens.extend(self._tokenizer.tokenize(word))
        return tokens

    def _spacify(self, text: str) -> list[str]:
        nlp = self._get_nlp()
        if nlp is not None:
            return [token.text for token in nlp(text) if token.lemma_ != ","]
        return re.findall(
            r"[A-Za-z0-9_.-]+|[^\w\s]",
            normalize_text(text),
        )

    def _get_nlp(self) -> Any | None:
        if self.spacy_model in _SPACY_CACHE:
            return _SPACY_CACHE[self.spacy_model]
        try:
            import spacy

            nlp = spacy.load(self.spacy_model)
        except Exception:
            nlp = None
        _SPACY_CACHE[self.spacy_model] = nlp
        return nlp

    def _fallback_label(
        self,
        *,
        question: str,
        text: str,
        useful_probability: float | None,
        threshold: float | None,
        model_error: str,
    ) -> RAGLabelResult:
        question_terms = set(self._ordered_keywords(question))
        text_terms = self._ordered_keywords(text)
        kept_tokens = [
            token for token in text_terms if token in question_terms
        ][:12]
        if (
            not kept_tokens
            and useful_probability is not None
            and threshold is not None
            and useful_probability >= threshold
        ):
            kept_tokens = text_terms[:12]
        if useful_probability is None or threshold is None:
            label = "useful" if kept_tokens else "useless"
        else:
            label = (
                "useful"
                if useful_probability >= threshold and kept_tokens
                else "useless"
            )
        return RAGLabelResult(
            label=label,
            kept_tokens=kept_tokens,
            dropped_tokens=[
                token for token in text_terms if token not in kept_tokens
            ][:20],
            metadata={
                "method": "efficientrag_labeler_fallback",
                "checkpoint": self.labeler_checkpoint,
                "model_error": model_error,
                "useful_probability": useful_probability,
                "threshold": threshold,
            },
        )

    def _ordered_keywords(self, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for token in re.findall(
            r"[a-z0-9][a-z0-9._-]{1,}",
            normalize_text(text).lower(),
        ):
            if token in self.STOPWORDS or len(token) <= 2 or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result


__all__ = ["EfficientRAGLabelerAdapter", "RAGLabelResult"]
