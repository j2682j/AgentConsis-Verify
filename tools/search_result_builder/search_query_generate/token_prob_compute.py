from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen3-4B"

SENTENCE = """A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes,
where each axis has a label word at both ends.
Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?"""

TARGET_UNITS = [
    "AI regulation",
    "arXiv.org",
    "June 2022",
    "three axes",
    "label word",
    "Physics and Society",
    "August 11, 2016",
    "type of society",
]


@dataclass
class TokenInfo:
    """
    儲存單一 token 與原始文字位置、機率分數的對齊資訊。

    Args:
        - token_index: token 在輸入序列中的索引。
        - token_id: tokenizer 對應的 token id。
        - token_text: decode 後的 token 文字。
        - original_text: token 對齊到原始句子的文字片段。
        - start_char: token 在原始句子的起始字元位置。
        - end_char: token 在原始句子的結束字元位置。
        - logprob: 模型預測此 token 的 log probability。
        - prob: 模型預測此 token 的 probability。

    Returns:
        - TokenInfo: token-level probability analysis result。
    """

    token_index: int
    token_id: int
    token_text: str
    original_text: str
    start_char: int
    end_char: int
    logprob: float | None
    prob: float | None


@dataclass
class TextUnitScore:
    """
    儲存一個文字單位跨多個 token 後的機率統計。

    Args:
        - text_unit: 要評分的原始文字單位。
        - char_span: 文字單位在原始句子的字元區間。
        - token_span: 文字單位覆蓋到的 token 區間。
        - tokens: 文字單位覆蓋到的 token 文字。
        - original_parts: token 對齊到的原始文字片段。
        - logprob_sum: 所有相關 token log probability 總和。
        - logprob_avg: 所有相關 token log probability 平均。
        - logprob_min: 相關 token 中最低的 log probability。

    Returns:
        - TextUnitScore: text-unit probability analysis result。
    """

    text_unit: str
    char_span: tuple[int, int]
    token_span: tuple[int, int] | None
    tokens: list[str]
    original_parts: list[str]
    logprob_sum: float
    logprob_avg: float | None
    logprob_min: float | None


@dataclass
class TokenProbabilityQueryCandidate:
    """
    儲存由低機率文字單位產生的 query candidate。

    Args:
        - query: 可送入搜尋流程的 query 文字。
        - text_unit: 產生 query 的原始文字單位。
        - source: candidate 來源，固定為 token_probability。
        - score: 排序分數，低機率文字會得到較高分。
        - logprob_avg: 文字單位的平均 log probability。
        - logprob_min: 文字單位內最低 token log probability。

    Returns:
        - TokenProbabilityQueryCandidate: token-probability-based query candidate。
    """

    query: str
    text_unit: str
    source: str = "token_probability"
    score: float = 0.0
    logprob_avg: float | None = None
    logprob_min: float | None = None


class TokenProbabilityAnalyzer:
    """
    使用 causal language model 計算句子 token 與指定文字單位的機率分數。

    Args:
        - model_name: HuggingFace causal LM 名稱。
        - device: 指定運算裝置，None 時自動使用 cuda 或 cpu。

    Returns:
        - TokenProbabilityAnalyzer: 可延遲載入模型並產生 token probability candidates 的分析器。
    """

    def __init__(self, *, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        """
        延遲載入 HuggingFace tokenizer 與 causal language model。

        Args:
            - 無。

        Returns:
            - None。
        """
        if self.tokenizer is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device

    def analyze_tokens(self, sentence: str) -> list[TokenInfo]:
        """
        對句子做 tokenization，並計算每個 token 的 next-token probability。

        Args:
            - sentence: 要分析的原始句子。

        Returns:
            - list[TokenInfo]: token 與原始文字對齊後的機率資訊。
        """
        self.load()

        import torch

        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0]
        encoded = encoded.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        input_ids = encoded["input_ids"][0]
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)

        token_infos: list[TokenInfo] = []
        for index, token_id in enumerate(input_ids):
            start_char, end_char = offsets[index].tolist()
            token_text = self.tokenizer.decode([token_id])
            original_text = sentence[start_char:end_char]

            if index == 0:
                token_logprob = None
                token_prob = None
            else:
                token_logprob = float(log_probs[index - 1, token_id].item())
                token_prob = float(torch.exp(log_probs[index - 1, token_id]).item())

            token_infos.append(
                TokenInfo(
                    token_index=index,
                    token_id=int(token_id),
                    token_text=token_text,
                    original_text=original_text,
                    start_char=int(start_char),
                    end_char=int(end_char),
                    logprob=token_logprob,
                    prob=token_prob,
                )
            )

        return token_infos

    def score_text_unit(
        self,
        sentence: str,
        text_unit: str,
        token_infos: list[TokenInfo] | None = None,
    ) -> TextUnitScore:
        """
        計算一個文字單位覆蓋到的 token log probability 統計。

        Args:
            - sentence: 原始句子。
            - text_unit: 要分析的文字單位。
            - token_infos: 已分析好的 token 資訊，未提供時會重新分析。

        Returns:
            - TextUnitScore: 文字單位的 token probability 統計。
        """
        token_infos = token_infos or self.analyze_tokens(sentence)
        start_char, end_char = self.find_text_span(sentence, text_unit)
        selected_tokens = self.tokens_overlapping_span(token_infos, start_char, end_char)
        logprobs = [token.logprob for token in selected_tokens if token.logprob is not None]

        logprob_sum = sum(logprobs)
        logprob_avg = logprob_sum / len(logprobs) if logprobs else None
        logprob_min = min(logprobs) if logprobs else None
        token_span = (
            selected_tokens[0].token_index,
            selected_tokens[-1].token_index + 1,
        ) if selected_tokens else None

        return TextUnitScore(
            text_unit=text_unit,
            char_span=(start_char, end_char),
            token_span=token_span,
            tokens=[token.token_text for token in selected_tokens],
            original_parts=[token.original_text for token in selected_tokens],
            logprob_sum=logprob_sum,
            logprob_avg=logprob_avg,
            logprob_min=logprob_min,
        )

    def score_text_units(
        self,
        sentence: str,
        text_units: list[str],
        *,
        sort_key: str = "logprob_avg",
    ) -> list[TextUnitScore]:
        """
        批次計算多個文字單位，並依照低機率優先排序。

        Args:
            - sentence: 原始句子。
            - text_units: 要分析的文字單位列表。
            - sort_key: 排序依據，可用 logprob_avg 或 logprob_min。

        Returns:
            - list[TextUnitScore]: 排序後的文字單位分數。
        """
        token_infos = self.analyze_tokens(sentence)
        scores = [
            self.score_text_unit(sentence, unit, token_infos)
            for unit in text_units
            if unit and unit in sentence
        ]
        scores.sort(key=lambda item: self._sortable_score(item, sort_key))
        return scores

    def generate_candidates(
        self,
        sentence: str,
        text_units: list[str],
        *,
        top_k: int = 8,
        sort_key: str = "logprob_avg",
    ) -> list[TokenProbabilityQueryCandidate]:
        """
        將低機率文字單位轉成 query candidates。

        Args:
            - sentence: 原始句子。
            - text_units: 候選文字單位，例如 NER entities 或模型候選片段。
            - top_k: 最多輸出的 candidate 數量。
            - sort_key: 排序依據，可用 logprob_avg 或 logprob_min。

        Returns:
            - list[TokenProbabilityQueryCandidate]: 低機率優先的 query candidates。
        """
        scores = self.score_text_units(sentence, text_units, sort_key=sort_key)
        candidates: list[TokenProbabilityQueryCandidate] = []
        for score in scores[:top_k]:
            value = getattr(score, sort_key)
            candidate_score = 0.0 if value is None else abs(float(value))
            candidates.append(
                TokenProbabilityQueryCandidate(
                    query=score.text_unit,
                    text_unit=score.text_unit,
                    score=round(candidate_score, 6),
                    logprob_avg=score.logprob_avg,
                    logprob_min=score.logprob_min,
                )
            )
        return candidates

    def find_text_span(self, sentence: str, text: str) -> tuple[int, int]:
        """
        找出文字單位在原始句子中的位置。

        Args:
            - sentence: 原始句子。
            - text: 要尋找的文字單位。

        Returns:
            - tuple[int, int]: 起始與結束字元位置。
        """
        start = sentence.index(text)
        return start, start + len(text)

    def tokens_overlapping_span(
        self,
        token_infos: list[TokenInfo],
        start_char: int,
        end_char: int,
    ) -> list[TokenInfo]:
        """
        找出與指定字元區間重疊的 tokens。

        Args:
            - token_infos: token 對齊資訊。
            - start_char: 字元區間起點。
            - end_char: 字元區間終點。

        Returns:
            - list[TokenInfo]: 與 span 重疊的 tokens。
        """
        return [
            token
            for token in token_infos
            if token.start_char < end_char and token.end_char > start_char
        ]

    def _sortable_score(self, item: TextUnitScore, sort_key: str) -> tuple[bool, float]:
        value = getattr(item, sort_key)
        return value is None, value if value is not None else float("inf")


def load_model(model_name: str = MODEL_NAME):
    """
    相容舊版函式：載入模型並回傳 tokenizer、model、device。

    Args:
        - model_name: HuggingFace causal LM 名稱。

    Returns:
        - tuple[Any, Any, str]: tokenizer、model、device。
    """
    analyzer = TokenProbabilityAnalyzer(model_name=model_name)
    analyzer.load()
    return analyzer.tokenizer, analyzer.model, analyzer.device


def analyze_tokens(sentence: str, tokenizer: Any, model: Any, device: str) -> list[TokenInfo]:
    """
    相容舊版函式：使用外部傳入的 tokenizer/model 分析 token。

    Args:
        - sentence: 原始句子。
        - tokenizer: HuggingFace tokenizer。
        - model: HuggingFace causal LM。
        - device: 運算裝置。

    Returns:
        - list[TokenInfo]: token 機率資訊。
    """
    analyzer = TokenProbabilityAnalyzer(device=device)
    analyzer.tokenizer = tokenizer
    analyzer.model = model
    analyzer.device = device
    return analyzer.analyze_tokens(sentence)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token probability for target text units.")
    parser.add_argument("--sentence", default="", help="Sentence to analyze. If omitted, stdin or default sentence is used.")
    parser.add_argument("--sentence-file", default="", help="Read sentence text from a UTF-8 text file.")
    parser.add_argument("--unit", action="append", default=[], help="Target text unit. Can be provided multiple times.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--sort-key", choices=["logprob_avg", "logprob_min"], default="logprob_avg")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args(argv)


def resolve_sentence(args: argparse.Namespace) -> str:
    if args.sentence:
        return args.sentence.strip()
    if args.sentence_file:
        return Path(args.sentence_file).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    return SENTENCE


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sentence = resolve_sentence(args)
    units = args.unit or TARGET_UNITS
    analyzer = TokenProbabilityAnalyzer(model_name=args.model_name)
    candidates = analyzer.generate_candidates(
        sentence,
        units,
        top_k=args.top_k,
        sort_key=args.sort_key,
    )
    print(json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
