from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
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


@dataclass
class SalientSpan:
    """
    儲存由多個 salient token 合併出的可讀文字片段。

    Args:
        - text: 從原始問題切出的 span 文字。
        - start: span 在原始問題中的起始字元位置。
        - end: span 在原始問題中的結束字元位置。
        - score: span 內 token salience 分數總和。
        - tokens: span 內的 tokenizer token 字串。
        - token_indices: span 內 token 的原始 index。

    Returns:
        - SalientSpan: 可交給 query model 使用的重要片段。
    """

    text: str
    start: int
    end: int
    score: float
    tokens: list[str]
    token_indices: list[int]



@dataclass
class SalienceQueryCandidate:
    """
    儲存由 salient spans 產生並經 coverage 排序的 query。

    Args:
        - query: 可直接丟給搜尋引擎的 query。
        - matched_spans: query 覆蓋到的重要 span。
        - coverage_score: query 對 salient spans 的加權覆蓋率。
        - score: 最終 query 排序分數。
        - source: query 來源。

    Returns:
        - SalienceQueryCandidate: mask/token salience query 候選。
    """

    query: str
    matched_spans: list[str]
    coverage_score: float
    score: float
    semantic_impact_score: float = 0.0
    source: str = "qwen3:4b"


class MaskSalienceQueryGenerator:
    """
    使用 encoder / embedding model 的 token deletion delta 找出重要 token，再交給 qwen3:4b 產生 query。

    Args:
        - hf_model_name: HuggingFace encoder / embedding model 名稱，預設讀取 SEARCH_SALIENCE_HF_MODEL。
        - query_model_name: Ollama query 生成模型，目前使用 qwen3:4b。
        - max_input_tokens: HF 模型分析問題時的最大 token 數。
        - max_salient_tokens: 進入 span merge 的最高 salience token 數量。
        - max_salient_spans: 最多保留的重要 span 數量。
        - max_query_candidates: 請 qwen3:4b 產生的 query 數量。
        - min_token_chars: token 原文最小字元長度。
        - merge_gap_chars: 合併相鄰 salient token 時允許的最大字元間隔。
        - device: HF 模型執行裝置，未指定時自動選 cuda / cpu。

    Returns:
        - MaskSalienceQueryGenerator: token salience query generator。
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
    WEAK_SINGLE_TERMS = {
        "algebra",
        "algebraic",
        "attached",
        "base",
        "camera",
        "day",
        "doesn",
        "document",
        "each",
        "exchange",
        "guarantee",
        "guarantees",
        "here",
        "line",
        "rest",
        "round",
        "selected",
        "studie",
    }
    PHRASE_EDGE_TERMS = {
        "about",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    MONTH_PATTERN = (
        r"January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
        r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
    )
    PUNCTUATION_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

    SYSTEM_PROMPT = """You are only a web search query generator.
    Return JSON only.
    Do not answer the question.
    Do not explain.
    """

    USER_TEMPLATE = """Question:
    {question}

    Top semantic-impact spans:
    {spans}

    Generate {num_candidates} concise web search queries.

    Rules:
    - Generate search queries, not answers.
    - Each query must include at least one top semantic-impact span.
    - Prefer exact names, dates, titles, organizations, acronyms, and constraints.
    - Keep each query short.
    - Do not include explanations.

    Return exactly this JSON shape:
    {{"queries": ["...", "..."]}}"""

    def __init__(
        self,
        *,
        hf_model_name: str | None = None,
        query_model_name: str = "qwen3:4b",
        max_input_tokens: int = 256,
        max_salient_tokens: int = 12,
        max_salient_spans: int = 5,
        max_query_candidates: int = 3,
        min_token_chars: int = 2,
        merge_gap_chars: int = 2,
        device: str | None = None,
    ) -> None:
        self.hf_model_name = hf_model_name or os.getenv("SEARCH_SALIENCE_HF_MODEL", DEFAULT_HF_MODEL_NAME)
        self.query_model_name = query_model_name
        self.max_input_tokens = max_input_tokens
        self.max_salient_tokens = max_salient_tokens
        self.max_salient_spans = max_salient_spans
        self.max_query_candidates = max_query_candidates
        self.min_token_chars = min_token_chars
        self.merge_gap_chars = merge_gap_chars
        self.device = device
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.last_token_salience: list[TokenSalient] = []
        self.last_salient_spans: list[SalientSpan] = []

    def generate(
        self,
        question: str,
        *,
        num_candidates: int = 5,
    ) -> list[SalienceQueryCandidate]:
        """
        根據問題產生 salient-span guided search query candidates。

        Args:
            - question: 原始問題。
            - num_candidates: 最多回傳的 query 數量。

        Returns:
            - list[SalienceQueryCandidate]: 依 coverage 分數排序後的 query candidates。
        """
        text = normalize_text(question)
        if not text:
            return []

        token_salience = self.score_tokens(text)
        kept_tokens = self.filter_tokens(token_salience)
        spans = self.select_top_spans(self.merge_salient_tokens(text, kept_tokens))
        raw_queries = self.generate_queries_with_qwen(text, spans, num_candidates=num_candidates)
        candidates = self.build_candidates(raw_queries, spans)

        if not candidates:
            candidates = self._fallback_candidates(text, spans)

        self.last_token_salience = token_salience
        self.last_salient_spans = spans
        return candidates[: max(1, num_candidates)]

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
            - list[TokenSalient]: 可進入 span merge 的 token。
        """
        kept = [token for token in tokens if token.keep and token.score > 0]
        kept.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        return kept[: self.max_salient_tokens]

    def merge_salient_tokens(
        self,
        question: str,
        tokens: list[TokenSalient],
    ) -> list[SalientSpan]:
        """
        將相鄰或接近的 salient tokens 合併成可讀 span。

        Args:
            - question: 原始問題。
            - tokens: 已過濾並排序的 salient tokens。

        Returns:
            - list[SalientSpan]: 合併後的 salient spans。
        """
        if not tokens:
            return []

        ordered = sorted(tokens, key=lambda item: (item.start, item.end))
        groups: list[list[TokenSalient]] = []
        current: list[TokenSalient] = []
        for token in ordered:
            if not current:
                current = [token]
                continue
            previous = current[-1]
            gap_text = question[previous.end : token.start]
            gap_ok = token.start - previous.end <= self.merge_gap_chars or re.fullmatch(r"[\s._:/-]*", gap_text or "")
            if gap_ok:
                current.append(token)
            else:
                groups.append(current)
                current = [token]
        if current:
            groups.append(current)

        spans: list[SalientSpan] = []
        for group in groups:
            start = min(token.start for token in group)
            end = max(token.end for token in group)
            start, end, text = self._repair_span(question, start, end)
            if not self._valid_span_text(text):
                continue
            spans.append(
                SalientSpan(
                    text=text,
                    start=start,
                    end=end,
                    score=round(sum(token.score for token in group), 6),
                    tokens=[token.token for token in group],
                    token_indices=[token.token_index for token in group],
                )
            )
        return self._dedupe_spans(spans)

    def select_top_spans(self, spans: list[SalientSpan]) -> list[SalientSpan]:
        """
        排序、去除包含關係，並選出最高分 salient spans。

        Args:
            - spans: merge_salient_tokens() 產生的 spans。

        Returns:
            - list[SalientSpan]: 最終給 qwen3:4b 的重要 spans。
        """
        deduped = self._dedupe_spans(spans)
        deduped.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        selected: list[SalientSpan] = []
        for span in deduped:
            span_key = self._normalize_for_match(span.text)
            if not span_key:
                continue
            contained = False
            for existing in selected:
                existing_key = self._normalize_for_match(existing.text)
                if span_key in existing_key and existing.score >= span.score:
                    contained = True
                    break
                if existing_key in span_key and span.score >= existing.score:
                    selected.remove(existing)
                    break
            if not contained:
                selected.append(span)
            if len(selected) >= self.max_salient_spans:
                break
        return selected

    def generate_queries_with_qwen(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        num_candidates: int,
    ) -> list[str]:
        """
        使用 qwen3:4b 根據 salient spans 產生 query。

        Args:
            - question: 原始問題。
            - spans: 已選出的重要 spans。
            - num_candidates: 需要產生的 query 數量。

        Returns:
            - list[str]: qwen3:4b 產生的 query 字串。
        """
        if not spans:
            return [question]

        messages = self._build_query_messages(
            question,
            spans,
            num_candidates=min(max(1, num_candidates), self.max_query_candidates),
        )
        try:
            raw_reply = self._invoke_query_model(messages)
        except Exception:
            return self._fallback_raw_queries(question, spans)
        return self._parse_query_json(raw_reply) or self._fallback_raw_queries(question, spans)

    def build_candidates(
        self,
        queries: list[str],
        spans: list[SalientSpan],
    ) -> list[SalienceQueryCandidate]:
        """
        根據 query 對 salient spans 的 coverage 排序。

        Args:
            - queries: 候選 query 字串。
            - spans: salient spans。

        Returns:
            - list[SalienceQueryCandidate]: 排序後 query candidates。
        """
        deduped = self._dedupe_queries(queries)
        candidates: list[SalienceQueryCandidate] = []
        total_score = sum(max(span.score, 0.0) for span in spans) or 1.0
        max_span_score = max((span.score for span in spans), default=1.0) or 1.0
        for query in deduped:
            matched = self._matched_spans(query, spans)
            matched_impact = sum(span.score for span in matched)
            coverage_score = matched_impact / total_score
            semantic_impact_score = (
                max((span.score for span in matched), default=0.0) / max_span_score
            )
            candidates.append(
                SalienceQueryCandidate(
                    query=normalize_text(query),
                    matched_spans=[span.text for span in matched],
                    coverage_score=round(max(0.0, min(coverage_score, 1.0)), 6),
                    score=0.0,
                    semantic_impact_score=round(max(0.0, semantic_impact_score), 6),
                    source=self.query_model_name,
                )
            )
        return candidates

    def diagnostics(self) -> dict[str, Any]:
        """
        回傳最近一次 generate() 的 token salience 與 span 診斷資訊。

        Args:
            - None.

        Returns:
            - dict[str, Any]: 可寫入 search plan 的診斷資訊。
        """
        return {
            "method": "embedding_delta_salience",
            "hf_model_name": self.hf_model_name,
            "query_model_name": self.query_model_name,
            "salient_spans": [asdict(span) for span in self.last_salient_spans],
            "token_salience": [asdict(token) for token in self.last_token_salience],
        }

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
        if not keep:
            return 0.0
        score = embedding_delta
        if any(char.isdigit() for char in text):
            score += 0.02
        if re.search(r"[A-Z]", text):
            score += 0.012
        if len(text) >= 8:
            score += 0.008
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

    def _valid_span_text(self, text: str) -> bool:
        cleaned = normalize_text(text).strip(" ,.;:!?()[]{}")
        if len(cleaned) < self.min_token_chars:
            return False
        lowered = cleaned.lower()
        if lowered in self.STOPWORDS or lowered in self.GENERIC_QUERY_TERMS:
            return False
        if self.PUNCTUATION_RE.fullmatch(cleaned):
            return False
        if re.fullmatch(r"\d{1,2}", cleaned):
            return False
        if len(cleaned) < 4 and not any(char.isdigit() for char in cleaned):
            return False
        if lowered in self.WEAK_SINGLE_TERMS:
            return False
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", cleaned)
        if len(words) == 1:
            word = words[0]
            if (
                word.lower() in self.WEAK_SINGLE_TERMS
                or word.lower() in self.STOPWORDS
                or word.lower() in self.GENERIC_QUERY_TERMS
            ):
                return False
            if len(word) <= 3 and not (word.isupper() or any(char.isdigit() for char in word)):
                return False
        return True

    def _repair_span(self, question: str, start: int, end: int) -> tuple[int, int, str]:
        start, end = self._expand_word_boundaries(question, start, end)

        quoted = self._quoted_span_containing(question, start, end)
        if quoted is not None:
            q_start, q_end = quoted
            if self._is_reasonable_expansion(question, start, end, q_start, q_end, max_chars=140):
                start, end = q_start, q_end

        start, end = self._expand_date_phrase(question, start, end)
        start, end = self._expand_capitalized_phrase(question, start, end)
        start, end = self._expand_domain_phrase(question, start, end)
        start, end = self._trim_span_edges(question, start, end)
        text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
        return start, end, text

    def _expand_word_boundaries(self, question: str, start: int, end: int) -> tuple[int, int]:
        start = max(0, start)
        end = min(len(question), end)
        while start > 0 and self._is_word_char(question[start - 1]):
            start -= 1
        while end < len(question) and self._is_word_char(question[end]):
            end += 1
        return start, end

    def _is_word_char(self, char: str) -> bool:
        return bool(re.match(r"[A-Za-z0-9_'-]", char))

    def _quoted_span_containing(self, question: str, start: int, end: int) -> tuple[int, int] | None:
        quote_pairs = [('"', '"')]
        for left_quote, right_quote in quote_pairs:
            left = question.rfind(left_quote, 0, start + 1)
            if left < 0:
                continue
            right = question.find(right_quote, max(end, left + 1))
            if right < 0:
                continue
            if left <= start and end <= right + 1:
                return left + 1, right
        return None

    def _expand_date_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        patterns = [
            rf"\b(?:{self.MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}\b",
            rf"\b(?:{self.MONTH_PATTERN})\s+\d{{4}}\b",
            r"\b\d{4}\s*(?:-|to|and)\s*\d{4}\b",
        ]
        return self._expand_by_patterns(question, start, end, patterns)

    def _expand_capitalized_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        word = r"[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|[A-Z]{2,}[A-Za-z0-9-]*|\d+"
        connector = r"(?:of|the|and|or|for|to|in|on|at|from|with|by)"
        pattern = rf"\b{word}(?:(?:\s+{connector})?\s+{word})*\b"
        return self._expand_by_patterns(question, start, end, [pattern], max_chars=90)

    def _expand_domain_phrase(self, question: str, start: int, end: int) -> tuple[int, int]:
        patterns = [
            r"\bWord\s+of\s+the\s+Day\b",
            r"\bFeatured\s+Article\b",
            r"\bofficial\s+script\b",
            r"\bminimum\s+perigee\b",
            r"\bSeries\s+\d+,\s*Episode\s+\d+\b",
            r"\b[a-zA-Z0-9_-]{8,}\b",
        ]
        return self._expand_by_patterns(question, start, end, patterns, max_chars=80)

    def _expand_by_patterns(
        self,
        question: str,
        start: int,
        end: int,
        patterns: list[str],
        *,
        max_chars: int = 120,
    ) -> tuple[int, int]:
        best_start, best_end = start, end
        for pattern in patterns:
            for match in re.finditer(pattern, question):
                m_start, m_end = match.span()
                if not self._overlaps(start, end, m_start, m_end):
                    continue
                if not self._is_reasonable_expansion(question, start, end, m_start, m_end, max_chars=max_chars):
                    continue
                current_len = best_end - best_start
                candidate_len = m_end - m_start
                if candidate_len > current_len:
                    best_start, best_end = m_start, m_end
        return best_start, best_end

    def _trim_span_edges(self, question: str, start: int, end: int) -> tuple[int, int]:
        while start < end:
            segment = question[start:end]
            match = re.match(
                rf"^\W*(?:{'|'.join(sorted(self.PHRASE_EDGE_TERMS))})\b\s*",
                segment,
                flags=re.IGNORECASE,
            )
            if not match:
                break
            start += match.end()

        while start < end:
            segment = question[start:end]
            match = re.search(
                rf"\s+\b(?:{'|'.join(sorted(self.PHRASE_EDGE_TERMS))})\b\W*$",
                segment,
                flags=re.IGNORECASE,
            )
            if not match:
                break
            end = start + match.start()
        return start, end

    def _overlaps(self, start: int, end: int, other_start: int, other_end: int) -> bool:
        return start < other_end and other_start < end

    def _is_reasonable_expansion(
        self,
        question: str,
        start: int,
        end: int,
        candidate_start: int,
        candidate_end: int,
        *,
        max_chars: int,
    ) -> bool:
        if candidate_start > start or candidate_end < end:
            return False
        candidate = normalize_text(question[candidate_start:candidate_end])
        if len(candidate) > max_chars:
            return False
        if candidate.count(" ") > 18:
            return False
        return True

    def _build_query_messages(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        num_candidates: int,
    ) -> list[dict[str, str]]:
        span_lines = "\n".join(f"- {span.text}" for span in spans[: self.max_salient_spans])
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    question=question,
                    spans=span_lines,
                    num_candidates=num_candidates,
                ),
            },
        ]

    def _invoke_query_model(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.query_model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 768,
            },
        }
        request = urllib.request.Request(
            self._ollama_chat_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds()) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        thinking = str(message.get("thinking") or "").strip()
        return content or self._extract_json_text(thinking)

    def _ollama_chat_url(self) -> str:
        base_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url.rstrip('/')}/api/chat"

    def _timeout_seconds(self) -> int:
        try:
            return int(os.getenv("OLLAMA_TIMEOUT", "180"))
        except ValueError:
            return 180

    def _extract_json_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start : end + 1]
        return ""

    def _parse_query_json(self, raw_reply: str) -> list[str]:
        cleaned = str(raw_reply or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed: Any | None = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    parsed = None

        raw_queries: Any
        if isinstance(parsed, dict):
            raw_queries = parsed.get("queries", [])
        elif isinstance(parsed, list):
            raw_queries = parsed
        else:
            raw_queries = []

        queries: list[str] = []
        for item in raw_queries:
            if isinstance(item, str):
                query = item
            elif isinstance(item, dict):
                query = str(item.get("query", "") or "")
            else:
                continue
            query = self._clean_query(query)
            if query:
                queries.append(query)
        return self._dedupe_queries(queries)

    def _fallback_raw_queries(self, question: str, spans: list[SalientSpan]) -> list[str]:
        span_texts = [span.text for span in spans if span.text]
        queries: list[str] = []
        if span_texts:
            queries.append(" ".join(span_texts[:4]))
            for span in span_texts[: self.max_query_candidates]:
                queries.append(span)
        queries.append(question)
        return self._dedupe_queries(queries)

    def _fallback_candidates(
        self,
        question: str,
        spans: list[SalientSpan],
    ) -> list[SalienceQueryCandidate]:
        return self.build_candidates(self._fallback_raw_queries(question, spans), spans)

    def _matched_spans(self, query: str, spans: list[SalientSpan]) -> list[SalientSpan]:
        query_key = self._normalize_for_match(query)
        matched: list[SalientSpan] = []
        for span in spans:
            span_key = self._normalize_for_match(span.text)
            if span_key and span_key in query_key:
                matched.append(span)
        return matched

    def _dedupe_spans(self, spans: list[SalientSpan]) -> list[SalientSpan]:
        best_by_key: dict[str, SalientSpan] = {}
        for span in spans:
            key = self._normalize_for_match(span.text)
            if not key:
                continue
            existing = best_by_key.get(key)
            if existing is None or span.score > existing.score:
                best_by_key[key] = span
        return list(best_by_key.values())

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for query in queries:
            cleaned = self._clean_query(query)
            key = self._normalize_for_match(cleaned)
            if cleaned and key and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    def _clean_query(self, query: str) -> str:
        cleaned = normalize_text(str(query or "")).strip().strip('"').strip("'")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) < 2 or len(cleaned) > 220:
            return ""
        lowered = cleaned.lower()
        if lowered.startswith(("answer:", "final answer", "the answer is")):
            return ""
        return cleaned

    def _normalize_for_match(self, value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower())
        return f" {' '.join(cleaned.split())} "


__all__ = [
    "MaskSalienceQueryGenerator",
    "SalienceQueryCandidate",
    "SalientSpan",
    "TokenSalient",
]
