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
        "candidate",
        "data",
        "example",
        "find",
        "information",
        "page",
        "question",
        "result",
        "search",
        "source",
        "title",
        "unknown",
    }
    PUNCTUATION_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

    SYSTEM_PROMPT = """You are only a web search query generator.
    Return JSON only.
    Do not answer the question.
    Do not explain.
    """

    USER_TEMPLATE = """Question:
    {question}

    Important spans:
    {spans}

    Generate {num_candidates} concise web search queries.

    Rules:
    - Generate search queries, not answers.
    - Each query must include at least one important span.
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
        max_salient_spans: int = 8,
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
        candidates = self.rank_queries(raw_queries, spans)

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
            text = normalize_text(question[start:end]).strip(" ,.;:!?()[]{}")
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

    def rank_queries(
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
        for query in deduped:
            matched = self._matched_spans(query, spans)
            coverage_score = sum(span.score for span in matched) / total_score
            score = coverage_score + self._query_bonus(query, matched) - self._generic_query_penalty(query)
            if score <= 0 and matched:
                score = coverage_score
            candidates.append(
                SalienceQueryCandidate(
                    query=normalize_text(query),
                    matched_spans=[span.text for span in matched],
                    coverage_score=round(max(0.0, min(coverage_score, 1.0)), 6),
                    score=round(max(0.0, score), 6),
                    source=self.query_model_name,
                )
            )
        candidates.sort(key=lambda item: (item.score, item.coverage_score, len(item.query)), reverse=True)
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
        if len(text) < self.min_token_chars:
            return False
        lowered = text.lower()
        if lowered in self.STOPWORDS or lowered in self.GENERIC_QUERY_TERMS:
            return False
        if self.PUNCTUATION_RE.fullmatch(text):
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
        return self.rank_queries(self._fallback_raw_queries(question, spans), spans)

    def _matched_spans(self, query: str, spans: list[SalientSpan]) -> list[SalientSpan]:
        query_key = self._normalize_for_match(query)
        matched: list[SalientSpan] = []
        for span in spans:
            span_key = self._normalize_for_match(span.text)
            if span_key and span_key in query_key:
                matched.append(span)
        return matched

    def _query_bonus(self, query: str, matched_spans: list[SalientSpan]) -> float:
        bonus = 0.0
        if re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b", query):
            bonus += 0.08
        if '"' in query or "'" in query:
            bonus += 0.04
        if len(matched_spans) >= 2:
            bonus += 0.06
        return bonus

    def _generic_query_penalty(self, query: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9_.-]+", query.lower())
        if not tokens:
            return 0.4
        generic_hits = sum(1 for token in tokens if token in self.GENERIC_QUERY_TERMS)
        meaningful = [token for token in tokens if token not in self.STOPWORDS and token not in self.GENERIC_QUERY_TERMS]
        penalty = generic_hits / max(len(tokens), 1) * 0.2
        if len(meaningful) < 2:
            penalty += 0.2
        return penalty

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
