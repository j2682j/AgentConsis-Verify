from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id
from utils.network_utils import normalize_text

from .search_intent_planner import SearchIntentPlan
from .semantic_impact import DEFAULT_HF_MODEL_NAME, SemanticImpactScorer, TokenSalient
from .span_repair import SalientSpan, SpanRepairer


@dataclass
class SalienceQueryCandidate:
    """
    Search query candidate produced from semantic-impact spans.

    Args:
        - query: Query text sent to the search backend.
        - matched_spans: Salient spans covered by the query.
        - coverage_score: Ratio of covered span salience.
        - score: Reserved query score field for downstream sorting.
        - semantic_impact_score: Highest matched span impact normalized to top span.
        - source: Query generation source.

    Returns:
        - SalienceQueryCandidate: Structured query candidate.
    """

    query: str
    matched_spans: list[str]
    coverage_score: float
    score: float
    semantic_impact_score: float = 0.0
    source: str = "qwen3:4b"


class MaskSalienceQueryGenerator:
    """
    Coordinate semantic-impact scoring, span repair, and model query generation.

    Args:
        - hf_model_name: HuggingFace encoder model used for deletion-impact scoring.
        - query_model_name: OpenAI-compatible provider 的 served model name。
        - llm_client: 共用的 provider-neutral LLMClient。
        - max_input_tokens: Maximum encoder input length.
        - max_salient_tokens: Maximum filtered tokens sent to span repair.
        - max_salient_spans: Maximum repaired spans sent to query generation.
        - max_query_candidates: Maximum model-generated query candidates.
        - min_token_chars: Minimum token length unless numeric content exists.
        - merge_gap_chars: Maximum gap used when repairing nearby tokens.
        - device: Optional encoder device, such as cuda or cpu.

    Returns:
        - MaskSalienceQueryGenerator: Query generation facade.
    """

    SYSTEM_PROMPT = """You are only a web search query generator.
Return JSON only.
Do not answer the question.
Do not explain.
"""

    QUERY_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }

    USER_TEMPLATE = """Question:
{question}

First search target:
{target}

Answer role:
{answer_role}

Must include:
{must_include}

Avoid:
{avoid_terms}

Preferred domain:
{preferred_domain}

Top semantic-impact spans:
{spans}

Generate {num_candidates} concise web search queries.

Rules:
- Use the first search target.
- Preserve the answer role in every query.
- Include must_include terms when possible.
- Avoid avoid_terms and answer guesses.

Return exactly this JSON shape:
{{"queries": ["...", "..."]}}"""

    def __init__(
        self,
        *,
        hf_model_name: str | None = None,
        query_model_name: str = "qwen3:4b",
        llm_client: LLMClient | None = None,
        max_input_tokens: int = 256,
        max_salient_tokens: int = 12,
        max_salient_spans: int = 5,
        max_query_candidates: int = 3,
        min_token_chars: int = 2,
        merge_gap_chars: int = 2,
        device: str | None = None,
    ) -> None:
        self.hf_model_name = hf_model_name or os.getenv("SEARCH_SALIENCE_HF_MODEL", DEFAULT_HF_MODEL_NAME)
        self.query_model_name = resolve_model_id(query_model_name)
        self.llm_client = llm_client or LLMClient()
        self.max_input_tokens = max_input_tokens
        self.max_salient_tokens = max_salient_tokens
        self.max_salient_spans = max_salient_spans
        self.max_query_candidates = max_query_candidates
        self.min_token_chars = min_token_chars
        self.merge_gap_chars = merge_gap_chars
        self.device = device
        self.semantic_scorer = SemanticImpactScorer(
            hf_model_name=self.hf_model_name,
            max_input_tokens=max_input_tokens,
            max_salient_tokens=max_salient_tokens,
            min_token_chars=min_token_chars,
            device=device,
        )
        self.span_repairer = SpanRepairer(
            max_salient_spans=max_salient_spans,
            merge_gap_chars=merge_gap_chars,
            min_token_chars=min_token_chars,
        )
        self.last_token_salience: list[TokenSalient] = []
        self.last_salient_spans: list[SalientSpan] = []

    def generate(
        self,
        question: str,
        *,
        num_candidates: int = 5,
        intent_plan: SearchIntentPlan | None = None,
    ) -> list[SalienceQueryCandidate]:
        """
        Generate search query candidates from a question.

        Args:
            - question: Original task question.
            - num_candidates: Maximum number of query candidates to return.

        Returns:
            - list[SalienceQueryCandidate]: Deduplicated query candidates.
        """
        text = normalize_text(question)
        if not text:
            return []

        token_salience = self.score_tokens(text)
        kept_tokens = self.filter_tokens(token_salience)
        spans = self.select_top_spans(self.merge_salient_tokens(text, kept_tokens))
        raw_queries = self.generate_queries_with_model(
            text,
            spans,
            num_candidates=num_candidates,
            intent_plan=intent_plan,
        )
        candidates = self.build_candidates(raw_queries, spans)

        if not candidates:
            candidates = self._fallback_candidates(text, spans)

        self.last_token_salience = token_salience
        self.last_salient_spans = spans
        return candidates[: max(1, num_candidates)]

    def score_tokens(self, question: str) -> list[TokenSalient]:
        """
        Score token deletion impact with the encoder embedding model.

        Args:
            - question: Original task question.

        Returns:
            - list[TokenSalient]: Token-level semantic impact results.
        """
        return self.semantic_scorer.score_tokens(question)

    def filter_tokens(self, tokens: list[TokenSalient]) -> list[TokenSalient]:
        """
        Keep high-impact searchable tokens.

        Args:
            - tokens: Token-level semantic impact results.

        Returns:
            - list[TokenSalient]: Filtered tokens for span repair.
        """
        return self.semantic_scorer.filter_tokens(tokens)

    def merge_salient_tokens(
        self,
        question: str,
        tokens: list[TokenSalient],
    ) -> list[SalientSpan]:
        """
        Repair filtered tokens into readable spans.

        Args:
            - question: Original task question.
            - tokens: Filtered salient tokens.

        Returns:
            - list[SalientSpan]: Repaired salient spans.
        """
        return self.span_repairer.merge_salient_tokens(question, tokens)

    def select_top_spans(self, spans: list[SalientSpan]) -> list[SalientSpan]:
        """
        Select final salient spans for query generation.

        Args:
            - spans: Repaired salient spans.

        Returns:
            - list[SalientSpan]: Top salient spans.
        """
        return self.span_repairer.select_top_spans(spans)

    def generate_queries_with_model(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        num_candidates: int,
        intent_plan: SearchIntentPlan | None = None,
    ) -> list[str]:
        """
        使用目前設定的 OpenAI-compatible model 產生精簡搜尋 query。

        Args:
            - question: Original task question.
            - spans: Top semantic-impact spans.
            - num_candidates: Maximum number of raw queries.

        Returns:
            - list[str]: Raw query strings.
        """
        if not spans:
            return [question]

        messages = self._build_query_messages(
            question,
            spans,
            num_candidates=min(max(1, num_candidates), self.max_query_candidates),
            intent_plan=intent_plan,
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
        Attach salient-span coverage metadata to raw query strings.

        Args:
            - queries: Raw query strings.
            - spans: Top semantic-impact spans.

        Returns:
            - list[SalienceQueryCandidate]: Structured query candidates.
        """
        deduped = self._dedupe_queries(queries)
        candidates: list[SalienceQueryCandidate] = []
        total_score = sum(max(span.score, 0.0) for span in spans) or 1.0
        max_span_score = max((span.score for span in spans), default=1.0) or 1.0
        for query in deduped:
            matched = self._matched_spans(query, spans)
            matched_impact = sum(span.score for span in matched)
            coverage_score = matched_impact / total_score
            semantic_impact_score = max((span.score for span in matched), default=0.0) / max_span_score
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
        Return the latest token and span diagnostics.

        Args:
            - None.

        Returns:
            - dict[str, Any]: Debug metadata from the latest generation.
        """
        return {
            "method": "embedding_delta_salience",
            "hf_model_name": self.hf_model_name,
            "query_model_name": self.query_model_name,
            "query_provider": self.llm_client.provider,
            "query_base_url": self.llm_client.base_url,
            "salient_spans": [asdict(span) for span in self.last_salient_spans],
            "token_salience": [asdict(token) for token in self.last_token_salience],
        }

    def _build_query_messages(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        num_candidates: int,
        intent_plan: SearchIntentPlan | None = None,
    ) -> list[dict[str, str]]:
        span_lines = "\n".join(f"- {span.text}" for span in spans[: self.max_salient_spans])
        must_include = intent_plan.must_include if intent_plan else []
        avoid_terms = intent_plan.avoid_terms if intent_plan else []
        preferred_domain = intent_plan.preferred_domain if intent_plan else ""
        target = intent_plan.target if intent_plan else ""
        answer_role = intent_plan.answer_role if intent_plan else "unknown"
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    question=question,
                    target=target or "Find the best first-hop source for the question.",
                    answer_role=answer_role or "unknown",
                    must_include=self._format_terms(must_include),
                    avoid_terms=self._format_terms(avoid_terms),
                    preferred_domain=preferred_domain or "",
                    spans=span_lines,
                    num_candidates=num_candidates,
                ),
            },
        ]

    def _format_terms(self, terms: list[str]) -> str:
        cleaned = [normalize_text(term) for term in terms if normalize_text(term)]
        return "\n".join(f"- {term}" for term in cleaned) if cleaned else "-"

    def _invoke_query_model(self, messages: list[dict[str, str]]) -> str:
        if self.llm_client.provider == "ollama":
            result = self.llm_client.ollama_native_chat(
                model=self.query_model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=768,
                think=False,
                json_format=self.QUERY_JSON_SCHEMA,
            )
        else:
            result = self.llm_client.chat(
                model=self.query_model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=768,
                enable_thinking=False,
            )
        return result.content

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
            cleaned_query = self._clean_query(query)
            if cleaned_query:
                queries.append(cleaned_query)
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
