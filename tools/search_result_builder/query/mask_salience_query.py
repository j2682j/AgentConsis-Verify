from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id
from utils.network_utils import normalize_text

from .question_role_extractor import QuestionRole, QuestionRoleExtractor
from .relation_plan import RelationPlan
from .relation_plan_validator import RelationPlanValidationResult, RelationPlanValidator
from .span_classifier import ClassifiedSpan, SpanRoleClassifier
from .semantic_impact import DEFAULT_HF_MODEL_NAME, SemanticImpactScorer, TokenSalient
from .span_repair import SalientSpan, SpanRepairer
from .source_requirement import SearchQueryRequest, SourceRequirement


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
    source_requirement: SourceRequirement = field(default_factory=SourceRequirement)


@dataclass(frozen=True)
class QueryGenerationOutput:
    """
    保存單次模型呼叫產生的 query requests 與 relation plan。

    Args:
     - query_requests: 帶有來源需求的查詢候選。
     - relation_plan: 依序執行的自然語言 relation goals。

    Returns:
     - QueryGenerationOutput: Query Generator 的結構化模型輸出。
    """

    query_requests: list[SearchQueryRequest] = field(default_factory=list)
    relation_plan: RelationPlan = field(default_factory=RelationPlan)

    def __iter__(self):
        return iter(self.query_requests)

    def __len__(self) -> int:
        return len(self.query_requests)

    def __getitem__(self, index: int) -> SearchQueryRequest:
        return self.query_requests[index]


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

    SYSTEM_PROMPT = """You generate retrieval queries and select the required source.
Return JSON only.
Do not answer the question.
Do not explain.

Source types:
- video: video frames, transcript, audio, or a supplied video URL.
- academic: papers, authors, citations, or publication metadata.
- collection: tables, catalogs, archives, or repeated records.
- web: other ordinary web pages.

Use direct_fetch when the question supplies the exact URL. Preserve URLs exactly.
"""

    QUERY_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source_kind": {
                            "type": "string",
                            "enum": ["web", "video", "academic", "collection"],
                        },
                        "access_mode": {
                            "type": "string",
                            "enum": ["search", "direct_fetch", "browser"],
                        },
                        "source_hint": {"type": "string"},
                    },
                    "required": [
                        "query",
                        "source_kind",
                        "access_mode",
                        "source_hint",
                    ],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 3,
            },
            "relation_goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "target": {"type": "string"},
                        "source_kind": {
                            "type": "string",
                            "enum": ["web", "video", "academic", "collection"],
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                        },
                        "verification_scope": {
                            "type": "string",
                            "enum": ["passage", "full_document", "collection"],
                        },
                    },
                    "required": [
                        "subject",
                        "relation",
                        "target",
                        "source_kind",
                        "polarity",
                        "verification_scope",
                    ],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["queries", "relation_goals"],
        "additionalProperties": False,
    }

    USER_TEMPLATE = """Question:
{question}

Search clues:
{search_clues}

Constraints:
{constraints}

Answer target:
{answer_target}

Avoid in queries:
{avoid_in_queries}

Other important spans:
{other_spans}

Generate up to {num_candidates} concise retrieval queries and select a source for each query.

Rules:
- Preserve names, dates, titles, URLs, source names, and answer constraints.
- Use browser only for rendered or interactive collection pages.
- Do not treat an answer format or placeholder as a known search fact.
- Leave source_hint empty unless the question names or supplies a source.
- Split the information need into at most 3 ordered relation goals.
- A later goal that depends on the previous result must use an empty subject.
- For "Who RELATION TARGET?", TARGET is the relation subject and person is the goal target.
- Never use Who, What, Which, When, or Where as a relation subject or target entity.
- If the target entity must first be identified from constraints, create an identification goal first.
- Keep event roles distinct: nominated by is not promoted by, reviewed by, illustrated by, or supported by.
- Use negative only when the task requires proving that an explicit term is absent.
- Negative goals must put that explicit term in target.
- Use full_document when absence cannot be verified from a passage.
- Use collection when every relevant record in a list must be checked.
- Do not guess or include a final answer.

Return exactly this JSON shape:
{{"queries": [{{"query": "...", "source_kind": "web|video|academic|collection", "access_mode": "search|direct_fetch|browser", "source_hint": "..."}}], "relation_goals": [{{"subject": "...", "relation": "...", "target": "...", "source_kind": "web|video|academic|collection", "polarity": "positive|negative", "verification_scope": "passage|full_document|collection"}}]}}"""

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
        self.question_role_extractor = QuestionRoleExtractor(scorer=self.semantic_scorer)
        self.relation_plan_validator = RelationPlanValidator()
        self.span_classifier = SpanRoleClassifier(scorer=self.semantic_scorer)
        self.last_token_salience: list[TokenSalient] = []
        self.last_salient_spans: list[SalientSpan] = []
        self.last_classified_spans: list[ClassifiedSpan] = []
        self.last_question_role: QuestionRole = QuestionRole()
        self.last_relation_plan: RelationPlan = RelationPlan()
        self.last_relation_plan_validation = RelationPlanValidationResult(
            valid=True,
            plan=RelationPlan(),
        )

    def generate(
        self,
        question: str,
        *,
        num_candidates: int = 5,
        intent_plan: Any | None = None,
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
        spans = self.span_repairer.build_spans(
            text,
            kept_tokens,
            scorer=self.semantic_scorer,
        )
        question_role = self.question_role_extractor.extract(text)
        classified_spans = self.classify_spans(text, spans, question_role=question_role)
        generation_output = self.generate_queries_with_model(
            text,
            spans,
            classified_spans=classified_spans,
            num_candidates=num_candidates,
        )
        relation_validation = self.relation_plan_validator.validate(
            generation_output.relation_plan,
            question_role=question_role,
        )
        generation_output = QueryGenerationOutput(
            query_requests=generation_output.query_requests,
            relation_plan=relation_validation.plan,
        )
        candidates = self.build_candidates(generation_output.query_requests, spans)

        if not candidates:
            candidates = self._fallback_candidates(text, spans)

        self.last_token_salience = token_salience
        self.last_salient_spans = spans
        self.last_classified_spans = classified_spans
        self.last_question_role = question_role
        self.last_relation_plan = generation_output.relation_plan
        self.last_relation_plan_validation = relation_validation
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

    def classify_spans(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        question_role: QuestionRole | None = None,
    ) -> list[ClassifiedSpan]:
        """
        Classify repaired spans into query-generation roles.

        Args:
            - question: Original task question.
            - spans: Repaired salient spans.

        Returns:
            - list[ClassifiedSpan]: Role-labeled spans for prompt sections.
        """
        question_role = question_role or self.question_role_extractor.extract(question)
        return self.span_classifier.classify(question, spans, question_role=question_role)

    def generate_queries_with_model(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        classified_spans: list[ClassifiedSpan] | None = None,
        num_candidates: int,
        intent_plan: Any | None = None,
    ) -> QueryGenerationOutput:
        """
        使用目前設定的 OpenAI-compatible model 產生精簡搜尋 query。

        Args:
            - question: Original task question.
            - spans: Top semantic-impact spans.
            - num_candidates: Maximum number of raw queries.

        Returns:
            - QueryGenerationOutput: Query requests and ordered relation goals.
        """
        if not spans:
            return QueryGenerationOutput(
                query_requests=[SearchQueryRequest.fallback(question)]
            )

        messages = self._build_query_messages(
            question,
            spans,
            num_candidates=min(max(1, num_candidates), self.max_query_candidates),
            classified_spans=classified_spans or [],
        )
        try:
            raw_reply = self._invoke_query_model(messages)
        except Exception:
            return QueryGenerationOutput(
                query_requests=self._fallback_query_requests(question, spans)
            )
        parsed = self._parse_query_json(
            raw_reply,
            question=question,
        )
        if parsed.query_requests:
            return parsed
        return QueryGenerationOutput(
            query_requests=self._fallback_query_requests(question, spans)
        )

    def build_candidates(
        self,
        requests: list[SearchQueryRequest],
        spans: list[SalientSpan],
    ) -> list[SalienceQueryCandidate]:
        """
        Attach salient-span coverage metadata to raw query strings.

        Args:
            - requests: Query strings bound to source requirements.
            - spans: Top semantic-impact spans.

        Returns:
            - list[SalienceQueryCandidate]: Structured query candidates.
        """
        deduped = self._dedupe_query_requests(requests)
        candidates: list[SalienceQueryCandidate] = []
        total_score = sum(max(span.score, 0.0) for span in spans) or 1.0
        max_span_score = max((span.score for span in spans), default=1.0) or 1.0
        for request in deduped:
            query = request.query
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
                    source_requirement=request.source_requirement,
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
            "question_role": self.last_question_role.to_dict(),
            "relation_plan": self.last_relation_plan.to_dict(),
            "salient_spans": [asdict(span) for span in self.last_salient_spans],
            "classified_spans": [span.to_dict() for span in self.last_classified_spans],
            "token_salience": [asdict(token) for token in self.last_token_salience],
        }

    def _build_query_messages(
        self,
        question: str,
        spans: list[SalientSpan],
        *,
        num_candidates: int,
        classified_spans: list[ClassifiedSpan] | None = None,
    ) -> list[dict[str, str]]:
        sections = self._classified_prompt_sections(classified_spans or [])
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    question=question,
                    search_clues=self._format_terms(sections["search_clues"]),
                    constraints=self._format_terms(sections["constraints"]),
                    answer_target=self._format_terms(sections["answer_target"]),
                    avoid_in_queries=self._format_terms(sections["avoid_in_queries"]),
                    other_spans=self._format_terms(sections["other_spans"]),
                    num_candidates=num_candidates,
                ),
            },
        ]

    def _classified_prompt_sections(
        self,
        classified_spans: list[ClassifiedSpan],
    ) -> dict[str, list[str]]:
        grouped = self.span_classifier.grouped(classified_spans)
        sections = {
            "search_clues": [span.text for span in grouped.get("source_clue", [])],
            "constraints": [span.text for span in grouped.get("constraint", [])],
            "answer_target": [span.text for span in grouped.get("answer_target", [])],
            "avoid_in_queries": [
                span.text
                for span in [
                    *grouped.get("format_instruction", []),
                    *grouped.get("weak_generic", []),
                ]
            ],
            "other_spans": [span.text for span in grouped.get("other", [])],
        }
        if not sections["search_clues"]:
            fallback = [
                span.text
                for span in sorted(
                    classified_spans,
                    key=lambda item: (item.score, item.confidence),
                    reverse=True,
                )[:3]
            ]
            sections["search_clues"] = fallback
        return sections

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
                keep_alive=0,
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

    def _parse_query_json(
        self,
        raw_reply: str,
        *,
        question: str = "",
    ) -> QueryGenerationOutput:
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

        requests: list[SearchQueryRequest] = []
        for item in raw_queries:
            if isinstance(item, str):
                request = SearchQueryRequest.fallback(item)
            elif isinstance(item, dict):
                request = SearchQueryRequest.from_dict(item)
            else:
                continue
            if request is None:
                continue
            request = self._repair_explicit_source_request(
                request,
                question=question,
            )
            cleaned_query = self._clean_query(request.query)
            if cleaned_query:
                requests.append(
                    SearchQueryRequest(
                        query=cleaned_query,
                        source_requirement=request.source_requirement,
                    )
                )
        relation_specs = (
            list(parsed.get("relation_goals") or [])
            if isinstance(parsed, dict)
            else []
        )
        return QueryGenerationOutput(
            query_requests=self._dedupe_query_requests(requests),
            relation_plan=RelationPlan.from_specs(relation_specs),
        )

    def _repair_explicit_source_request(
        self,
        request: SearchQueryRequest,
        *,
        question: str,
    ) -> SearchQueryRequest:
        video_url = self._explicit_video_url(question)
        if not video_url:
            return request
        query = request.query
        if re.search(r"https?://", query, flags=re.IGNORECASE):
            query = re.sub(
                r"https?://[^\s)>\]\"']+",
                video_url,
                query,
                count=1,
                flags=re.IGNORECASE,
            )
        elif video_url not in query:
            query = f"{query} {video_url}"
        return SearchQueryRequest(
            query=normalize_text(query),
            source_requirement=SourceRequirement(
                source_kind="video",
                access_mode="direct_fetch",
                source_hint=video_url,
            ),
        )

    def _explicit_video_url(self, text: str) -> str:
        match = re.search(
            r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[^\s)>\]\"']+",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0).rstrip(".,)")
        match = re.search(
            r"https?://[^\s)>\]\"']+\.(?:mp4|mov|mkv|webm)(?:\?[^\s)>\]]*)?",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        return match.group(0).rstrip(".,)") if match else ""

    def _fallback_raw_queries(self, question: str, spans: list[SalientSpan]) -> list[str]:
        span_texts = [span.text for span in spans if span.text]
        queries: list[str] = []
        if span_texts:
            queries.append(" ".join(span_texts[:4]))
            for span in span_texts[: self.max_query_candidates]:
                queries.append(span)
        queries.append(question)
        return self._dedupe_queries(queries)

    def _fallback_query_requests(
        self,
        question: str,
        spans: list[SalientSpan],
    ) -> list[SearchQueryRequest]:
        return [
            SearchQueryRequest.fallback(query)
            for query in self._fallback_raw_queries(question, spans)
        ]

    def _fallback_candidates(
        self,
        question: str,
        spans: list[SalientSpan],
    ) -> list[SalienceQueryCandidate]:
        return self.build_candidates(self._fallback_query_requests(question, spans), spans)

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

    def _dedupe_query_requests(
        self,
        requests: list[SearchQueryRequest],
    ) -> list[SearchQueryRequest]:
        result: list[SearchQueryRequest] = []
        seen: set[str] = set()
        for request in requests:
            cleaned = self._clean_query(request.query)
            key = self._normalize_for_match(cleaned)
            if not cleaned or not key or key in seen:
                continue
            result.append(
                SearchQueryRequest(
                    query=cleaned,
                    source_requirement=request.source_requirement,
                )
            )
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
    "QueryGenerationOutput",
    "SalienceQueryCandidate",
    "SearchQueryRequest",
    "SourceRequirement",
    "SalientSpan",
    "TokenSalient",
]
