from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any

from core.llm_client import LLMClient
from tools.evidence.fact_extraction import (
    AnswerBoundFactValidator,
    DirectEvidencePromoter,
    EvidenceFact,
    FactGroundingValidator,
    GroundedAnswerValue,
    PromotionDiagnostic,
)
from utils.network_utils import normalize_text


ANSWER_SUPPORT = "ANSWER_SUPPORT"
BRIDGE = "BRIDGE"
NOISE = "NOISE"
VALID_ROLES = {ANSWER_SUPPORT, BRIDGE, NOISE}


@dataclass(frozen=True)
class CandidateSpan:
    """
    候選 span 與其附近上下文。

    Args:
     - id: 批次分類時使用的穩定識別碼。
     - text: SpanRecovery 或 Labeler 還原出的 span。
     - local_context: span 附近的短上下文。
     - source_title: span 來源標題。

    Returns:
     - CandidateSpan: Span role classifier 的單筆輸入。

    """

    id: str
    text: str
    local_context: str
    source_title: str = ""
    source_id: str = ""
    source_type: str = "web"


@dataclass(frozen=True)
class SpanRoleResult:
    """
    span 角色分類結果。

    Args:
     - id: 對應 CandidateSpan 的識別碼。
     - text: 原始候選 span。
     - role: ANSWER_SUPPORT / BRIDGE / NOISE。

    Returns:
     - SpanRoleResult: 單筆 span role 分類結果。

    """

    id: str
    text: str
    role: str
    goal_id: str = ""
    semantic_facts: list[EvidenceFact] = field(default_factory=list)
    model_role: str = ""
    grounded_answer_values: list[GroundedAnswerValue] = field(default_factory=list)
    promotion_diagnostics: list[PromotionDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanRoleBatchResult:
    """
    span 批次分類結果與診斷資訊。

    Args:
     - results: 每個候選 span 的分類結果。
     - diagnostics: 模型、候選數量、失敗原因等診斷資訊。

    Returns:
     - SpanRoleBatchResult: 一次批次分類的完整結果。

    """

    results: list[SpanRoleResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "diagnostics": dict(self.diagnostics),
        }


class SpanRoleClassifier:
    """
    使用小型 SLM 將 useful span 分成答案支撐、橋接線索與雜訊。

    Args:
     - model_name: Ollama 模型名稱。
     - llm_client: Ollama native chat client。
     - max_spans_per_call: 單次批次分類最多處理的 span 數量。
     - max_context_chars: 每個 span 的 local context 長度上限。
     - max_tokens: 模型輸出 token 上限。

    Returns:
     - SpanRoleClassifier: 批次 span role classifier。

    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        max_spans_per_call: int = 15,
        max_context_chars: int = 300,
        max_tokens: int = 768,
        grounding_validator: FactGroundingValidator | None = None,
        answer_bound_validator: AnswerBoundFactValidator | None = None,
        direct_evidence_promoter: DirectEvidencePromoter | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("SPAN_ROLE_CLASSIFIER_MODEL")
            or "qwen3:1.7b"
        )
        self.llm_client = llm_client or LLMClient(provider="ollama")
        self.max_spans_per_call = max(1, max_spans_per_call)
        self.max_context_chars = max(80, max_context_chars)
        self.max_tokens = max(64, max_tokens)
        self.grounding_validator = grounding_validator or FactGroundingValidator()
        self.answer_bound_validator = (
            answer_bound_validator or AnswerBoundFactValidator()
        )
        self.direct_evidence_promoter = (
            direct_evidence_promoter
            or DirectEvidencePromoter(
                answer_bound_validator=self.answer_bound_validator,
            )
        )

    def classify_batch(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        answer_target: str = "",
        active_goal: str = "",
        next_goal: str = "",
        relation_goals: list[dict[str, str]] | None = None,
        spans: list[CandidateSpan],
    ) -> SpanRoleBatchResult:
        """
        批次分類候選 spans。

        Args:
         - question: 原始任務問題。
         - answer_requirement: 自然語言答案需求。
         - answer_target: 答案需求綁定的目標。
         - spans: 候選 span 列表。

        Returns:
         - SpanRoleBatchResult: span role 結果與診斷資訊。

        """
        candidates = self._dedupe_candidates(spans)[: self.max_spans_per_call]
        diagnostics: dict[str, Any] = {
            "provider": "ollama_native",
            "model": self.model_name,
            "candidate_count": len(candidates),
            "max_spans_per_call": self.max_spans_per_call,
            "keep_alive": 0,
            "success": False,
        }
        if not candidates:
            diagnostics["success"] = True
            diagnostics["empty_reason"] = "no_candidate_spans"
            return SpanRoleBatchResult(diagnostics=diagnostics)

        prompt = self._prompt(
            question=question,
            answer_requirement=answer_requirement,
            answer_target=answer_target,
            active_goal=active_goal,
            next_goal=next_goal,
            relation_goals=relation_goals,
            spans=candidates,
        )
        try:
            response = self.llm_client.ollama_native_chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify evidence spans. Return JSON only. "
                            "Do not explain. Do not include reasoning."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                think=False,
                json_format=self._json_schema(
                    [candidate.id for candidate in candidates]
                ),
                keep_alive=0,
            )
            parsed = self._parse_response(response.content)
            valid_goal_ids = (
                {
                    normalize_text(str(goal.get("goal_id", "")))
                    for goal in list(relation_goals or [])
                    if normalize_text(str(goal.get("goal_id", "")))
                }
                if relation_goals
                else None
            )
            results = self._normalize_results(
                parsed,
                candidates,
                valid_goal_ids=valid_goal_ids,
                question=question,
                answer_requirement=answer_requirement,
                answer_target=answer_target,
            )
            if len(results) != len(candidates):
                diagnostics.update(
                    {
                        "success": False,
                        "error": (
                            "incomplete_span_role_response:"
                            f" expected={len(candidates)} actual={len(results)}"
                        ),
                        "raw_response": response.content[:1000],
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    }
                )
                return SpanRoleBatchResult(diagnostics=diagnostics)
            diagnostics.update(
                {
                    "success": True,
                    "raw_response": response.content[:1000],
                    "answer_support_count": sum(
                        1 for result in results if result.role == ANSWER_SUPPORT
                    ),
                    "bridge_count": sum(
                        1 for result in results if result.role == BRIDGE
                    ),
                    "noise_count": sum(
                        1 for result in results if result.role == NOISE
                    ),
                    "semantic_fact_count": sum(
                        len(result.semantic_facts) for result in results
                    ),
                    "grounded_fact_count": sum(
                        fact.grounding_status == "grounded"
                        for result in results
                        for fact in result.semantic_facts
                    ),
                    "direct_bound_fact_count": sum(
                        fact.qualifiers.get("answer_binding") == "direct"
                        for result in results
                        for fact in result.semantic_facts
                    ),
                    "demoted_answer_fact_count": sum(
                        fact.qualifiers.get("original_role") == ANSWER_SUPPORT
                        for result in results
                        for fact in result.semantic_facts
                    ),
                    "promotion_attempted_count": sum(
                        bool(result.promotion_diagnostics) for result in results
                    ),
                    "promotion_accepted_count": sum(
                        len(result.grounded_answer_values) for result in results
                    ),
                    "promotion_rejected_count": sum(
                        not item.accepted
                        for result in results
                        for item in result.promotion_diagnostics
                    ),
                    "goal_assignment_counts": self._goal_assignment_counts(results),
                    "invalid_goal_assignment_count": self._invalid_goal_assignment_count(
                        parsed,
                        valid_goal_ids=valid_goal_ids,
                    ),
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )
            return SpanRoleBatchResult(results=results, diagnostics=diagnostics)
        except Exception as exc:
            diagnostics.update(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return SpanRoleBatchResult(diagnostics=diagnostics)

    def _prompt(
        self,
        *,
        question: str,
        answer_requirement: str,
        answer_target: str,
        active_goal: str,
        next_goal: str,
        relation_goals: list[dict[str, str]] | None = None,
        spans: list[CandidateSpan],
    ) -> str:
        span_lines: list[str] = []
        for span in spans:
            context = self._truncate(span.local_context, self.max_context_chars)
            title = self._truncate(span.source_title, 90)
            span_lines.extend(
                [
                    f"{span.id}.",
                    f"Span: {span.text}",
                    f"Source Title: {title}",
                    f"Context: {context}",
                ]
            )
        goal_lines = self._goal_lines(
            relation_goals=relation_goals,
            active_goal=active_goal,
            next_goal=next_goal,
        )
        goal_rule = (
            "For ANSWER_SUPPORT or BRIDGE, goal_id must name the one goal supported by the span."
            if relation_goals
            else "No relation goals are defined; use an empty goal_id for every span."
        )
        return "\n".join(
            [
                "Classify each candidate span for solving the question.",
                f"You must return exactly {len(spans)} JSON objects, one for each candidate id.",
                "Do not skip any candidate id.",
                "",
                "Labels:",
                "ANSWER_SUPPORT = directly supports the original question's final answer.",
                "BRIDGE = fills the active goal and is needed by the next goal.",
                "NOISE = irrelevant, page chrome, navigation, login, captcha, or generic text.",
                "For ANSWER_SUPPORT and BRIDGE, extract explicit subject-relation-object facts.",
                "Copy one or two exact evidence_spans from Context; never paraphrase them.",
                "ANSWER_SUPPORT means the fact object itself is a final answer value for Answer Requirement.",
                "A clue, entity, row, date, or intermediate value is BRIDGE, even when it is relevant.",
                "For count, maximum, minimum, list, or calculated questions, an individual item is not ANSWER_SUPPORT unless Context explicitly states the requested aggregate result.",
                "When Context states an explicit relation, ANSWER_SUPPORT and BRIDGE must contain at least one fact.",
                "Use an empty facts array when no explicit relation can be grounded.",
                goal_rule,
                "For NOISE, goal_id must be an empty string.",
                "",
                f"Question: {normalize_text(question)}",
                f"Answer Requirement: {normalize_text(answer_requirement) or 'Not specified'}",
                f"Answer Target: {normalize_text(answer_target) or 'Not specified'}",
                "Relation Goals:",
                *goal_lines,
                "",
                "Candidate Spans:",
                *span_lines,
                "",
                f"Return JSON only as an array with exactly {len(spans)} objects.",
            ]
        )

    def _json_schema(self, candidate_ids: list[str] | None = None) -> dict[str, Any]:
        id_schema: dict[str, Any] = {"type": "string"}
        if candidate_ids:
            id_schema["enum"] = list(candidate_ids)
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": id_schema,
                    "role": {
                        "type": "string",
                        "enum": [ANSWER_SUPPORT, BRIDGE, NOISE],
                    },
                    "goal_id": {"type": "string"},
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "relation": {"type": "string"},
                                "object": {"type": "string"},
                                "qualifiers": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                },
                                "polarity": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                },
                                "evidence_spans": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 2,
                                },
                            },
                            "required": [
                                "subject",
                                "relation",
                                "object",
                                "qualifiers",
                                "polarity",
                                "evidence_spans",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "role", "goal_id", "facts"],
                "additionalProperties": False,
            },
        }

    def _parse_response(self, content: str) -> Any:
        text = normalize_text(content)
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        recovered: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in self._balanced_json_objects(text):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or "id" not in value or "role" not in value:
                continue
            span_id = normalize_text(str(value.get("id") or ""))
            if not span_id or span_id in seen_ids:
                continue
            recovered.append(value)
            seen_ids.add(span_id)
        return recovered

    @staticmethod
    def _balanced_json_objects(text: str) -> list[str]:
        objects: list[str] = []
        starts: list[int] = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                starts.append(index)
            elif char == "}" and starts:
                start = starts.pop()
                objects.append(text[start : index + 1])
        return objects

    def _normalize_results(
        self,
        parsed: Any,
        candidates: list[CandidateSpan],
        *,
        valid_goal_ids: set[str] | None = None,
        question: str = "",
        answer_requirement: str = "",
        answer_target: str = "",
    ) -> list[SpanRoleResult]:
        if isinstance(parsed, dict):
            if "id" in parsed and "role" in parsed:
                parsed_items = [parsed]
            else:
                parsed_items = parsed.get("results") or parsed.get("spans") or []
        elif isinstance(parsed, list):
            parsed_items = parsed
        else:
            parsed_items = []

        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        results: list[SpanRoleResult] = []
        seen: set[str] = set()
        for item in parsed_items:
            if not isinstance(item, dict):
                continue
            span_id = normalize_text(str(item.get("id", "") or ""))
            if span_id not in candidate_by_id and len(candidates) == 1 and not seen:
                span_id = candidates[0].id
            if span_id not in candidate_by_id or span_id in seen:
                continue
            role = normalize_text(str(item.get("role", "") or "")).upper()
            if role not in VALID_ROLES:
                role = NOISE
            model_role = role
            goal_id = normalize_text(str(item.get("goal_id", "") or ""))
            if role == NOISE:
                goal_id = ""
            elif valid_goal_ids is not None and goal_id not in valid_goal_ids:
                role = NOISE
                goal_id = ""
            elif valid_goal_ids is None:
                goal_id = ""
            candidate = candidate_by_id[span_id]
            semantic_facts = self._semantic_facts(
                item=item,
                candidate=candidate,
                role=role,
                goal_id=goal_id,
                question=question,
                answer_requirement=answer_requirement,
                answer_target=answer_target,
            )
            grounded_answer_values: list[GroundedAnswerValue] = []
            promotion_diagnostics: list[PromotionDiagnostic] = []
            if role == ANSWER_SUPPORT and not any(
                self.answer_bound_validator.is_direct(fact)
                for fact in semantic_facts
            ):
                promotion = self.direct_evidence_promoter.promote(
                    model_role=model_role,
                    candidate_span=candidate.text,
                    context=candidate.local_context,
                    question=question,
                    answer_requirement=answer_requirement,
                    answer_target=answer_target,
                    source_id=candidate.source_id or f"span:{candidate.id}",
                    source_title=candidate.source_title,
                    document_id=candidate.source_id,
                    goal_id=goal_id,
                    semantic_facts=semantic_facts,
                )
                grounded_answer_values = list(promotion.promoted_values)
                promotion_diagnostics = list(promotion.diagnostics)
                if grounded_answer_values:
                    semantic_facts = [*semantic_facts, *promotion.promoted_facts]
                else:
                    role = BRIDGE if semantic_facts else NOISE
                    if role == NOISE:
                        goal_id = ""
            results.append(
                SpanRoleResult(
                    id=span_id,
                    text=candidate.text,
                    role=role,
                    goal_id=goal_id,
                    semantic_facts=semantic_facts,
                    model_role=model_role,
                    grounded_answer_values=grounded_answer_values,
                    promotion_diagnostics=promotion_diagnostics,
                )
            )
            seen.add(span_id)
        return results

    def _semantic_facts(
        self,
        *,
        item: dict[str, Any],
        candidate: CandidateSpan,
        role: str,
        goal_id: str,
        question: str = "",
        answer_requirement: str = "",
        answer_target: str = "",
    ) -> list[EvidenceFact]:
        if role == NOISE:
            return []
        source_id = candidate.source_id or f"span:{candidate.id}"
        facts: list[EvidenceFact] = []
        for index, raw_fact in enumerate(list(item.get("facts") or []), start=1):
            if not isinstance(raw_fact, dict):
                continue
            fact = EvidenceFact.from_dict(
                {
                    **raw_fact,
                    "fact_id": f"{source_id}:F{index}",
                    "role": role,
                    "goal_id": goal_id,
                    "context": candidate.local_context,
                    "source_id": source_id,
                    "source_type": candidate.source_type,
                    "source_title": candidate.source_title,
                    "extraction_method": "span_role_semantic_model",
                }
            )
            grounded = self.grounding_validator.validate(
                fact,
                source_text=candidate.local_context,
            )
            facts.append(
                self.answer_bound_validator.bind(
                    grounded,
                    question=question,
                    answer_requirement=answer_requirement,
                    answer_target=answer_target,
                )
            )
        return facts

    def _goal_lines(
        self,
        *,
        relation_goals: list[dict[str, str]] | None,
        active_goal: str,
        next_goal: str,
    ) -> list[str]:
        goals = list(relation_goals or [])
        if goals:
            output: list[str] = []
            for goal in goals:
                goal_id = normalize_text(str(goal.get("goal_id", "")))
                state = normalize_text(str(goal.get("state", "pending"))).upper()
                subject = normalize_text(str(goal.get("subject", ""))) or "?"
                relation = normalize_text(str(goal.get("relation", ""))) or "?"
                target = normalize_text(str(goal.get("target", ""))) or "?"
                output.append(
                    f"{goal_id} [{state}]: {subject} -> {relation} -> {target}"
                )
            return output
        return [
            f"Active Goal: {normalize_text(active_goal) or 'Not specified'}",
            f"Next Goal: {normalize_text(next_goal) or 'None'}",
        ]

    def _goal_assignment_counts(
        self,
        results: list[SpanRoleResult],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in results:
            if not result.goal_id:
                continue
            counts[result.goal_id] = counts.get(result.goal_id, 0) + 1
        return counts

    def _invalid_goal_assignment_count(
        self,
        parsed: Any,
        *,
        valid_goal_ids: set[str] | None,
    ) -> int:
        if valid_goal_ids is None:
            return 0
        if isinstance(parsed, dict):
            items = [parsed] if "id" in parsed else parsed.get("results") or parsed.get("spans") or []
        elif isinstance(parsed, list):
            items = parsed
        else:
            return 0
        invalid = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            role = normalize_text(str(item.get("role", ""))).upper()
            goal_id = normalize_text(str(item.get("goal_id", "")))
            if role in {ANSWER_SUPPORT, BRIDGE} and goal_id not in valid_goal_ids:
                invalid += 1
        return invalid

    def _dedupe_candidates(self, spans: list[CandidateSpan]) -> list[CandidateSpan]:
        output: list[CandidateSpan] = []
        seen: set[str] = set()
        for span in spans:
            text = normalize_text(span.text)
            key = text.casefold()
            if not text or key in seen:
                continue
            output.append(
                CandidateSpan(
                    id=normalize_text(span.id),
                    text=text,
                    local_context=normalize_text(span.local_context),
                    source_title=normalize_text(span.source_title),
                    source_id=normalize_text(span.source_id),
                    source_type=normalize_text(span.source_type) or "web",
                )
            )
            seen.add(key)
        return output

    def _truncate(self, text: str, max_chars: int) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


__all__ = [
    "ANSWER_SUPPORT",
    "BRIDGE",
    "NOISE",
    "CandidateSpan",
    "SpanRoleBatchResult",
    "SpanRoleClassifier",
    "SpanRoleResult",
]
