from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any

from core.llm_client import LLMClient
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
    model_role: str = ""

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
        max_tokens: int = 512,
        max_retries: int = 1,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("SPAN_ROLE_CLASSIFIER_MODEL")
            or "qwen3:4b"
        )
        self.llm_client = llm_client or LLMClient(provider="ollama")
        self.max_spans_per_call = max(1, max_spans_per_call)
        self.max_context_chars = max(80, max_context_chars)
        self.max_tokens = max(64, max_tokens)
        self.max_retries = max(0, max_retries)

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
        keep_alive: int | str = 0,
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
            "keep_alive": keep_alive,
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
        valid_goal_ids = (
            {
                normalize_text(str(goal.get("goal_id", "")))
                for goal in list(relation_goals or [])
                if normalize_text(str(goal.get("goal_id", "")))
            }
            if relation_goals
            else None
        )

        best_results: list[SpanRoleResult] = []
        best_updates: dict[str, Any] = {}
        best_parsed: Any = []
        attempts = max(1, self.max_retries + 1)
        for attempt in range(1, attempts + 1):
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
                    keep_alive=keep_alive,
                )
                parsed = self._parse_response(response.content)
                results = self._normalize_results(
                    parsed,
                    candidates,
                    valid_goal_ids=valid_goal_ids,
                )
                complete = len(results) == len(candidates)
                updates: dict[str, Any] = {
                    "success": complete,
                    "attempt_count": attempt,
                    "raw_response": response.content[:1000],
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
                if not complete:
                    updates["error"] = (
                        "incomplete_span_role_response:"
                        f" expected={len(candidates)} actual={len(results)}"
                    )
                if len(results) > len(best_results):
                    best_results = results
                    best_updates = updates
                    best_parsed = parsed
                if complete:
                    break
            except Exception as exc:
                if not best_results:
                    best_updates = {
                        "success": False,
                        "attempt_count": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue

        if not best_results:
            diagnostics.update(best_updates)
            return SpanRoleBatchResult(diagnostics=diagnostics)

        if len(best_results) != len(candidates):
            best_updates["partial"] = True
            best_updates.setdefault(
                "error",
                (
                    "incomplete_span_role_response:"
                    f" expected={len(candidates)} actual={len(best_results)}"
                ),
            )

        diagnostics.update(
            {
                **best_updates,
                "answer_support_count": sum(
                    1 for result in best_results if result.role == ANSWER_SUPPORT
                ),
                "bridge_count": sum(
                    1 for result in best_results if result.role == BRIDGE
                ),
                "noise_count": sum(
                    1 for result in best_results if result.role == NOISE
                ),
                "goal_assignment_counts": self._goal_assignment_counts(best_results),
                "invalid_goal_assignment_count": self._invalid_goal_assignment_count(
                    best_parsed,
                    valid_goal_ids=valid_goal_ids,
                ),
            }
        )
        return SpanRoleBatchResult(results=best_results, diagnostics=diagnostics)

    def unload(self) -> dict[str, Any]:
        try:
            self.llm_client.ollama_native_chat(
                model=self.model_name,
                messages=[{"role": "user", "content": ""}],
                temperature=0,
                max_tokens=1,
                think=False,
                keep_alive=0,
            )
            return {"model": self.model_name, "unloaded": True, "warning": ""}
        except Exception as exc:
            return {
                "model": self.model_name,
                "unloaded": False,
                "warning": f"{type(exc).__name__}: {exc}",
            }

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
                "ANSWER_SUPPORT means the fact object itself is a final answer value for Answer Requirement.",
                "A clue, entity, row, date, or intermediate value is BRIDGE, even when it is relevant.",
                "For count, maximum, minimum, list, or calculated questions, an individual item is not ANSWER_SUPPORT unless Context explicitly states the requested aggregate result.",
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
                },
                "required": ["id", "role", "goal_id"],
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
            results.append(
                SpanRoleResult(
                    id=span_id,
                    text=candidate.text,
                    role=role,
                    goal_id=goal_id,
                    model_role=model_role,
                )
            )
            seen.add(span_id)
        return results

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
