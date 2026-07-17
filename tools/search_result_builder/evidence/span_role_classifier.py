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
        max_context_chars: int = 220,
        max_tokens: int = 512,
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

    def classify_batch(
        self,
        *,
        question: str,
        answer_requirement: str = "",
        answer_target: str = "",
        active_goal: str = "",
        next_goal: str = "",
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
                json_format=self._json_schema(),
                keep_alive=0,
            )
            parsed = self._parse_response(response.content)
            results = self._normalize_results(parsed, candidates)
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
                "",
                f"Question: {normalize_text(question)}",
                f"Answer Requirement: {normalize_text(answer_requirement) or 'Not specified'}",
                f"Answer Target: {normalize_text(answer_target) or 'Not specified'}",
                f"Active Goal: {normalize_text(active_goal) or 'Not specified'}",
                f"Next Goal: {normalize_text(next_goal) or 'None'}",
                "",
                "Candidate Spans:",
                *span_lines,
                "",
                f"Return JSON only as an array with exactly {len(spans)} objects:",
                '[{"id":"1","role":"ANSWER_SUPPORT"},{"id":"2","role":"BRIDGE"},{"id":"3","role":"NOISE"}]',
            ]
        )

    def _json_schema(self) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [ANSWER_SUPPORT, BRIDGE, NOISE],
                    },
                },
                "required": ["id", "role"],
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
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if not match:
            return []
        return json.loads(match.group(1))

    def _normalize_results(
        self,
        parsed: Any,
        candidates: list[CandidateSpan],
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
            if span_id not in candidate_by_id or span_id in seen:
                continue
            role = normalize_text(str(item.get("role", "") or "")).upper()
            if role not in VALID_ROLES:
                role = NOISE
            candidate = candidate_by_id[span_id]
            results.append(
                SpanRoleResult(
                    id=span_id,
                    text=candidate.text,
                    role=role,
                )
            )
            seen.add(span_id)
        return results

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
