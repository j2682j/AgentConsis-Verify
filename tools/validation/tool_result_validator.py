from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from score.answer_validator import AnswerValidator


@dataclass
class ToolValidationResult:
    """
    保存單次 tool output 的輕量驗證結果。

    Args:
        - valid: tool output 是否可進入下一步 evidence / state。
        - status: ok、empty_output、error、missing_input 等狀態。
        - reasons: 驗證判斷原因。
        - candidate_answer: 從工具結果中取得的答案候選。
        - evidence_text: 可進入 Agent context 的 evidence 文字。
        - missing_inputs: tool / handler 回報缺少的輸入。
        - next_action_hint: 下一步補資料建議。
        - normalized_payload: 標準化後的 tool payload。

    Returns:
        - ToolValidationResult: tool output validation 結果。
    """

    valid: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    candidate_answer: str = ""
    evidence_text: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    next_action_hint: str = ""
    normalized_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolResultValidator:
    """
    對工具執行結果做低成本 schema / empty / error 檢查。

    Args:
        - answer_validator: 用於辨識 tool-call JSON、拒答與無效候選答案。

    Returns:
        - ToolResultValidator: 可重複使用的 tool output validator。
    """

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def validate(
        self,
        *,
        tool_name: str,
        raw_result: Any = None,
        output_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ToolValidationResult:
        """
        驗證單次 tool output 是否可進入 evidence 準備流程。

        Args:
            - tool_name: 工具名稱。
            - raw_result: 工具原始輸出。
            - output_text: 已轉成文字的工具輸出。
            - metadata: tool_usage 中的額外欄位。

        Returns:
            - ToolValidationResult: 輕量驗證結果。
        """
        metadata = metadata or {}
        payload = self._normalize_payload(
            tool_name=tool_name,
            raw_result=raw_result,
            output_text=output_text,
            metadata=metadata,
        )
        reasons: list[str] = []
        missing_inputs = self._list_value(payload.get("missing_inputs"))
        error = str(payload.get("error") or payload.get("error_message") or "").strip()
        ok = bool(payload.get("ok", metadata.get("ok", bool(output_text.strip()))))
        evidence_text = str(payload.get("output_text") or output_text or "").strip()
        candidate_answer = self._candidate_answer(payload)

        if error:
            reasons.append("tool_error")
        if missing_inputs:
            reasons.append("missing_inputs")
        if not evidence_text and not candidate_answer:
            reasons.append("empty_output")
        if candidate_answer and self.answer_validator.is_tool_call_like(candidate_answer):
            reasons.append("candidate_is_tool_call")
        if evidence_text and self.answer_validator.is_tool_call_like(evidence_text):
            reasons.append("evidence_is_tool_call")

        valid = ok and not error and not missing_inputs and bool(evidence_text or candidate_answer)
        if "candidate_is_tool_call" in reasons or "evidence_is_tool_call" in reasons:
            valid = False

        status = "ok" if valid else self._status_from_reasons(reasons, ok=ok)
        return ToolValidationResult(
            valid=valid,
            status=status,
            reasons=reasons,
            candidate_answer=candidate_answer,
            evidence_text=evidence_text,
            missing_inputs=missing_inputs,
            next_action_hint=str(payload.get("next_action_hint") or payload.get("retry_hint") or "").strip(),
            normalized_payload=payload,
        )

    def _normalize_payload(
        self,
        *,
        tool_name: str,
        raw_result: Any,
        output_text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(raw_result, dict):
            payload.update(raw_result)
        payload.update(metadata)
        payload.setdefault("tool_name", tool_name)
        payload.setdefault("output_text", output_text)
        if "raw_result" not in payload:
            payload["raw_result"] = raw_result
        return payload

    def _candidate_answer(self, payload: dict[str, Any]) -> str:
        for key in ("answer", "answer_text", "candidate_answer", "final_answer"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return self.answer_validator.clean(value)
        evidence = payload.get("evidence")
        if isinstance(evidence, dict):
            for key in ("answer", "answer_text"):
                value = evidence.get(key)
                if value is not None and str(value).strip():
                    return self.answer_validator.clean(value)
        return ""

    def _list_value(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _status_from_reasons(self, reasons: list[str], *, ok: bool) -> str:
        if "missing_inputs" in reasons:
            return "missing_input"
        if "tool_error" in reasons:
            return "error"
        if "empty_output" in reasons:
            return "empty_output"
        if "candidate_is_tool_call" in reasons or "evidence_is_tool_call" in reasons:
            return "invalid_format"
        if not ok:
            return "not_ok"
        return "invalid"


__all__ = ["ToolResultValidator", "ToolValidationResult"]
