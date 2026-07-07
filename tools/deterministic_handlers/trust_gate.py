from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tools.validation import CandidateResultValidator

from .base import HandlerResult


@dataclass
class HandlerTrustResult:
    """
    保存 deterministic handler 執行後的 trust gate 判斷。

    Args:
        - trusted: handler 結果是否可進 solver evidence。
        - status: trusted、missing_input、handler_error、invalid_candidate 等狀態。
        - reasons: trust gate 判斷原因。
        - answer: 通過清理的 handler answer。
        - evidence_text: 可交給 Agent 的 handler evidence。
        - missing_inputs: handler 回報缺少的輸入。
        - next_action_hint: 下一步補資料建議。
        - confidence: handler 信心分數。
        - candidate_validation: answer candidate validation 結果。

    Returns:
        - HandlerTrustResult: handler trust gate 結果。
    """

    trusted: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    answer: str = ""
    evidence_text: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    next_action_hint: str = ""
    confidence: float = 0.0
    candidate_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HandlerTrustGate:
    """
    驗證 deterministic handler result 是否可信並可作為 evidence。

    Args:
        - candidate_validator: 用於檢查 handler answer candidate。

    Returns:
        - HandlerTrustGate: 可重複使用的 handler trust gate。
    """

    def __init__(self, candidate_validator: CandidateResultValidator | None = None) -> None:
        self.candidate_validator = candidate_validator or CandidateResultValidator()

    def validate(
        self,
        result: HandlerResult,
        *,
        question: str,
        handler_plan: dict[str, Any] | None = None,
    ) -> HandlerTrustResult:
        """
        驗證 handler result 是否可進入 solver evidence。

        Args:
            - result: DeterministicHandlerRouter 回傳結果。
            - question: 原始問題。
            - handler_plan: Planner 產生的 handler plan。

        Returns:
            - HandlerTrustResult: trust gate 判斷結果。
        """
        handler_plan = handler_plan or {}
        planned_name = str(handler_plan.get("handler_name") or "").strip()
        reasons: list[str] = []
        warnings: list[str] = []

        if planned_name and planned_name != result.handler_name:
            reasons.append("planned_handler_mismatch")
        if result.status in {"missing_inputs", "no_match"} or result.missing_inputs:
            reasons.append("missing_inputs")
        if result.status == "error" or result.error:
            reasons.append("handler_error")
        if not result.ok:
            reasons.append("handler_not_ok")
        if not str(result.answer or "").strip():
            reasons.append("empty_answer")

        source_binding = {
            "handler_name": result.handler_name,
            "input_summary": result.input_summary,
            "planned_handler_name": planned_name,
        }
        candidate_validation = self.candidate_validator.validate(
            result.answer,
            question=question,
            evidence_text=result.evidence_text,
            source_binding=source_binding,
        )
        if not candidate_validation.valid:
            reasons.append(f"candidate_{candidate_validation.status}")

        if not result.input_summary:
            warnings.append("missing_input_summary")
        structured = result.structured_result if isinstance(result.structured_result, dict) else {}
        if "output_contract" not in structured:
            warnings.append("missing_output_contract")

        hard_reasons = [reason for reason in reasons if reason]
        trusted = not hard_reasons
        status = "trusted" if trusted else self._status_from_reasons(hard_reasons)
        return HandlerTrustResult(
            trusted=trusted,
            status=status,
            reasons=hard_reasons + warnings,
            answer=candidate_validation.cleaned_answer,
            evidence_text=result.evidence_text if trusted else "",
            missing_inputs=list(result.missing_inputs or []),
            next_action_hint=result.next_action_hint,
            confidence=float(result.confidence or 0.0),
            candidate_validation=candidate_validation.to_dict(),
        )

    def _status_from_reasons(self, reasons: list[str]) -> str:
        if "planned_handler_mismatch" in reasons:
            return "handler_mismatch"
        if "missing_inputs" in reasons:
            return "missing_input"
        if "handler_error" in reasons:
            return "handler_error"
        if any(reason.startswith("candidate_") for reason in reasons):
            return "invalid_candidate"
        if "handler_not_ok" in reasons:
            return "handler_not_ok"
        return "untrusted"


__all__ = ["HandlerTrustGate", "HandlerTrustResult"]
