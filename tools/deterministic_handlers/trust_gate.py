from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
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
    output_type: str = ""
    semantic_role: str = ""
    supporting_inputs: list[str] = field(default_factory=list)

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
        required_handler_role = str(handler_plan.get("required_handler_role") or "").strip()
        plan_status = str(handler_plan.get("status") or "").strip()
        reasons: list[str] = []
        warnings: list[str] = []

        if planned_name and planned_name != result.handler_name:
            reasons.append("planned_handler_mismatch")
        if plan_status and plan_status != "ready":
            reasons.append("handler_plan_not_ready")
        if result.status == "missing_handler":
            reasons.append("missing_handler")
        if result.status in {"missing_inputs", "no_match"} or result.missing_inputs:
            reasons.append("missing_inputs")
        if result.status == "error" or result.error:
            reasons.append("handler_error")
        if not result.ok:
            reasons.append("handler_not_ok")
        if not str(result.answer or "").strip():
            reasons.append("empty_answer")
        output_type = str(result.output_type or "").strip() or "intermediate_value"
        semantic_role = str(result.semantic_role or "").strip()
        supporting_inputs = [
            str(item).strip()
            for item in list(result.supporting_inputs or [])
            if str(item).strip()
        ]
        if output_type != "final_answer":
            reasons.append("intermediate_value_not_final_evidence")
        if not semantic_role:
            reasons.append("missing_semantic_role")
        if not supporting_inputs:
            reasons.append("missing_supporting_inputs")

        structured = result.structured_result if isinstance(result.structured_result, dict) else {}
        planned_provenance = handler_plan.get("input_provenance")
        if isinstance(planned_provenance, dict) and planned_provenance:
            source = str(planned_provenance.get("source") or "").strip()
            parse_status = str(planned_provenance.get("parse_status") or "").strip()
            result_provenance = structured.get("input_provenance")
            if source not in {
                "attachment_reader",
                "provided_attachment_context",
                "specialized_attachment_input",
            }:
                reasons.append("invalid_attachment_provenance")
            if parse_status not in {"success", "partial"}:
                reasons.append("attachment_parse_not_ready")
            if not isinstance(result_provenance, dict) or not result_provenance:
                reasons.append("missing_result_provenance")
        handler_role = str(structured.get("handler_role") or "").strip()
        if required_handler_role and handler_role and handler_role != required_handler_role:
            reasons.append("handler_role_mismatch")
        inferred_required_role = self._infer_required_handler_role(question)
        if inferred_required_role and handler_role and handler_role != inferred_required_role:
            reasons.append("answer_role_binding_failed")

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
            output_type=output_type,
            semantic_role=semantic_role,
            supporting_inputs=supporting_inputs,
        )

    def _status_from_reasons(self, reasons: list[str]) -> str:
        if "planned_handler_mismatch" in reasons:
            return "handler_mismatch"
        if any(
            reason in reasons
            for reason in {
                "handler_plan_not_ready",
                "invalid_attachment_provenance",
                "attachment_parse_not_ready",
                "missing_result_provenance",
            }
        ):
            return "input_provenance_failed"
        if "handler_role_mismatch" in reasons or "answer_role_binding_failed" in reasons:
            return "handler_role_mismatch"
        if "missing_handler" in reasons:
            return "missing_handler"
        if "missing_inputs" in reasons:
            return "missing_input"
        if "handler_error" in reasons:
            return "handler_error"
        if any(reason.startswith("candidate_") for reason in reasons):
            return "invalid_candidate"
        if "intermediate_value_not_final_evidence" in reasons:
            return "intermediate_value"
        if "missing_semantic_role" in reasons or "missing_supporting_inputs" in reasons:
            return "missing_output_contract"
        if "handler_not_ok" in reasons:
            return "handler_not_ok"
        return "untrusted"

    def _infer_required_handler_role(self, question: str) -> str:
        lowered = str(question or "").lower()
        if re.search(r"\b(odds|probability|random|randomly|maximize|expected value)\b", lowered):
            return "probability_simulation"
        if re.search(r"\b(chess|algebraic notation|checkmate|black's turn|white's turn)\b", lowered):
            return "chess_tactics"
        if "logically equivalent" in lowered or "truth table" in lowered:
            return "logic_equivalence"
        if re.search(r"\bfamily reunion\b|\badults?\b.*\bkids?\b.*\bbags?\b", lowered, flags=re.DOTALL):
            return "multi_step_counting"
        return ""


__all__ = ["HandlerTrustGate", "HandlerTrustResult"]
