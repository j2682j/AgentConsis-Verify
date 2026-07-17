from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from score.answer_validator import AnswerValidator


@dataclass
class CandidateResultValidation:
    """
    保存工具或 handler 產生的答案候選驗證結果。

    Args:
        - valid: 候選答案是否可進 candidate pool。
        - status: ok、empty、refusal、too_verbose、invalid_format 等狀態。
        - reasons: 驗證原因。
        - cleaned_answer: 清理後的候選答案。
        - evidence_bound: 是否有 evidence 或來源綁定。

    Returns:
        - CandidateResultValidation: candidate answer validation 結果。
    """

    valid: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    cleaned_answer: str = ""
    evidence_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateResultValidator:
    """
    驗證工具輸出的明確答案候選是否適合進入候選池。

    Args:
        - answer_validator: 既有 AnswerValidator。

    Returns:
        - CandidateResultValidator: 可重複使用的 candidate validator。
    """

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def validate(
        self,
        answer: Any,
        *,
        question: str = "",
        evidence_text: str = "",
        source_binding: dict[str, Any] | None = None,
    ) -> CandidateResultValidation:
        """
        驗證候選答案格式與 evidence binding。

        Args:
            - answer: 候選答案。
            - question: 原始問題。
            - evidence_text: 支撐候選答案的 evidence。
            - source_binding: row、source、handler input 等來源資訊。

        Returns:
            - CandidateResultValidation: candidate validation 結果。
        """
        cleaned = self.answer_validator.clean(answer)
        reasons: list[str] = []
        source_binding = source_binding or {}
        evidence_bound = bool(str(evidence_text or "").strip() or source_binding)

        if not cleaned:
            reasons.append("empty_answer")
        if cleaned and self.answer_validator.is_tool_call_like(cleaned):
            reasons.append("tool_call_like")
        if cleaned and self.answer_validator.is_refusal_like(cleaned):
            if not self.answer_validator.question_allow_refusal(question):
                reasons.append("refusal_like")
        if (
            cleaned
            and self.answer_validator.is_too_verbose(cleaned)
            and not self._is_requested_compact_list(cleaned, question)
        ):
            reasons.append("too_verbose")
        if cleaned and not evidence_bound:
            reasons.append("missing_evidence_binding")

        valid = not reasons
        return CandidateResultValidation(
            valid=valid,
            status="ok" if valid else self._status_from_reasons(reasons),
            reasons=reasons,
            cleaned_answer=cleaned,
            evidence_bound=evidence_bound,
        )

    @staticmethod
    def _is_requested_compact_list(answer: str, question: str) -> bool:
        request = str(question or "").lower()
        if not re.search(r"\b(comma[- ]separated|list of|as a list|order the list)\b", request):
            return False
        if "\n" in answer or "," not in answer:
            return False
        items = [item.strip() for item in answer.split(",")]
        return 2 <= len(items) <= 100 and all(item and len(item) <= 80 for item in items)

    def _status_from_reasons(self, reasons: list[str]) -> str:
        if "empty_answer" in reasons:
            return "empty"
        if "tool_call_like" in reasons:
            return "invalid_format"
        if "refusal_like" in reasons:
            return "refusal"
        if "too_verbose" in reasons:
            return "too_verbose"
        if "missing_evidence_binding" in reasons:
            return "missing_evidence_binding"
        return "invalid"


__all__ = ["CandidateResultValidation", "CandidateResultValidator"]
