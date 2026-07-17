from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text


EMPTY_REQUIREMENT_VALUES = {"", "unknown", "none", "null", "n/a"}


def _clean_contract_text(value: Any) -> str:
    text = normalize_text(str(value or ""))
    if text.casefold() in EMPTY_REQUIREMENT_VALUES:
        return ""
    return text


@dataclass(frozen=True)
class EvidenceSelectionContract:
    """
    Evidence selection 使用的簡單任務契約。

    Args:
     - question: 原始問題。
     - answer_requirement: 自然語言答案需求，例如 how many studio albums。
     - answer_target: 答案需求綁定的目標，例如 Mercedes Sosa studio albums between 2000 and 2009。
     - must_include: evidence 應盡量保留的題目限制或來源線索。

    Returns:
     - EvidenceSelectionContract: EvidenceConverter 與 compatibility gate 共用的任務契約。

    """

    question: str
    answer_requirement: str = ""
    answer_target: str = ""
    must_include: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_parts(
        cls,
        *,
        question: str,
        answer_requirement: str = "",
        answer_target: str = "",
        must_include: list[Any] | None = None,
    ) -> "EvidenceSelectionContract":
        return cls(
            question=normalize_text(question),
            answer_requirement=_clean_contract_text(answer_requirement),
            answer_target=_clean_contract_text(answer_target),
            must_include=[
                normalize_text(str(item))
                for item in list(must_include or [])
                if _clean_contract_text(item)
            ],
        )


__all__ = ["EvidenceSelectionContract"]
