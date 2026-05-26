from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DeterministicReadiness:
    """
    描述 deterministic solver 是否適合處理目前問題。

    Args:
        - is_deterministic_task: 問題是否屬於 deterministic 類型。
        - is_closed_world: 問題是否可只靠已知輸入資料解答。
        - has_complete_data: 是否具備完整輸入資料。
        - evidence_source: 判斷依據來源。
        - reason: readiness 判斷原因。

    Returns:
        - DeterministicReadiness: solver readiness metadata。
    """

    is_deterministic_task: bool = False
    is_closed_world: bool = False
    has_complete_data: bool = False
    evidence_source: str = "none"
    reason: str = ""


@dataclass
class DeterministicSolverResult:
    """
    保存 deterministic solver 的解題結果與證據資訊。

    Args:
        - used_deterministic_solver: 是否成功使用 deterministic solver。
        - task_type: solver 判定或處理的任務類型。
        - answer: 原始答案值。
        - answer_text: 可放入 prompt 的答案文字。
        - confidence: solver 對答案的信心分數。
        - evidence: 解題使用的結構化證據。
        - evidence_source: 證據來源。
        - readiness: solver readiness metadata。
        - error: 失敗或未使用原因。

    Returns:
        - DeterministicSolverResult: solver 結果物件。
    """

    used_deterministic_solver: bool
    task_type: str
    answer: Any = None
    answer_text: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    evidence_source: str = "none"
    readiness: DeterministicReadiness = field(default_factory=DeterministicReadiness)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        將 solver 結果轉成可序列化 dict。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: dataclass 轉換後的 dict。
        """
        return asdict(self)

    @classmethod
    def miss(cls, task_type: str = "unsupported", error: str | None = None) -> "DeterministicSolverResult":
        """
        建立未命中或不適用 deterministic solver 的結果。

        Args:
            - task_type: 未命中的任務類型。
            - error: 未命中或失敗原因。

        Returns:
            - DeterministicSolverResult: used_deterministic_solver=False 的結果。
        """
        return cls(
            used_deterministic_solver=False,
            task_type=task_type,
            confidence=0.0,
            error=error,
            readiness=DeterministicReadiness(
                is_deterministic_task=task_type not in {"unsupported", "not_deterministic"},
                is_closed_world=False,
                has_complete_data=False,
                evidence_source="none",
                reason=error or "no deterministic handler matched",
            ),
        )
