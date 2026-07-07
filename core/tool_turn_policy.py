from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveToolTurnPolicy:
    """
    根據工具結果的資訊增益動態調整 Stage1 工具回合。

    Args:
        - base_budget: 初始可執行工具回合數。
        - hard_limit: 有進展時最多可延長到的工具回合數。
        - no_progress_limit: 連續多少次無進展後停止。

    Returns:
        - AdaptiveToolTurnPolicy: 單次 Agent run 的工具預算狀態。
    """

    base_budget: int = 2
    hard_limit: int = 4
    no_progress_limit: int = 2
    allowed_budget: int = field(init=False)
    turns_used: int = 0
    progress_count: int = 0
    no_progress_streak: int = 0
    extension_reasons: list[str] = field(default_factory=list)
    stop_reason: str = ""
    force_final: bool = False
    _evidence_fingerprints: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.base_budget = max(0, self.base_budget)
        self.hard_limit = (
            0 if self.base_budget == 0 else max(self.base_budget, self.hard_limit)
        )
        self.no_progress_limit = max(1, self.no_progress_limit)
        self.allowed_budget = self.base_budget
        if self.base_budget == 0:
            self.force_final = True
            self.stop_reason = "tool_use_disabled"

    def can_execute(self) -> bool:
        return (
            not self.force_final
            and self.turns_used < self.allowed_budget
            and self.turns_used < self.hard_limit
        )

    def observe(self, result: dict[str, Any]) -> bool:
        """
        記錄工具結果，回傳該結果是否帶來新進展。
        """
        self.turns_used += 1
        progress = self._is_progress(result)
        if progress:
            self.progress_count += 1
            self.no_progress_streak = 0
            if self.allowed_budget < self.hard_limit:
                self.allowed_budget += 1
                self.extension_reasons.append(
                    f"turn_{self.turns_used}:new_valid_evidence"
                )
        else:
            self.no_progress_streak += 1

        if self.turns_used >= self.hard_limit:
            self._stop("hard_tool_turn_limit")
        elif self.no_progress_streak >= self.no_progress_limit:
            self._stop("consecutive_no_progress")
        elif self.turns_used >= self.allowed_budget:
            self._stop("adaptive_budget_exhausted")
        return progress

    def block_result(self, tool_name: str) -> dict[str, Any]:
        reason = self.stop_reason or "adaptive_budget_exhausted"
        return {
            "ok": False,
            "tool_name": tool_name,
            "status": "duplicate_blocked",
            "output_text": "",
            "raw_result": {"tool_turn_policy": self.snapshot()},
            "error_code": "tool_turn_budget_exhausted",
            "error_message": f"tool request blocked: {reason}",
            "error": f"tool request blocked: {reason}",
            "retryable": False,
            "retry_hint": "Return the best final answer using existing evidence.",
            "evidence_valid": False,
            "cache_hit": False,
            "duplicate_request": False,
        }

    def request_final_answer(self, reason: str) -> None:
        """
        Force the next Stage1 turn to produce a final answer without executing more tools.
        """
        self.force_final = True
        self.stop_reason = reason or self.stop_reason or "final_answer_required"

    def snapshot(self) -> dict[str, Any]:
        return {
            "base_budget": self.base_budget,
            "allowed_budget": self.allowed_budget,
            "hard_limit": self.hard_limit,
            "turns_used": self.turns_used,
            "remaining_turns": max(0, self.allowed_budget - self.turns_used),
            "progress_count": self.progress_count,
            "no_progress_streak": self.no_progress_streak,
            "extension_reasons": list(self.extension_reasons),
            "force_final": self.force_final,
            "stop_reason": self.stop_reason,
        }

    def format_prompt(self) -> str:
        state = self.snapshot()
        instruction = (
            "Return final_answer now; further tool requests are blocked."
            if self.force_final
            else "Request another tool only if it can add new evidence."
        )
        return (
            f"Base budget: {state['base_budget']}\n"
            f"Current budget: {state['allowed_budget']}\n"
            f"Hard limit: {state['hard_limit']}\n"
            f"Turns used: {state['turns_used']}\n"
            f"Remaining turns: {state['remaining_turns']}\n"
            f"No-progress streak: {state['no_progress_streak']}\n"
            f"Stop reason: {state['stop_reason'] or 'none'}\n"
            f"Instruction: {instruction}"
        )

    def _is_progress(self, result: dict[str, Any]) -> bool:
        if not result.get("evidence_valid", False):
            return False
        if result.get("status") in {
            "already_available",
            "duplicate_blocked",
            "unsupported",
            "fatal",
        }:
            return False
        content = str(result.get("output_text", "") or "").strip()
        if not content:
            content = repr(result.get("raw_result"))
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if fingerprint in self._evidence_fingerprints:
            return False
        self._evidence_fingerprints.add(fingerprint)
        return True

    def _stop(self, reason: str) -> None:
        self.force_final = True
        self.stop_reason = reason


__all__ = ["AdaptiveToolTurnPolicy"]
