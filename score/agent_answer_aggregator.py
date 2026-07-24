from __future__ import annotations

from dataclasses import dataclass, field
import re

from core.config import EachAgentReply
from .answer_validator import AnswerValidator
from utils.network_utils import extract_math_answer, normalize_for_exact


# Presentation wrappers agents put around an answer without changing it.
# The inner-content guard keeps currency ("$5.00") from being treated as math.
_MATH_DELIMITED_RE = re.compile(r"^\s*\$+\s*(.+?)\s*\$+\s*$", re.DOTALL)
_LATEX_COMMAND_RE = re.compile(
    r"^\s*\\(?:boxed|text|mathrm|mathbf|textbf)\s*\{(.+)\}\s*$", re.DOTALL
)
_LIST_SEPARATOR_RE = re.compile(r"\s*,\s*")


@dataclass
class AgentAnswerAggregation:
    """
    單一 Agent 多次 final answer 的內部聚合結果。

    Args:
     - answer: 聚合後答案。
     - confidence_score: 依 3/3、2/3、1/3 一致性得到的 confidence。
     - status: 聚合狀態，例如 consensus_3_of_3、consensus_2_of_3、needs_review。
     - answer_counts: normalization 後的答案出現次數。
     - selected_run_indices: 被選中答案對應的 run index。
     - needs_review: 是否需要觸發一次 self-review。

    Returns:
     - AgentAnswerAggregation: 可寫入 AgentReasoningSummary metadata 的聚合結果。
    """

    answer: str = ""
    confidence_score: float = 0.0
    status: str = "no_valid_answers"
    answer_counts: dict[str, int] = field(default_factory=dict)
    selected_run_indices: list[int] = field(default_factory=list)
    needs_review: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "answer_counts": dict(self.answer_counts),
            "selected_run_indices": list(self.selected_run_indices),
            "needs_review": self.needs_review,
            "reason": self.reason,
        }


class AgentAnswerAggregator:
    """
    只做單一 Agent 內部 3-run 答案聚合，不做跨 Agent 比較。

    Args:
     - answer_validator: 用來清理與過濾 invalid / unknown / 過長答案。

    Returns:
     - AgentAnswerAggregator: 提供 aggregate(runs) 方法。
    """

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def aggregate(self, runs: list[EachAgentReply]) -> AgentAnswerAggregation:
        valid_runs = [
            run
            for run in runs or []
            if self._is_valid_run_answer(run)
        ]
        if not valid_runs:
            return AgentAnswerAggregation(
                confidence_score=0.0,
                status="no_valid_answers",
                needs_review=True,
                reason="no_valid_answers",
            )

        groups = self._group_runs(valid_runs)
        groups.sort(key=len, reverse=True)
        best_group = groups[0]
        best_answer = self._representative_answer(best_group)
        answer_counts = {
            self._answer_key(group[0].final_answer): len(group)
            for group in groups
            if self._answer_key(group[0].final_answer)
        }
        selected_run_indices = [int(run.run_index) for run in best_group]
        if len(best_group) >= 3:
            return AgentAnswerAggregation(
                answer=best_answer,
                confidence_score=1.0,
                status="consensus_3_of_3",
                answer_counts=answer_counts,
                selected_run_indices=selected_run_indices,
                needs_review=False,
                reason="three_runs_same_answer",
            )
        if len(best_group) == 2:
            return AgentAnswerAggregation(
                answer=best_answer,
                confidence_score=0.67,
                status="consensus_2_of_3",
                answer_counts=answer_counts,
                selected_run_indices=selected_run_indices,
                needs_review=False,
                reason="two_runs_same_answer",
            )
        return AgentAnswerAggregation(
            answer=best_answer,
            confidence_score=0.33,
            status="needs_review",
            answer_counts=answer_counts,
            selected_run_indices=selected_run_indices,
            needs_review=True,
            reason="all_valid_answers_different",
        )

    def _is_valid_run_answer(self, run: EachAgentReply) -> bool:
        answer = self.answer_validator.clean(getattr(run, "final_answer", ""))
        if not answer:
            return False
        if not getattr(run, "parse_completed", False):
            return False
        if not getattr(run, "eligible_for_winner", True):
            return False
        return self.answer_validator.is_valid(answer)

    def _group_runs(self, runs: list[EachAgentReply]) -> list[list[EachAgentReply]]:
        groups: list[list[EachAgentReply]] = []
        for run in runs:
            for group in groups:
                if self._answers_equivalent(run.final_answer, group[0].final_answer):
                    group.append(run)
                    break
            else:
                groups.append([run])
        return groups

    def _answers_equivalent(self, answer_a: str, answer_b: str) -> bool:
        if self._answer_key(answer_a) == self._answer_key(answer_b):
            return True
        math_a = extract_math_answer(answer_a)
        math_b = extract_math_answer(answer_b)
        return bool(math_a is not None and math_b is not None and math_a == math_b)

    def _representative_answer(self, group: list[EachAgentReply]) -> str:
        """Return the group's answer without its presentation wrapper.

        Runs in a group already agree, so only the wording is being chosen.
        Prefer a run that wrote the answer plainly: an agent whose first run
        wrote ``$\\boxed{2}$`` used to report that verbatim, which then failed
        to cluster with another agent's plain ``2`` and split the vote.

        Run order is otherwise preserved. Ranking members by anything else
        (shortest, for instance) silently rewrites answers that only look
        equivalent — it turned ``FF0099FF`` into ``0099FF`` and ``b, e`` into
        ``b,e`` on saved runs.
        """
        for run in group:
            raw = self.answer_validator.clean(getattr(run, "final_answer", ""))
            if raw and self._surface_form(getattr(run, "final_answer", "")) == raw:
                return raw
        return self._surface_form(group[0].final_answer)

    def _surface_form(self, answer: str) -> str:
        """Strip presentation-only wrappers without changing the answer."""
        text = self.answer_validator.clean(answer)
        for _ in range(3):
            before = text
            match = _MATH_DELIMITED_RE.match(text)
            # Only unwrap "$...$" when the content actually looks like math,
            # so a currency amount keeps its symbol.
            if match and ("\\" in match.group(1) or "{" in match.group(1)):
                text = match.group(1).strip()
            match = _LATEX_COMMAND_RE.match(text)
            if match:
                text = match.group(1).strip()
            if text == before:
                break
        return text

    def _answer_key(self, answer: str) -> str:
        cleaned = self._surface_form(answer)
        if not cleaned:
            return ""
        # Separator spacing is formatting, not content: "a, b" and "a,b" are the
        # same list, and treating them as different answers let a single
        # differing run outvote two runs that actually agreed.
        normalized = _LIST_SEPARATOR_RE.sub(", ", normalize_for_exact(cleaned))
        return normalized.strip().lower()


__all__ = ["AgentAnswerAggregation", "AgentAnswerAggregator"]
