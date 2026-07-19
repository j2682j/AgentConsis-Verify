from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class TaskAnswerRequirementContract:
    """保存單一任務在全流程共用的答案需求。"""

    requirement_text: str
    answer_role: str = ""
    answer_target: str = ""
    required_relation: str = ""
    required_relation_goal_id: str = ""
    source: str = "question_fallback"
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        question: str,
        answer_requirement: str = "",
        answer_role: str = "",
        answer_target: str = "",
        required_relation: str = "",
        required_relation_goal_id: str = "",
        relation_plan: Mapping[str, Any] | None = None,
        source: str = "",
    ) -> "TaskAnswerRequirementContract":
        requirement = normalize_text(answer_requirement)
        role = normalize_text(answer_role)
        target = normalize_text(answer_target)
        relation = normalize_text(required_relation)
        relation_goal_id = normalize_text(required_relation_goal_id)
        relation_payload = relation_plan if isinstance(relation_plan, Mapping) else {}
        goals = list(relation_payload.get("goals") or [])
        final_goal = goals[-1] if goals and isinstance(goals[-1], Mapping) else {}
        if not relation:
            relation = normalize_text(str(final_goal.get("relation") or ""))
        if not relation_goal_id:
            relation_goal_id = normalize_text(str(final_goal.get("goal_id") or ""))
        selected_source = normalize_text(source)
        if cls._placeholder(requirement):
            requirement = normalize_text(question)
            selected_source = "question_fallback"
        elif not selected_source:
            selected_source = "question_role_extractor"
        resolved = bool(requirement and not cls._placeholder(requirement))
        if not role:
            role = cls._infer_role(question)
        return cls(
            requirement_text=requirement,
            answer_role=role,
            answer_target=target,
            required_relation=relation,
            required_relation_goal_id=relation_goal_id,
            source=selected_source,
            resolved=resolved,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        question: str = "",
    ) -> "TaskAnswerRequirementContract":
        payload = value or {}
        return cls.build(
            question=question,
            answer_requirement=str(
                payload.get("requirement_text")
                or payload.get("answer_requirement")
                or ""
            ),
            answer_role=str(payload.get("answer_role") or ""),
            answer_target=str(payload.get("answer_target") or ""),
            required_relation=str(payload.get("required_relation") or ""),
            required_relation_goal_id=str(
                payload.get("required_relation_goal_id") or ""
            ),
            relation_plan=(
                payload.get("relation_plan")
                if isinstance(payload.get("relation_plan"), Mapping)
                else None
            ),
            source=str(payload.get("source") or ""),
        )

    @staticmethod
    def _placeholder(value: str) -> bool:
        return normalize_text(value).casefold() in {
            "",
            "unknown",
            "natural language answer requirement",
            "specific information required",
        }

    @staticmethod
    def _infer_role(question: str) -> str:
        text = normalize_text(question).casefold()
        question_clauses = re.findall(r"[^.!?]*\?", text)
        answer_clause = question_clauses[-1].strip() if question_clauses else text
        if re.search(r"\b(?:translate|translation)\b", text):
            return "translated_text"
        if re.match(
            r"^(?:can|could|does|do|did|is|are|was|were|has|have|will|would|should)\b",
            answer_clause,
        ):
            return "boolean"
        if re.search(
            r"\b(?:how many|number of|count of|total number)\b",
            answer_clause,
        ):
            return "count"
        if re.search(r"\b(?:who|whom|whose)\b", answer_clause):
            return "person"
        if re.search(
            r"\b(?:volume|distance|height|weight|duration|speed|area|capacity)\b",
            answer_clause,
        ):
            return "measurement"
        return ""


__all__ = ["TaskAnswerRequirementContract"]
