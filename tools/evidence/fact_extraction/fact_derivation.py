from __future__ import annotations

import hashlib
import re

from utils.network_utils import normalize_text

from .derivation_models import FactDerivation, FactDerivationResult
from .answer_bound_validator import AnswerBoundFactValidator
from .fact_store import TaskFactStore
from .models import EvidenceFact


class FactDerivationEngine:
    """組合已 grounded 的關係鏈，過程不再呼叫語言模型。"""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        max_derivations: int = 64,
        answer_bound_validator: AnswerBoundFactValidator | None = None,
    ) -> None:
        self.max_depth = max(1, int(max_depth))
        self.max_derivations = max(1, int(max_derivations))
        self.answer_bound_validator = (
            answer_bound_validator or AnswerBoundFactValidator()
        )

    def derive(
        self,
        fact_store: TaskFactStore,
        *,
        answer_requirement: str = "",
        required_relation: str = "",
        required_relation_goal_id: str = "",
        answer_role: str = "",
    ) -> FactDerivationResult:
        derivations: list[FactDerivation] = []
        added_ids: list[str] = []
        base_facts = [
            fact
            for fact in fact_store.all()
            if (
                fact.grounding_status == "grounded"
                and fact.polarity == "positive"
                and not fact.parent_fact_ids
            )
        ]
        frontier = list(base_facts)
        for depth in range(1, self.max_depth + 1):
            next_frontier: list[EvidenceFact] = []
            for left in frontier:
                for right in base_facts:
                    if len(derivations) >= self.max_derivations:
                        break
                    if not self._can_compose(left, right):
                        continue
                    result_fact = self._compose(left, right, depth=depth)
                    result_fact = self.answer_bound_validator.bind(
                        result_fact,
                        question=answer_requirement,
                        answer_requirement=answer_requirement,
                    )
                    if not fact_store.add(result_fact):
                        continue
                    derivation = FactDerivation(
                        derivation_id=f"derivation-{result_fact.fact_id}",
                        derivation_type="relation_chain",
                        parent_fact_ids=list(result_fact.parent_fact_ids),
                        result_fact_id=result_fact.fact_id,
                        explanation=(
                            f"{left.subject} --{left.relation}--> {left.object}; "
                            f"{right.subject} --{right.relation}--> {right.object}"
                        ),
                        grounded=True,
                    )
                    derivations.append(derivation)
                    added_ids.append(result_fact.fact_id)
                    next_frontier.append(result_fact)
                if len(derivations) >= self.max_derivations:
                    break
            if not next_frontier or len(derivations) >= self.max_derivations:
                break
            frontier = next_frontier
        return FactDerivationResult(
            derivations=derivations,
            added_fact_ids=added_ids,
            diagnostics={
                "answer_requirement": normalize_text(answer_requirement),
                "required_relation": normalize_text(required_relation),
                "required_relation_goal_id": normalize_text(
                    required_relation_goal_id
                ),
                "answer_role": normalize_text(answer_role),
                "base_fact_count": len(base_facts),
                "derived_fact_count": len(added_ids),
                "max_depth": self.max_depth,
                "max_derivations": self.max_derivations,
            },
        )

    def _can_compose(self, left: EvidenceFact, right: EvidenceFact) -> bool:
        if left.fact_id == right.fact_id:
            return False
        if right.fact_id in left.parent_fact_ids:
            return False
        if self._same_entity(left.subject, left.object):
            return False
        if self._same_entity(right.subject, right.object):
            return False
        if not self._same_entity(left.object, right.subject):
            return False
        if self._same_entity(left.subject, right.object):
            return False
        return bool(left.subject and left.relation and right.relation and right.object)

    def _compose(
        self,
        left: EvidenceFact,
        right: EvidenceFact,
        *,
        depth: int,
    ) -> EvidenceFact:
        relation = f"{left.relation} -> {right.relation}"
        parent_ids = [left.fact_id, right.fact_id]
        fact_id = self._derived_fact_id(left.subject, relation, right.object, parent_ids)
        completes_goal = bool(
            left.goal_id
            and right.goal_id
            and normalize_text(left.goal_id).casefold()
            == normalize_text(right.goal_id).casefold()
        )
        role = (
            "ANSWER_SUPPORT"
            if right.role == "ANSWER_SUPPORT" or completes_goal
            else "BRIDGE"
        )
        context_parts = [left.context, right.context]
        context = "\n".join(part for part in context_parts if part).strip()
        spans = self._dedupe([*left.evidence_spans, *right.evidence_spans])[:4]
        return EvidenceFact(
            fact_id=fact_id,
            subject=left.subject,
            relation=relation,
            object=right.object,
            qualifiers={
                **left.qualifiers,
                **right.qualifiers,
                "derivation_depth": str(depth),
                "answer_binding": (
                    "direct" if role == "ANSWER_SUPPORT" else "bridge"
                ),
            },
            polarity="positive",
            role=role,
            goal_id=right.goal_id or left.goal_id,
            evidence_spans=spans,
            context=context,
            source_id=f"derived:{left.fact_id}+{right.fact_id}",
            source_type="derived",
            source_title="Relation derivation",
            grounding_status="grounded",
            extraction_method="fact_derivation",
            parent_fact_ids=parent_ids,
            derivation_type="relation_chain",
        )

    @staticmethod
    def _same_entity(first: str, second: str) -> bool:
        return FactDerivationEngine._entity_key(first) == FactDerivationEngine._entity_key(second)

    @staticmethod
    def _entity_key(value: str) -> str:
        text = normalize_text(value).casefold()
        text = re.sub(r"^(?:the|a|an)\s+", "", text)
        return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)

    @staticmethod
    def _derived_fact_id(
        subject: str,
        relation: str,
        object_value: str,
        parent_ids: list[str],
    ) -> str:
        payload = "\x1f".join([subject, relation, object_value, *parent_ids])
        return "derived-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result


__all__ = ["FactDerivationEngine"]
