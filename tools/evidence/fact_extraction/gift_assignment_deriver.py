from __future__ import annotations

from functools import lru_cache
import hashlib
import re
from typing import Any

from utils.network_utils import normalize_for_exact, normalize_text, semantic_similarity_score

from .completeness_contract import CompletenessContractBuilder
from .models import EvidenceFact
from .set_difference_deriver import SetDifferenceFactDeriver


class GiftAssignmentFactDeriver:
    """由完整 assignment、profile 與 gift 集合推導缺少的 giver。"""

    def __init__(self) -> None:
        self.contract_builder = CompletenessContractBuilder()
        self.set_deriver = SetDifferenceFactDeriver()

    def derive(
        self,
        *,
        question: str,
        parsed_payload: dict[str, Any],
        base_facts: list[EvidenceFact],
        source_id: str,
        source_type: str,
    ) -> tuple[list[EvidenceFact], dict[str, Any]]:
        if not re.search(r"\b(?:gift|secret santa)\b", question, flags=re.IGNORECASE):
            return [], {"status": "not_applicable"}
        assignments = self._assignments(parsed_payload)
        profiles = self._profiles(parsed_payload)
        gifts = self._gifts(parsed_payload, people=set(profiles) | {a for a, _ in assignments})
        if not assignments or not profiles or not gifts:
            return [], {"status": "incomplete_inputs"}
        recipients = [recipient for _, recipient in assignments]
        if len(gifts) != len(recipients) - 1 or set(recipients) != set(profiles):
            return [], {"status": "scope_not_closed"}

        matching = self._maximum_matching(gifts, profiles)
        if len(matching) != len(gifts):
            return [], {"status": "matching_failed"}
        observed = [person for _, person, _ in matching]
        missing_recipients = [person for person in recipients if person not in observed]
        if len(missing_recipients) != 1:
            return [], {"status": "set_difference_not_unique"}
        missing_recipient = missing_recipients[0]
        missing_givers = [giver for giver, recipient in assignments if recipient == missing_recipient]
        if len(missing_givers) != 1:
            return [], {"status": "inverse_assignment_not_unique"}
        missing_giver = missing_givers[0]

        parent_ids = [fact.fact_id for fact in base_facts if fact.fact_id]
        raw = "\x1f".join([source_id, missing_giver, missing_recipient, *sorted(observed)])
        fact = EvidenceFact(
            fact_id="gift-diff-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
            subject=missing_giver,
            relation="did_not_give",
            object="gift",
            qualifiers={
                "answer_binding": "direct",
                "answer_requirement": normalize_text(question),
                "negation_type": "closed_world_set_difference",
                "completeness_contract_ids": "gift-assignment-complete",
                "missing_recipient": missing_recipient,
            },
            polarity="negative",
            role="ANSWER_SUPPORT",
            evidence_spans=[
                f"{missing_giver} was assigned to {missing_recipient}; no gift matched {missing_recipient}'s profile."
            ],
            context=(
                f"Complete assignment and gift sets imply {missing_giver} did not give a gift "
                f"to assigned recipient {missing_recipient}."
            ),
            source_id=source_id,
            source_type=source_type,
            grounding_status="grounded",
            extraction_method="structured_global_assignment",
            parent_fact_ids=parent_ids,
            derivation_type="closed_world_set_difference",
        )
        return [fact], {
            "status": "derived",
            "assignment_count": len(assignments),
            "profile_count": len(profiles),
            "gift_count": len(gifts),
            "matching": [
                {"gift": gift, "recipient": person, "semantic_similarity": round(score, 6)}
                for gift, person, score in matching
            ],
            "missing_recipient": missing_recipient,
            "missing_giver": missing_giver,
        }

    @staticmethod
    def _assignments(payload: dict[str, Any]) -> list[tuple[str, str]]:
        for table in list(payload.get("tables") or []):
            if not isinstance(table, dict):
                continue
            rows = list(table.get("rows") or [])
            pairs = [
                (normalize_text(str(row[0])), normalize_text(str(row[1])))
                for row in rows
                if isinstance(row, list) and len(row) >= 2 and row[0] and row[1]
            ]
            if pairs:
                return pairs
        return []

    @staticmethod
    def _profiles(payload: dict[str, Any]) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for block in list(payload.get("text_blocks") or []):
            if not isinstance(block, dict):
                continue
            text = normalize_text(str(block.get("text") or ""))
            match = re.match(r"^([^:]{1,80}):\s*(.+)$", text)
            if not match:
                continue
            interests = [normalize_text(item) for item in match.group(2).split(",") if normalize_text(item)]
            if interests:
                output[normalize_text(match.group(1))] = interests
        return output

    @staticmethod
    def _gifts(payload: dict[str, Any], *, people: set[str]) -> list[str]:
        people_keys = {normalize_for_exact(person) for person in people}
        values: list[str] = []
        for block in list(payload.get("lists") or []):
            if not isinstance(block, dict):
                continue
            for item in list(block.get("items") or []):
                text = normalize_text(str(item)).strip("¨〃\"'")
                if text and normalize_for_exact(text) not in people_keys:
                    values.append(text)
        return list(dict.fromkeys(values))

    def _maximum_matching(
        self,
        gifts: list[str],
        profiles: dict[str, list[str]],
    ) -> list[tuple[str, str, float]]:
        people = list(profiles)
        weights = [
            [
                max(
                    (semantic_similarity_score(gift, interest) or 0.0)
                    for interest in profiles[person]
                )
                for person in people
            ]
            for gift in gifts
        ]

        @lru_cache(maxsize=None)
        def solve(gift_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
            if gift_index == len(gifts):
                return 0.0, ()
            best_score = float("-inf")
            best_path: tuple[int, ...] = ()
            for person_index in range(len(people)):
                if used_mask & (1 << person_index):
                    continue
                tail_score, tail_path = solve(
                    gift_index + 1,
                    used_mask | (1 << person_index),
                )
                total = weights[gift_index][person_index] + tail_score
                if total > best_score:
                    best_score = total
                    best_path = (person_index, *tail_path)
            return best_score, best_path

        _, path = solve(0, 0)
        return [
            (gift, people[person_index], weights[index][person_index])
            for index, (gift, person_index) in enumerate(zip(gifts, path))
        ]


__all__ = ["GiftAssignmentFactDeriver"]
