from __future__ import annotations

from fractions import Fraction
import itertools
import math
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class ProbabilitySimulationRouterHandler:
    name = "probability_simulation"
    handler_role = "probability_simulation"
    capability_description = (
        "Compute exact probabilities for small dice or coin experiments by exhaustive enumeration."
    )
    supported_attachment_types: set[str] = {".txt"}
    supported_task_roles: set[str] = {"probability_simulation"}
    supported_answer_roles: set[str] = {"probability", "fraction", "number"}
    input_schema = io_contract(
        name,
        [
            input_field("experiment", "str", True, "Dice or coin random experiment.", "question"),
            input_field("event", "str", True, "Target event condition.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        parsed = self._parse(handler_input.combined_text())
        missing = []
        if not parsed.get("experiment"):
            missing.append("probability_experiment")
        if not parsed.get("event"):
            missing.append("probability_event")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.94 if not missing else 0.2,
            reason="small_probability_experiment_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        parsed = self._parse(handler_input.combined_text())
        parsed["question"] = handler_input.question
        return parsed

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        experiment = str(inputs.get("experiment") or "")
        event = str(inputs.get("event") or "")
        if experiment == "dice":
            count = int(inputs.get("count") or 0)
            sides = int(inputs.get("sides") or 0)
            if count <= 0 or sides <= 0 or count > 6 or sides > 30:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["small_dice_experiment"],
                    next_action_hint="Provide a dice experiment with at most 6 dice and 30 sides.",
                )
            favorable, total = self._enumerate_dice(count, sides, inputs)
        elif experiment == "coin":
            count = int(inputs.get("count") or 0)
            if count <= 0 or count > 20:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["small_coin_experiment"],
                    next_action_hint="Provide a coin experiment with at most 20 flips.",
                )
            favorable, total = self._enumerate_coins(count, inputs)
        else:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["probability_experiment"],
            )
        probability = Fraction(favorable, total)
        answer = self._format_fraction(probability)
        structured = {
            "task_type": "probability_simulation",
            "experiment": experiment,
            "event": event,
            "favorable": favorable,
            "total": total,
            "probability": str(probability),
            "decimal": float(probability),
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Experiment: {experiment}\n"
                f"Event: {event}\n"
                f"Favorable outcomes: {favorable}\n"
                f"Total outcomes: {total}\n"
                f"Answer: {answer}\n"
                "Instruction: use this exact probability for the stated random experiment."
            ),
            structured_result=structured,
            confidence=0.96,
            output_type="final_answer",
            semantic_role="probability_answer",
            supporting_inputs=[experiment, event],
        )

    def _parse(self, text: str) -> dict[str, Any]:
        lowered = str(text or "").lower()
        dice = re.search(r"(\d+)\s*d\s*(\d+)", lowered)
        if not dice:
            dice = re.search(r"(?:roll|rolling)\s+(\d+|one|two|three|four|five|six)\s+(?:fair\s+)?(?:dice|die)(?:\s+with\s+(\d+)\s+sides?)?", lowered)
        if dice:
            count = self._number(dice.group(1))
            sides = int(dice.group(2) or 6)
            event = self._parse_sum_event(lowered)
            return {
                "experiment": "dice",
                "count": count,
                "sides": sides,
                "event": event.get("event", ""),
                **event,
            }
        coin = re.search(r"(?:flip|toss|flipping|tossing)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:fair\s+)?coins?", lowered)
        if coin:
            count = self._number(coin.group(1))
            heads = self._parse_heads_event(lowered)
            return {
                "experiment": "coin",
                "count": count,
                "event": heads.get("event", ""),
                **heads,
            }
        return {"experiment": "", "event": ""}

    def _parse_sum_event(self, lowered: str) -> dict[str, Any]:
        patterns = [
            (r"sum\s+(?:is\s+)?(?:at\s+least|greater than or equal to)\s+(\d+)", ">="),
            (r"sum\s+(?:is\s+)?(?:greater than|more than|above)\s+(\d+)", ">"),
            (r"sum\s+(?:is\s+)?(?:at\s+most|less than or equal to)\s+(\d+)", "<="),
            (r"sum\s+(?:is\s+)?(?:less than|below)\s+(\d+)", "<"),
            (r"sum\s+(?:is\s+)?(?:exactly|equal to|equals|is)\s+(\d+)", "=="),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, lowered)
            if match:
                return {
                    "event": f"sum {operator} {match.group(1)}",
                    "operator": operator,
                    "target": int(match.group(1)),
                }
        return {"event": ""}

    def _parse_heads_event(self, lowered: str) -> dict[str, Any]:
        patterns = [
            (r"(?:exactly|equal to)\s+(\d+)\s+heads?", "=="),
            (r"(?:at\s+least)\s+(\d+)\s+heads?", ">="),
            (r"(?:at\s+most)\s+(\d+)\s+heads?", "<="),
            (r"(?:no|zero)\s+heads?", "==", 0),
        ]
        for item in patterns:
            pattern, operator = item[0], item[1]
            match = re.search(pattern, lowered)
            if match:
                target = int(item[2] if len(item) > 2 else match.group(1))
                return {"event": f"heads {operator} {target}", "operator": operator, "target": target}
        return {"event": ""}

    def _enumerate_dice(self, count: int, sides: int, inputs: dict[str, Any]) -> tuple[int, int]:
        operator = str(inputs.get("operator") or "")
        target = int(inputs.get("target") or 0)
        total = sides**count
        favorable = 0
        for outcome in itertools.product(range(1, sides + 1), repeat=count):
            if self._compare(sum(outcome), operator, target):
                favorable += 1
        return favorable, total

    def _enumerate_coins(self, count: int, inputs: dict[str, Any]) -> tuple[int, int]:
        operator = str(inputs.get("operator") or "")
        target = int(inputs.get("target") or 0)
        total = 2**count
        favorable = 0
        for outcome in itertools.product([0, 1], repeat=count):
            if self._compare(sum(outcome), operator, target):
                favorable += 1
        return favorable, total

    def _compare(self, value: int, operator: str, target: int) -> bool:
        if operator == ">=":
            return value >= target
        if operator == ">":
            return value > target
        if operator == "<=":
            return value <= target
        if operator == "<":
            return value < target
        if operator == "==":
            return value == target
        return False

    def _format_fraction(self, value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"

    def _number(self, value: str) -> int:
        words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        return int(value) if str(value).isdigit() else words.get(str(value).lower(), 0)


__all__ = ["ProbabilitySimulationRouterHandler"]
