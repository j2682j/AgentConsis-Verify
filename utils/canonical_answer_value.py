from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from utils.network_utils import normalize_for_exact, normalize_text


@dataclass(frozen=True)
class CanonicalAnswerValue:
    """表示可跨 parser、Fact Store 與 evaluator 比對的答案值。"""

    raw_text: str
    normalized_text: str
    value_type: str
    numeric_value: Decimal | None = None
    dimension: str = ""
    canonical_unit: str = ""
    unit_explicit: bool = False
    unit_inherited_from_question: bool = False


class CanonicalAnswerValueParser:
    """正規化短答案，並保留單位維度與題目隱含單位。"""

    _MEASUREMENT_RE = re.compile(
        r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*"
        r"(m(?:\^?3|³)|cubic\s+met(?:er|re)s?|l(?:iter|itre)s?|km|mi|m|cm|mm|%|percent)?\s*$",
        re.IGNORECASE,
    )
    _UNIT_IN_QUESTION_RE = re.compile(
        r"\b(m(?:\^?3|³)|cubic\s+met(?:er|re)s?|l(?:iter|itre)s?|km|mi|cm|mm|percent|%)\b",
        re.IGNORECASE,
    )

    def parse(
        self,
        value: str,
        *,
        answer_requirement: str = "",
    ) -> CanonicalAnswerValue:
        raw = normalize_text(unicodedata.normalize("NFKC", str(value or "")))
        match = self._MEASUREMENT_RE.fullmatch(raw)
        if not match:
            return CanonicalAnswerValue(
                raw_text=raw,
                normalized_text=normalize_for_exact(raw),
                value_type="text",
            )
        number = self._decimal(match.group(1))
        explicit_unit = self._normalize_unit(match.group(2) or "")
        inherited = False
        unit = explicit_unit
        if not unit:
            implied = self._question_unit(answer_requirement)
            if implied:
                unit = implied
                inherited = True
        dimension, canonical_unit, canonical_number = self._canonical_measurement(
            number,
            unit,
        )
        value_type = "measurement" if unit else "number"
        normalized = self._format_decimal(canonical_number)
        if canonical_unit:
            normalized = f"{normalized} {canonical_unit}"
        return CanonicalAnswerValue(
            raw_text=raw,
            normalized_text=normalized,
            value_type=value_type,
            numeric_value=canonical_number,
            dimension=dimension,
            canonical_unit=canonical_unit,
            unit_explicit=bool(explicit_unit),
            unit_inherited_from_question=inherited,
        )

    def equivalent(
        self,
        first: str,
        second: str,
        *,
        answer_requirement: str = "",
    ) -> bool:
        left = self.parse(first, answer_requirement=answer_requirement)
        right = self.parse(second, answer_requirement=answer_requirement)
        if left.normalized_text == right.normalized_text and left.normalized_text:
            return True
        if left.numeric_value is None or right.numeric_value is None:
            return False
        if left.dimension and right.dimension and left.dimension != right.dimension:
            return False
        if bool(left.dimension) != bool(right.dimension):
            return False
        return left.numeric_value == right.numeric_value

    def _question_unit(self, requirement: str) -> str:
        matches = {
            self._normalize_unit(match.group(1))
            for match in self._UNIT_IN_QUESTION_RE.finditer(requirement or "")
        }
        matches.discard("")
        return next(iter(matches)) if len(matches) == 1 else ""

    @staticmethod
    def _normalize_unit(value: str) -> str:
        unit = normalize_text(value).casefold().replace("³", "3").replace("^", "")
        unit = re.sub(r"\s+", " ", unit)
        if unit in {"m3", "cubic meter", "cubic meters", "cubic metre", "cubic metres"}:
            return "m3"
        if unit in {"l", "liter", "liters", "litre", "litres"}:
            return "l"
        if unit in {"%", "percent"}:
            return "%"
        return unit

    def _canonical_measurement(
        self,
        number: Decimal | None,
        unit: str,
    ) -> tuple[str, str, Decimal | None]:
        if number is None:
            return "", unit, None
        if unit == "l":
            return "volume", "m3", number / Decimal("1000")
        if unit == "m3":
            return "volume", "m3", number
        dimensions = {
            "km": "distance",
            "mi": "distance",
            "m": "distance",
            "cm": "distance",
            "mm": "distance",
            "%": "percentage",
        }
        return dimensions.get(unit, ""), unit, number

    @staticmethod
    def _decimal(value: str) -> Decimal | None:
        try:
            return Decimal(value.replace(",", ""))
        except (InvalidOperation, AttributeError):
            return None

    @staticmethod
    def _format_decimal(value: Decimal | None) -> str:
        if value is None:
            return ""
        return format(value.normalize(), "f")


__all__ = ["CanonicalAnswerValue", "CanonicalAnswerValueParser"]
