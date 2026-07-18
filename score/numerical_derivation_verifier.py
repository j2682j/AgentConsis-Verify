from __future__ import annotations

import ast
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import Any, Iterable

from core.config import ToolEvidenceRecord


@dataclass(frozen=True)
class EvidenceQuantity:
    """One normalized numeric value with evidence provenance."""

    value: Decimal
    normalized_value: Decimal
    unit: str = ""
    dimension: str = "dimensionless"
    provenance_ids: tuple[str, ...] = ()
    source_tools: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    raw_text: str = ""
    derived: bool = False
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": str(self.value),
            "normalized_value": str(self.normalized_value),
            "unit": self.unit,
            "dimension": self.dimension,
            "provenance_ids": list(self.provenance_ids),
            "source_tools": list(self.source_tools),
            "goal_ids": list(self.goal_ids),
            "raw_text": self.raw_text,
            "derived": self.derived,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class NumericalStepVerification:
    """Deterministic verification result for one explicit calculation step."""

    step_index: int
    status: str
    reason: str
    expression: str = ""
    claimed_value: str = ""
    computed_value: str = ""
    unit: str = ""
    matched_values: tuple[str, ...] = ()
    provenance_ids: tuple[str, ...] = ()
    source_tools: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "status": self.status,
            "reason": self.reason,
            "expression": self.expression,
            "claimed_value": self.claimed_value,
            "computed_value": self.computed_value,
            "unit": self.unit,
            "matched_values": list(self.matched_values),
            "provenance_ids": list(self.provenance_ids),
            "source_tools": list(self.source_tools),
            "goal_ids": list(self.goal_ids),
        }


@dataclass(frozen=True)
class NumericalDerivationSummary:
    """Agent-level result for an evidence-grounded numerical derivation."""

    status: str = "not_applicable"
    final_supported: bool = False
    final_contradicted: bool = False
    terminal_value: str = ""
    step_results: list[NumericalStepVerification] = field(default_factory=list)
    evidence_quantities: list[EvidenceQuantity] = field(default_factory=list)
    provenance_ids: list[str] = field(default_factory=list)
    source_tools: list[str] = field(default_factory=list)
    goal_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_supported": self.final_supported,
            "final_contradicted": self.final_contradicted,
            "terminal_value": self.terminal_value,
            "step_results": [item.to_dict() for item in self.step_results],
            "evidence_quantities": [
                item.to_dict() for item in self.evidence_quantities
            ],
            "provenance_ids": list(self.provenance_ids),
            "source_tools": list(self.source_tools),
            "goal_ids": list(self.goal_ids),
            "reason": self.reason,
        }


class NumericalDerivationVerifier:
    """Verify explicit arithmetic as a provenance-preserving derivation chain."""

    _NUMBER = r"[-+]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    _TIME_RE = re.compile(r"(?<!\d)(\d{1,3}):(\d{2})(?::(\d{2}(?:\.\d+)?))?(?!\d)")
    _QUANTITY_RE = re.compile(
        rf"(?P<number>{_NUMBER})\s*(?P<unit>"
        r"thousand\s+hours?|km\s*/\s*h|km\s+per\s+hour|m\s*/\s*s|"
        r"miles?\s+per\s+hour|mph|kilomet(?:er|re)s?|km|met(?:er|re)s?|"
        r"miles?|mi|hours?|hrs?|h|minutes?|mins?|seconds?|secs?|s|m|"
        r"percent|%|square\s+(?:feet|foot|meters?|metres?)|"
        r"cubic\s+(?:meters?|metres?)|m\^?[23]|m[23])?",
        re.IGNORECASE,
    )
    _UNIT_RE = re.compile(
        r"\b(?:thousand\s+hours?|km\s*/\s*h|m\s*/\s*s|"
        r"km\s+per\s+hour|kilomet(?:er|re)s?|"
        r"met(?:er|re)s?|miles?\s+per\s+hour|miles?|hours?|hrs?|"
        r"minutes?|mins?|seconds?|secs?|percent|square\s+(?:feet|foot|"
        r"meters?|metres?)|cubic\s+(?:meters?|metres?))\b|"
        r"mph|m\^?[23]|m[23]|\b(?:km|mi|m|h|s)\b|%",
        re.IGNORECASE,
    )
    _ALLOWED_NAMES = {"abs", "round", "min", "max", "sqrt"}
    _ALLOWED_CONSTANTS = {
        Decimal("0"),
        Decimal("1"),
        Decimal("2"),
        Decimal("3"),
        Decimal("4"),
        Decimal("10"),
        Decimal("12"),
        Decimal("24"),
        Decimal("60"),
        Decimal("100"),
        Decimal("1000"),
        Decimal("3600"),
    }

    _UNITS: dict[str, tuple[str, str, Decimal]] = {
        "": ("", "dimensionless", Decimal("1")),
        "m": ("m", "distance", Decimal("1")),
        "meter": ("m", "distance", Decimal("1")),
        "metre": ("m", "distance", Decimal("1")),
        "km": ("km", "distance", Decimal("1000")),
        "kilometer": ("km", "distance", Decimal("1000")),
        "kilometre": ("km", "distance", Decimal("1000")),
        "mile": ("mile", "distance", Decimal("1609.344")),
        "mi": ("mile", "distance", Decimal("1609.344")),
        "s": ("s", "time", Decimal("1")),
        "sec": ("s", "time", Decimal("1")),
        "second": ("s", "time", Decimal("1")),
        "min": ("min", "time", Decimal("60")),
        "minute": ("min", "time", Decimal("60")),
        "h": ("h", "time", Decimal("3600")),
        "hr": ("h", "time", Decimal("3600")),
        "hour": ("h", "time", Decimal("3600")),
        "thousand hour": ("thousand hour", "time", Decimal("3600000")),
        "km/h": ("km/h", "speed", Decimal("1000") / Decimal("3600")),
        "km per hour": ("km/h", "speed", Decimal("1000") / Decimal("3600")),
        "m/s": ("m/s", "speed", Decimal("1")),
        "mile per hour": (
            "mph",
            "speed",
            Decimal("1609.344") / Decimal("3600"),
        ),
        "mph": ("mph", "speed", Decimal("1609.344") / Decimal("3600")),
        "%": ("%", "ratio", Decimal("0.01")),
        "percent": ("%", "ratio", Decimal("0.01")),
        "square foot": ("square foot", "area", Decimal("0.09290304")),
        "square meter": ("square meter", "area", Decimal("1")),
        "cubic meter": ("cubic meter", "volume", Decimal("1")),
        "m2": ("square meter", "area", Decimal("1")),
        "m^2": ("square meter", "area", Decimal("1")),
        "m3": ("cubic meter", "volume", Decimal("1")),
        "m^3": ("cubic meter", "volume", Decimal("1")),
    }

    def verify(
        self,
        *,
        question: str,
        reasoning_steps: list[tuple[int, str]],
        final_answer: str,
        records: list[ToolEvidenceRecord],
    ) -> NumericalDerivationSummary:
        evidence_quantities = self.extract_evidence_quantities(records)
        known = list(evidence_quantities)
        known.extend(self._question_quantities(question))
        step_results: list[NumericalStepVerification] = []
        derived_values: list[EvidenceQuantity] = []

        for step_index, step_text in reasoning_steps:
            verification, derived = self._verify_step(
                step_index=step_index,
                step_text=step_text,
                known=known,
            )
            if verification.status != "not_applicable":
                step_results.append(verification)
            if derived is not None:
                known.append(derived)
                derived_values.append(derived)

        contradicted_steps = [
            item for item in step_results if item.status == "contradicted"
        ]
        terminal = derived_values[-1] if derived_values else None
        final_quantity = self._final_answer_quantity(final_answer, question)
        final_supported = bool(
            terminal is not None
            and final_quantity is not None
            and self._quantities_equivalent(terminal, final_quantity)
            and terminal.provenance_ids
        )
        final_contradicted = bool(
            terminal is not None
            and final_quantity is not None
            and terminal.provenance_ids
            and not self._quantities_equivalent(terminal, final_quantity)
        )
        provenance_ids = self._unique(
            item for value in derived_values for item in value.provenance_ids
        )
        source_tools = self._unique(
            item for value in derived_values for item in value.source_tools
        )
        goal_ids = self._unique(
            item for value in derived_values for item in value.goal_ids
        )

        if contradicted_steps or final_contradicted:
            status = "contradicted"
            reason = (
                "derived_final_conflicts_with_verified_terminal"
                if final_contradicted
                else "calculation_step_conflicts_with_recomputed_value"
            )
        elif final_supported:
            status = "derived_evidence_supported"
            reason = "final_answer_matches_evidence_grounded_derivation"
        elif step_results:
            status = "incomplete"
            reason = "numeric_derivation_does_not_reach_final_answer"
        else:
            status = "not_applicable"
            reason = "no_explicit_numeric_derivation"
        return NumericalDerivationSummary(
            status=status,
            final_supported=final_supported,
            final_contradicted=final_contradicted,
            terminal_value=(str(terminal.value) if terminal is not None else ""),
            step_results=step_results,
            evidence_quantities=evidence_quantities,
            provenance_ids=provenance_ids,
            source_tools=source_tools,
            goal_ids=goal_ids,
            reason=reason,
        )

    def extract_evidence_quantities(
        self,
        records: Iterable[ToolEvidenceRecord],
    ) -> list[EvidenceQuantity]:
        output: list[EvidenceQuantity] = []
        seen: set[tuple[Any, ...]] = set()
        for record_index, record in enumerate(records, start=1):
            if not record.evidence_valid or record.output_type == "failed":
                continue
            metadata = record.metadata if isinstance(record.metadata, dict) else {}
            evidence_id = str(metadata.get("evidence_id") or "")
            source_id = str(metadata.get("source_id") or "")
            provenance_id = evidence_id or source_id or f"{record.tool_name}:{record_index}"
            goal_ids = self._string_values(metadata.get("goal_ids"))
            texts = [record.value, record.evidence_text]
            texts.extend(self._string_values(metadata.get("answer_spans")))
            for text in texts:
                for quantity in self._extract_quantities(
                    text,
                    provenance_ids=(provenance_id,),
                    source_tools=(record.tool_name,),
                    goal_ids=tuple(goal_ids),
                ):
                    key = (
                        quantity.normalized_value,
                        quantity.dimension,
                        quantity.provenance_ids,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(quantity)
        return output

    def _verify_step(
        self,
        *,
        step_index: int,
        step_text: str,
        known: list[EvidenceQuantity],
    ) -> tuple[NumericalStepVerification, EvidenceQuantity | None]:
        parsed = self._parse_equation(step_text)
        if parsed is None:
            return (
                NumericalStepVerification(
                    step_index=step_index,
                    status="not_applicable",
                    reason="no_explicit_equation",
                ),
                None,
            )
        expression, claimed = parsed
        try:
            node = ast.parse(expression, mode="eval").body
            literal_values = self._literal_values(node)
        except (SyntaxError, ValueError):
            return (
                NumericalStepVerification(
                    step_index=step_index,
                    status="unsupported",
                    reason="unparseable_calculation_expression",
                    expression=expression,
                    claimed_value=str(claimed.value),
                    unit=claimed.unit,
                ),
                None,
            )

        matched_known: list[EvidenceQuantity] = []
        missing_literals: list[Decimal] = []
        for literal in literal_values:
            match = self._match_known_value(literal, known)
            if match is not None:
                matched_known.append(match)
            elif literal not in self._ALLOWED_CONSTANTS:
                missing_literals.append(literal)
        if missing_literals or not any(item.provenance_ids for item in matched_known):
            return (
                NumericalStepVerification(
                    step_index=step_index,
                    status="unsupported",
                    reason="calculation_uses_ungrounded_numeric_inputs",
                    expression=expression,
                    claimed_value=str(claimed.value),
                    unit=claimed.unit,
                    matched_values=tuple(
                        self._unique(str(item.value) for item in matched_known)
                    ),
                ),
                None,
            )

        try:
            computed, inferred_dimension = self._evaluate(node, known)
        except (ArithmeticError, InvalidOperation, ValueError, TypeError):
            return (
                NumericalStepVerification(
                    step_index=step_index,
                    status="unsupported",
                    reason="calculation_could_not_be_recomputed",
                    expression=expression,
                    claimed_value=str(claimed.value),
                    unit=claimed.unit,
                ),
                None,
            )

        provenance_ids = tuple(
            self._unique(
                item for value in matched_known for item in value.provenance_ids
            )
        )
        source_tools = tuple(
            self._unique(item for value in matched_known for item in value.source_tools)
        )
        goal_ids = tuple(
            self._unique(item for value in matched_known for item in value.goal_ids)
        )
        dimension_conflict = bool(
            claimed.dimension not in {"dimensionless", "unknown"}
            and inferred_dimension not in {
                "dimensionless",
                "unknown",
                claimed.dimension,
            }
        )
        if dimension_conflict or not self._decimal_close(
            computed,
            claimed.value,
            claimed.raw_text,
        ):
            return (
                NumericalStepVerification(
                    step_index=step_index,
                    status="contradicted",
                    reason=(
                        "calculation_unit_dimension_conflict"
                        if dimension_conflict
                        else "calculation_result_mismatch"
                    ),
                    expression=expression,
                    claimed_value=str(claimed.value),
                    computed_value=str(computed),
                    unit=claimed.unit,
                    matched_values=tuple(
                        self._unique(str(item.value) for item in matched_known)
                    ),
                    provenance_ids=provenance_ids,
                    source_tools=source_tools,
                    goal_ids=goal_ids,
                ),
                None,
            )

        output_dimension = (
            claimed.dimension
            if claimed.dimension != "dimensionless" or inferred_dimension == "dimensionless"
            else inferred_dimension
        )
        output_unit = claimed.unit
        normalized_value = (
            claimed.normalized_value
            if output_unit
            else claimed.value
        )
        derived = EvidenceQuantity(
            value=claimed.value,
            normalized_value=normalized_value,
            unit=output_unit,
            dimension=output_dimension,
            provenance_ids=provenance_ids,
            source_tools=source_tools,
            goal_ids=goal_ids,
            raw_text=claimed.raw_text,
            derived=True,
            step_index=step_index,
        )
        return (
            NumericalStepVerification(
                step_index=step_index,
                status="derived_supported",
                reason="calculation_recomputed_from_grounded_values",
                expression=expression,
                claimed_value=str(claimed.value),
                computed_value=str(computed),
                unit=output_unit,
                matched_values=tuple(
                    self._unique(str(item.value) for item in matched_known)
                ),
                provenance_ids=provenance_ids,
                source_tools=source_tools,
                goal_ids=goal_ids,
            ),
            derived,
        )

    def _parse_equation(
        self,
        text: str,
    ) -> tuple[str, EvidenceQuantity] | None:
        normalized = (
            str(text or "")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("≈", "=")
            .replace("^", "**")
        )
        parts = [part.strip() for part in normalized.split("=") if part.strip()]
        if len(parts) < 2:
            return None
        for index in range(len(parts) - 2, -1, -1):
            expression = self._sanitize_expression(parts[index])
            claimed_quantities = self._extract_quantities(parts[index + 1])
            if expression and claimed_quantities:
                return expression, claimed_quantities[0]
        return None

    def _sanitize_expression(self, text: str) -> str:
        candidate = str(text or "")
        if ":" in candidate and not self._TIME_RE.search(candidate):
            candidate = candidate.rsplit(":", 1)[-1]
        candidate = self._TIME_RE.sub(self._time_as_hours, candidate)
        candidate = re.sub(r"(?<=\d),(?=\d)", "", candidate)
        candidate = self._UNIT_RE.sub("", candidate)
        candidate = re.sub(r"\b(?:calc(?:ulation)?|result|therefore|thus|is)\b", " ", candidate, flags=re.I)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .;,")
        if not re.search(r"[+\-*/()]|\b(?:round|abs|min|max|sqrt)\s*\(", candidate):
            return ""
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", candidate))
        if names - self._ALLOWED_NAMES:
            return ""
        if re.search(r"[^0-9eE+\-*/()., _A-Za-z]", candidate):
            return ""
        return candidate

    def _evaluate(
        self,
        node: ast.AST,
        known: list[EvidenceQuantity],
    ) -> tuple[Decimal, str]:
        with localcontext() as context:
            context.prec = 32
            return self._evaluate_node(node, known)

    def _evaluate_node(
        self,
        node: ast.AST,
        known: list[EvidenceQuantity],
    ) -> tuple[Decimal, str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Decimal(str(node.value))
            match = self._match_known_value(value, known)
            return value, match.dimension if match is not None else "dimensionless"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value, dimension = self._evaluate_node(node.operand, known)
            return (-value if isinstance(node.op, ast.USub) else value), dimension
        if isinstance(node, ast.BinOp):
            left, left_dimension = self._evaluate_node(node.left, known)
            right, right_dimension = self._evaluate_node(node.right, known)
            if isinstance(node.op, ast.Add):
                return left + right, self._same_dimension(left_dimension, right_dimension)
            if isinstance(node.op, ast.Sub):
                return left - right, self._same_dimension(left_dimension, right_dimension)
            if isinstance(node.op, ast.Mult):
                return left * right, self._multiply_dimension(left_dimension, right_dimension)
            if isinstance(node.op, ast.Div):
                return left / right, self._divide_dimension(left_dimension, right_dimension)
            if isinstance(node.op, ast.Pow):
                return left**int(right), self._power_dimension(left_dimension, right)
            raise ValueError("unsupported arithmetic operator")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in self._ALLOWED_NAMES:
                raise ValueError("unsupported function")
            values = [self._evaluate_node(argument, known) for argument in node.args]
            numbers = [item[0] for item in values]
            dimension = values[0][1] if values else "dimensionless"
            if name == "abs":
                return abs(numbers[0]), dimension
            if name == "round":
                digits = int(numbers[1]) if len(numbers) > 1 else 0
                quantum = Decimal("1").scaleb(-digits)
                return numbers[0].quantize(quantum), dimension
            if name == "min":
                return min(numbers), dimension
            if name == "max":
                return max(numbers), dimension
            if name == "sqrt":
                return numbers[0].sqrt(), self._sqrt_dimension(dimension)
        raise ValueError("unsupported expression node")

    def _literal_values(self, node: ast.AST) -> list[Decimal]:
        output: list[Decimal] = []
        for item in ast.walk(node):
            if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
                output.append(Decimal(str(item.value)))
        return output

    def _extract_quantities(
        self,
        text: str,
        *,
        provenance_ids: tuple[str, ...] = (),
        source_tools: tuple[str, ...] = (),
        goal_ids: tuple[str, ...] = (),
    ) -> list[EvidenceQuantity]:
        raw = str(text or "")
        if not raw.strip():
            return []
        output: list[EvidenceQuantity] = []
        occupied: list[tuple[int, int]] = []
        for match in self._TIME_RE.finditer(raw):
            value = self._time_decimal_hours(match)
            output.append(
                self._quantity(
                    value=value,
                    unit="hour",
                    raw_text=match.group(0),
                    provenance_ids=provenance_ids,
                    source_tools=source_tools,
                    goal_ids=goal_ids,
                )
            )
            occupied.append(match.span())
        for match in self._QUANTITY_RE.finditer(raw):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            try:
                value = Decimal(
                    match.group("number").replace(",", "").replace(" ", "")
                )
            except InvalidOperation:
                continue
            output.append(
                self._quantity(
                    value=value,
                    unit=match.group("unit") or "",
                    raw_text=match.group(0),
                    provenance_ids=provenance_ids,
                    source_tools=source_tools,
                    goal_ids=goal_ids,
                )
            )
        return output

    def _quantity(
        self,
        *,
        value: Decimal,
        unit: str,
        raw_text: str,
        provenance_ids: tuple[str, ...],
        source_tools: tuple[str, ...],
        goal_ids: tuple[str, ...],
    ) -> EvidenceQuantity:
        canonical, dimension, factor = self._normalize_unit(unit)
        return EvidenceQuantity(
            value=value,
            normalized_value=value * factor,
            unit=canonical,
            dimension=dimension,
            provenance_ids=provenance_ids,
            source_tools=source_tools,
            goal_ids=goal_ids,
            raw_text=raw_text,
        )

    def _normalize_unit(self, unit: str) -> tuple[str, str, Decimal]:
        key = re.sub(r"\s+", " ", str(unit or "").strip().casefold())
        key = re.sub(r"\s*/\s*", "/", key)
        key = key.replace("kilometres", "kilometre").replace("kilometers", "kilometer")
        key = key.replace("metres", "metre").replace("meters", "meter")
        key = key.replace("miles", "mile").replace("hours", "hour")
        key = key.replace("hrs", "hr").replace("minutes", "minute")
        key = key.replace("mins", "min").replace("seconds", "second")
        key = key.replace("secs", "sec").replace("square feet", "square foot")
        key = key.replace("square meters", "square meter").replace("square metres", "square meter")
        key = key.replace("cubic meters", "cubic meter").replace("cubic metres", "cubic meter")
        return self._UNITS.get(key, (key, "unknown", Decimal("1")))

    def _question_quantities(self, question: str) -> list[EvidenceQuantity]:
        return [
            EvidenceQuantity(
                value=item.value,
                normalized_value=item.normalized_value,
                unit=item.unit,
                dimension=item.dimension,
                provenance_ids=(),
                source_tools=("question",),
                raw_text=item.raw_text,
            )
            for item in self._extract_quantities(question)
        ]

    def _final_answer_quantity(
        self,
        final_answer: str,
        question: str,
    ) -> EvidenceQuantity | None:
        quantities = self._extract_quantities(final_answer)
        if not quantities:
            return None
        quantity = quantities[0]
        if quantity.unit:
            return quantity
        question_key = str(question or "").casefold()
        if re.search(
            r"(?:how many|in|as)\s+thousand\s+hours?",
            question_key,
        ):
            return self._quantity(
                value=quantity.value,
                unit="thousand hours",
                raw_text=quantity.raw_text,
                provenance_ids=(),
                source_tools=(),
                goal_ids=(),
            )
        return quantity

    def _match_known_value(
        self,
        value: Decimal,
        known: list[EvidenceQuantity],
    ) -> EvidenceQuantity | None:
        candidates = [item for item in known if self._decimal_close(value, item.value)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                bool(item.provenance_ids),
                item.derived,
                bool(item.unit),
            ),
        )

    def _quantities_equivalent(
        self,
        left: EvidenceQuantity,
        right: EvidenceQuantity,
    ) -> bool:
        if (
            left.dimension not in {"dimensionless", "unknown"}
            and right.dimension not in {"dimensionless", "unknown"}
            and left.dimension != right.dimension
        ):
            return False
        if left.unit and right.unit:
            return self._decimal_close(left.normalized_value, right.normalized_value)
        return self._decimal_close(left.value, right.value)

    def _decimal_close(
        self,
        left: Decimal,
        right: Decimal,
        displayed: str = "",
    ) -> bool:
        places = self._decimal_places(displayed)
        display_tolerance = Decimal("0.5").scaleb(-places) if places >= 0 else Decimal("0")
        relative_tolerance = max(abs(left), abs(right), Decimal("1")) * Decimal("0.000001")
        return abs(left - right) <= max(display_tolerance, relative_tolerance)

    def _decimal_places(self, text: str) -> int:
        match = re.search(r"[-+]?\d[\d, ]*(?:\.(\d+))?", str(text or ""))
        if not match:
            return -1
        return len(match.group(1) or "")

    def _time_decimal_hours(self, match: re.Match[str]) -> Decimal:
        hours = Decimal(match.group(1))
        minutes = Decimal(match.group(2))
        seconds = Decimal(match.group(3) or "0")
        return hours + minutes / Decimal("60") + seconds / Decimal("3600")

    def _time_as_hours(self, match: re.Match[str]) -> str:
        return str(self._time_decimal_hours(match))

    def _same_dimension(self, left: str, right: str) -> str:
        if left == right:
            return left
        if left == "dimensionless":
            return right
        if right == "dimensionless":
            return left
        return "unknown"

    def _multiply_dimension(self, left: str, right: str) -> str:
        if left == "dimensionless":
            return right
        if right == "dimensionless":
            return left
        if {left, right} == {"speed", "time"}:
            return "distance"
        if left == right == "distance":
            return "area"
        return "unknown"

    def _divide_dimension(self, left: str, right: str) -> str:
        if right == "dimensionless":
            return left
        if left == right:
            return "dimensionless"
        if left == "distance" and right == "time":
            return "speed"
        if left == "distance" and right == "speed":
            return "time"
        return "unknown"

    def _power_dimension(self, dimension: str, exponent: Decimal) -> str:
        if dimension == "distance" and exponent == Decimal("2"):
            return "area"
        if dimension == "distance" and exponent == Decimal("3"):
            return "volume"
        return dimension if exponent == Decimal("1") else "unknown"

    def _sqrt_dimension(self, dimension: str) -> str:
        if dimension == "area":
            return "distance"
        return "unknown" if dimension != "dimensionless" else "dimensionless"

    def _string_values(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item)]
        return []

    def _unique(self, values: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            output.append(text)
            seen.add(key)
        return output


__all__ = [
    "EvidenceQuantity",
    "NumericalDerivationSummary",
    "NumericalDerivationVerifier",
    "NumericalStepVerification",
]
