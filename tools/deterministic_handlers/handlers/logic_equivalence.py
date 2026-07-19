from __future__ import annotations

import itertools
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class LogicEquivalenceRouterHandler:
    name = "logic_equivalence"
    handler_role = "logic_equivalence"
    capability_description = (
        "Check whether two small propositional-logic expressions are logically equivalent "
        "by exhaustive truth-table evaluation."
    )
    supported_attachment_types: set[str] = {".txt"}
    supported_task_roles: set[str] = {"logic_equivalence"}
    supported_answer_roles: set[str] = {"boolean", "yes_no"}
    input_schema = io_contract(
        name,
        [
            input_field("expression_a", "str", True, "First logical expression.", "question|attachment"),
            input_field("expression_b", "str", True, "Second logical expression.", "question|attachment"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        text = handler_input.combined_text()
        if not self._has_logic_operation(text):
            return HandlerMatch(
                handler_name=self.name,
                matched=False,
                confidence=0.0,
                reason="logic_operation_not_explicit",
                handler_role=self.handler_role,
                missing_inputs=["explicit_logic_equivalence_operation"],
            )
        expr_a, expr_b = self._extract_expressions(text)
        missing = []
        if not expr_a:
            missing.append("expression_a")
        if not expr_b:
            missing.append("expression_b")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.95 if not missing else 0.25,
            reason="logic_expression_pair_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        text = handler_input.combined_text()
        if not self._has_logic_operation(text):
            return {"expression_a": "", "expression_b": ""}
        expr_a, expr_b = self._extract_expressions(text)
        return {"expression_a": expr_a, "expression_b": expr_b}

    @staticmethod
    def _has_logic_operation(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:logically\s+equivalent|logic(?:al)?\s+equivalence|truth\s+table|propositional\s+logic)\b",
                text or "",
                flags=re.IGNORECASE,
            )
        )

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        expr_a = str(inputs.get("expression_a") or "").strip()
        expr_b = str(inputs.get("expression_b") or "").strip()
        if not expr_a or not expr_b:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=[
                    item for item, value in (("expression_a", expr_a), ("expression_b", expr_b)) if not value
                ],
                next_action_hint="Provide two explicit propositional expressions.",
            )
        try:
            variables = sorted(set(self._variables(expr_a)) | set(self._variables(expr_b)))
            if not variables or len(variables) > 8:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["small_variable_set"],
                    structured_result={"variables": variables},
                    next_action_hint="Use at most 8 propositional variables.",
                )
            mismatches = []
            for values in itertools.product([False, True], repeat=len(variables)):
                env = dict(zip(variables, values))
                left = self._eval(expr_a, env)
                right = self._eval(expr_b, env)
                if left != right:
                    mismatches.append({"assignment": env, "left": left, "right": right})
            equivalent = not mismatches
        except Exception as exc:
            return HandlerResult.error_result(handler_name=self.name, error=str(exc))

        answer = "yes" if equivalent else "no"
        structured = {
            "task_type": "logic_equivalence",
            "expression_a": expr_a,
            "expression_b": expr_b,
            "variables": variables,
            "mismatch_count": len(mismatches),
            "sample_mismatches": mismatches[:3],
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Expression A: {expr_a}\n"
                f"Expression B: {expr_b}\n"
                f"Truth-table equivalent: {equivalent}\n"
                f"Answer: {answer}\n"
                "Instruction: use this result only for the stated logical equivalence question."
            ),
            structured_result=structured,
            confidence=0.97,
            output_type="final_answer",
            semantic_role="logic_equivalence_answer",
            supporting_inputs=[expr_a, expr_b],
        )

    def _extract_expressions(self, text: str) -> tuple[str, str]:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', text or "")
        values = [next(part for part in match if part).strip() for match in quoted if any(match)]
        if len(values) >= 2:
            return values[0], values[1]
        match = re.search(
            r"(?:whether|if|are|is)\s+(.+?)\s+(?:and|equivalent to)\s+(.+?)\s+(?:logically\s+)?equivalent",
            text or "",
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" ?.,"), match.group(2).strip(" ?.,")
        return "", ""

    def _variables(self, expression: str) -> list[str]:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression)
        keywords = {"and", "or", "not", "xor", "true", "false", "implies", "iff"}
        return [token for token in tokens if token.lower() not in keywords]

    def _eval(self, expression: str, env: dict[str, bool]) -> bool:
        normalized = expression
        normalized = re.sub(r"\bAND\b|&&|∧", " and ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bOR\b|\|\||∨", " or ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bNOT\b|!|¬", " not ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bXOR\b|⊕", " != ", normalized, flags=re.IGNORECASE)
        normalized = normalized.replace("<->", " == ")
        normalized = normalized.replace("->", " <= ")
        for variable, value in sorted(env.items(), key=lambda item: len(item[0]), reverse=True):
            normalized = re.sub(rf"\b{re.escape(variable)}\b", str(bool(value)), normalized)
        normalized = re.sub(r"\bTRUE\b", "True", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bFALSE\b", "False", normalized, flags=re.IGNORECASE)
        if re.search(r"[^A-Za-z0-9_()\s=!<>]", normalized):
            raise ValueError("unsupported logical expression syntax")
        return bool(eval(normalized, {"__builtins__": {}}, {}))


__all__ = ["LogicEquivalenceRouterHandler"]
