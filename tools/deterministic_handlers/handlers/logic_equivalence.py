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
        "Check whether small propositional-logic expressions are equivalent, or identify "
        "the unique non-equivalent expression, by exhaustive truth-table evaluation."
    )
    supported_attachment_types: set[str] = {".txt"}
    supported_task_roles: set[str] = {"logic_equivalence"}
    supported_answer_roles: set[str] = {"boolean", "yes_no", "text", "choice"}
    input_schema = io_contract(
        name,
        [
            input_field(
                "mode",
                "str",
                True,
                "Either pair_comparison or outlier_detection.",
                "question|attachment",
            ),
            input_field("expression_a", "str", False, "First logical expression.", "question|attachment"),
            input_field("expression_b", "str", False, "Second logical expression.", "question|attachment"),
            input_field(
                "expressions",
                "list[str]",
                False,
                "Expression set used for outlier detection.",
                "question|attachment",
            ),
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
        expressions = self._extract_expression_list(text)
        if len(expressions) >= 3 and self._asks_for_outlier(text):
            return HandlerMatch(
                handler_name=self.name,
                matched=True,
                confidence=0.98,
                reason="logic_expression_set_readiness",
                handler_role=self.handler_role,
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
            return {
                "mode": "",
                "expression_a": "",
                "expression_b": "",
                "expressions": [],
            }
        expressions = self._extract_expression_list(text)
        if len(expressions) >= 3 and self._asks_for_outlier(text):
            return {
                "mode": "outlier_detection",
                "expression_a": "",
                "expression_b": "",
                "expressions": expressions,
            }
        expr_a, expr_b = self._extract_expressions(text)
        return {
            "mode": "pair_comparison",
            "expression_a": expr_a,
            "expression_b": expr_b,
            "expressions": [],
        }

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
        expressions = [
            str(value or "").strip()
            for value in list(inputs.get("expressions") or [])
            if str(value or "").strip()
        ]
        if len(expressions) >= 3:
            return self._run_outlier(expressions)
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

    def _run_outlier(self, expressions: list[str]) -> HandlerResult:
        try:
            variables = sorted(
                {
                    variable
                    for expression in expressions
                    for variable in self._variables(expression)
                }
            )
            if not variables or len(variables) > 8:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["small_variable_set"],
                    structured_result={"variables": variables},
                    next_action_hint="Use at most 8 propositional variables.",
                )
            assignments = [
                dict(zip(variables, values))
                for values in itertools.product([False, True], repeat=len(variables))
            ]
            signatures: dict[tuple[bool, ...], list[int]] = {}
            for index, expression in enumerate(expressions):
                signature = tuple(self._eval(expression, env) for env in assignments)
                signatures.setdefault(signature, []).append(index)
            singleton_groups = [indices for indices in signatures.values() if len(indices) == 1]
            repeated_groups = [indices for indices in signatures.values() if len(indices) >= 2]
            if len(singleton_groups) != 1 or not repeated_groups:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["unique_logic_outlier"],
                    structured_result={
                        "variables": variables,
                        "equivalence_groups": [
                            [expressions[index] for index in indices]
                            for indices in signatures.values()
                        ],
                    },
                    next_action_hint="The expression set does not contain one unique truth-table outlier.",
                )
            outlier_index = singleton_groups[0][0]
            answer = expressions[outlier_index]
        except Exception as exc:
            return HandlerResult.error_result(handler_name=self.name, error=str(exc))

        structured = {
            "task_type": "logic_equivalence_outlier",
            "operation": "logic_equivalence",
            "expressions": expressions,
            "variables": variables,
            "outlier_index": outlier_index,
            "equivalence_groups": [
                [expressions[index] for index in indices]
                for indices in signatures.values()
            ],
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Expressions checked: {len(expressions)}\n"
                f"Truth-table outlier: {answer}\n"
                "Instruction: use the complete outlier expression as the final answer."
            ),
            structured_result=structured,
            confidence=0.99,
            output_type="final_answer",
            semantic_role="logic_equivalence_outlier_answer",
            supporting_inputs=expressions,
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

    @staticmethod
    def _asks_for_outlier(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:not\s+logically\s+equivalent|does(?:n't|\s+not)\s+fit|"
                r"does(?:n't|\s+not)\s+belong|odd\s+one\s+out)\b",
                text or "",
                flags=re.IGNORECASE,
            )
        )

    def _extract_expression_list(self, text: str) -> list[str]:
        result: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
            if not line or not self._looks_like_expression(line):
                continue
            if line not in result:
                result.append(line)
        return result

    @staticmethod
    def _looks_like_expression(text: str) -> bool:
        has_operator = bool(
            re.search(r"(?:<->|->|↔|→|¬|∧|∨|\b(?:and|or|not|implies|iff)\b)", text)
        )
        variables = set(re.findall(r"\b[A-Z]\b", text))
        return has_operator and bool(variables)

    def _variables(self, expression: str) -> list[str]:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression)
        keywords = {"and", "or", "not", "xor", "true", "false", "implies", "iff"}
        return [token for token in tokens if token.lower() not in keywords]

    def _eval(self, expression: str, env: dict[str, bool]) -> bool:
        tokens = self._logic_tokens(expression)
        position = 0

        def peek() -> str:
            return tokens[position] if position < len(tokens) else ""

        def consume(expected: str = "") -> str:
            nonlocal position
            token = peek()
            if not token or (expected and token != expected):
                raise ValueError(f"expected {expected or 'logical token'}")
            position += 1
            return token

        def parse_atom() -> bool:
            token = peek()
            if token == "(":
                consume("(")
                value = parse_iff()
                consume(")")
                return value
            consume()
            if token == "TRUE":
                return True
            if token == "FALSE":
                return False
            if token not in env:
                raise ValueError(f"unknown propositional variable: {token}")
            return bool(env[token])

        def parse_not() -> bool:
            if peek() == "NOT":
                consume("NOT")
                return not parse_not()
            return parse_atom()

        def parse_and() -> bool:
            value = parse_not()
            while peek() == "AND":
                consume("AND")
                right = parse_not()
                value = value and right
            return value

        def parse_xor() -> bool:
            value = parse_and()
            while peek() == "XOR":
                consume("XOR")
                value = value != parse_and()
            return value

        def parse_or() -> bool:
            value = parse_xor()
            while peek() == "OR":
                consume("OR")
                right = parse_xor()
                value = value or right
            return value

        def parse_implies() -> bool:
            value = parse_or()
            if peek() == "IMPLIES":
                consume("IMPLIES")
                right = parse_implies()
                return (not value) or right
            return value

        def parse_iff() -> bool:
            value = parse_implies()
            while peek() == "IFF":
                consume("IFF")
                value = value == parse_implies()
            return value

        result = parse_iff()
        if position != len(tokens):
            raise ValueError("unexpected trailing logical tokens")
        return result

    @staticmethod
    def _logic_tokens(expression: str) -> list[str]:
        pattern = re.compile(
            r"\s*(<->|↔|->|→|&&|\|\||¬|!|∧|∨|⊕|\(|\)|"
            r"\bAND\b|\bOR\b|\bNOT\b|\bXOR\b|\bIMPLIES\b|\bIFF\b|"
            r"\bTRUE\b|\bFALSE\b|[A-Za-z][A-Za-z0-9_]*)",
            flags=re.IGNORECASE,
        )
        raw_tokens = pattern.findall(expression or "")
        residue = pattern.sub("", expression or "").strip()
        if residue:
            raise ValueError(f"unsupported logical expression syntax: {residue}")
        aliases = {
            "<->": "IFF",
            "↔": "IFF",
            "IFF": "IFF",
            "->": "IMPLIES",
            "→": "IMPLIES",
            "IMPLIES": "IMPLIES",
            "&&": "AND",
            "∧": "AND",
            "AND": "AND",
            "||": "OR",
            "∨": "OR",
            "OR": "OR",
            "¬": "NOT",
            "!": "NOT",
            "NOT": "NOT",
            "⊕": "XOR",
            "XOR": "XOR",
            "TRUE": "TRUE",
            "FALSE": "FALSE",
        }
        return [aliases.get(token.upper(), aliases.get(token, token)) for token in raw_tokens]


__all__ = ["LogicEquivalenceRouterHandler"]
