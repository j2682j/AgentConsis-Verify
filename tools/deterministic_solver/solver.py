from __future__ import annotations

import re
from typing import Any

from .handlers import ListHandler, MathHandler, StringHandler, TableHandler, UnitHandler
from .schemas import DeterministicReadiness, DeterministicSolverResult


class DeterministicSolver:
    """DeterministicSolver 類別，封裝此模組的資料結構與服務邏輯。"""

    EXTERNAL_FACTUAL_RE = re.compile(
        r"\b("
        r"who|where|which|when|website|paper|book|video|youtube|article|journal|"
        r"company|institution|university|wikipedia|according to|as of|between \d{4}|"
        r"published|authored|nominated|recipient|nationality|country|city|species|"
        r"album|changelog|trial|olympics|base|bbc|cornell|nih|nasa"
        r")\b",
        re.IGNORECASE,
    )
    DETERMINISTIC_SIGNAL_RE = re.compile(
        r"\b("
        r"calculate|compute|evaluate|convert|sort|order|reverse|uppercase|lowercase|"
        r"title case|remove spaces|character count|word count|table|spreadsheet|cell|"
        r"row|column|csv|median|average|mean|sum|total|maximum|minimum|max|min|"
        r"how many characters|how many words"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        """
        ??????????????
        
        Args:
            - ????????????
        
        Returns:
            - None?
        """
        self.handlers = [
            TableHandler(),
            UnitHandler(),
            MathHandler(),
            ListHandler(),
            StringHandler(),
        ]

    def solve(
        self,
        question: str,
        *,
        attachment_context: str | None = None,
        table_data: Any = None,
        best_verified_candidate: dict[str, Any] | None = None,
    ) -> DeterministicSolverResult:
        """
        ??? deterministic ?????????
        
        Args:
            - ????????????
        
        Returns:
            - DeterministicSolverResult ????????
        """
        readiness = self._assess_readiness(
            question,
            attachment_context=attachment_context,
            table_data=table_data,
            best_verified_candidate=best_verified_candidate,
        )
        if not (readiness.is_deterministic_task and readiness.is_closed_world and readiness.has_complete_data):
            return DeterministicSolverResult(
                used_deterministic_solver=False,
                task_type="not_ready",
                confidence=0.0,
                evidence_source=readiness.evidence_source,
                readiness=readiness,
                error=readiness.reason,
            )

        verified_answer = self._verified_answer(best_verified_candidate)
        if verified_answer:
            return DeterministicSolverResult(
                used_deterministic_solver=True,
                task_type="search_verified_answer",
                answer=verified_answer,
                answer_text=verified_answer,
                confidence=0.9,
                evidence={"best_verified_candidate": best_verified_candidate or {}},
                evidence_source="search_verified",
                readiness=readiness,
            )

        for handler in self.handlers:
            result = handler.solve(
                question,
                attachment_context=attachment_context,
                table_data=table_data,
            )
            if result.used_deterministic_solver:
                result.evidence_source = readiness.evidence_source
                result.readiness = readiness
                return result
        return DeterministicSolverResult.miss("unsupported", "no deterministic handler matched")

    def _assess_readiness(
        self,
        question: str,
        *,
        attachment_context: str | None,
        table_data: Any,
        best_verified_candidate: dict[str, Any] | None,
    ) -> DeterministicReadiness:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        text = str(question or "")
        has_verified = bool(self._verified_answer(best_verified_candidate))
        has_table_data = bool(table_data)
        has_attachment_structured = self._has_structured_attachment(attachment_context)
        has_question_table = "|" in text or ("\n" in text and "," in text)
        has_quoted_string = bool(re.search(r'"[^"]+"|\'[^\']+\'', text))
        has_local_list = self._has_local_list(text)
        has_math_expression = bool(
            re.search(r"\d+\s*[%+\-*/]\s*\d+|\d+\s*%\s+of\s+\d+", text)
            or re.search(r"\b(sum|total|average|mean|median|largest|smallest|maximum|minimum)\b", text, re.IGNORECASE)
        )
        has_unit_conversion = bool(re.search(r"\bconvert\b.+\bto\b", text, re.IGNORECASE))
        has_deterministic_signal = bool(self.DETERMINISTIC_SIGNAL_RE.search(text))
        has_external_signal = bool(self.EXTERNAL_FACTUAL_RE.search(text))

        is_deterministic = any(
            [
                has_deterministic_signal,
                has_question_table,
                has_quoted_string,
                has_local_list,
                has_math_expression,
                has_unit_conversion,
                has_table_data,
                has_attachment_structured,
                has_verified,
            ]
        )

        if has_verified:
            return DeterministicReadiness(True, True, True, "search_verified", "search provided a verified candidate")
        if has_table_data:
            return DeterministicReadiness(True, True, True, "attachment", "attachment provided structured table_data")
        if has_attachment_structured:
            return DeterministicReadiness(True, True, True, "attachment", "attachment context contains structured extracted data")
        if has_external_signal:
            return DeterministicReadiness(
                is_deterministic,
                False,
                False,
                "none",
                "question appears to require external factual evidence and no verified search evidence is available",
            )

        has_complete_question_data = any(
            [has_question_table, has_quoted_string, has_local_list, has_math_expression, has_unit_conversion]
        )
        return DeterministicReadiness(
            is_deterministic,
            has_complete_question_data,
            has_complete_question_data,
            "question" if has_complete_question_data else "none",
            "question contains complete deterministic input data"
            if has_complete_question_data
            else "question does not contain complete deterministic input data",
        )

    def _verified_answer(self, best_verified_candidate: dict[str, Any] | None) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if not isinstance(best_verified_candidate, dict):
            return ""
        answer = str(best_verified_candidate.get("answer", "") or "").strip()
        try:
            score = float(best_verified_candidate.get("verification_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return answer if answer and score >= 0.7 else ""

    def _has_structured_attachment(self, attachment_context: str | None) -> bool:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        text = str(attachment_context or "")
        if not text.strip():
            return False
        markers = (
            "Direct answer:",
            "Answer text:",
            "Extracted answer:",
            "Workbook sheets:",
            "Sheet:",
            "CSV",
            "Rows:",
            "Columns:",
            "Transcript:",
        )
        return any(marker in text for marker in markers)

    def _has_local_list(self, text: str) -> bool:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        lowered = text.lower()
        if not any(term in lowered for term in ("sort", "order", "list", "items", "numbers")):
            return False
        tail = text.split(":", 1)[1] if ":" in text else text
        return tail.count(",") >= 1 or tail.count(";") >= 1
