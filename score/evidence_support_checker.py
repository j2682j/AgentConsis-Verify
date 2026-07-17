from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable

from core.config import (
    AgentEvidenceSupportSummary,
    AgentReasoningSummary,
    StepSupportResult,
    ToolEvidenceRecord,
)
from score.answer_validator import AnswerValidator
from utils.network_utils import normalize_for_exact


class EvidenceSupportChecker:
    """
    檢查 Agent 推理與最終答案是否使用可信的工具或附件結果。

    Args:
     - answer_validator: 清理與驗證候選答案的 AnswerValidator。

    Returns:
     - EvidenceSupportChecker: 產生 step-level 與 Agent-level 支持分類的檢查器。
    """

    SUPPORT_PRIORITY = {
        "contradicted": -1,
        "tool_failed_model_only": 0,
        "no_support": 1,
        "search_evidence_supported": 3,
        "attachment_evidence_supported": 3,
        "tool_intermediate_supported": 4,
        "tool_final_supported": 5,
    }
    _IGNORED_TOOLS = {
        "tool_planner",
        "attachment_strategy_planner",
        "attachment_strategy_reviewer",
    }
    _DETERMINISTIC_TOOLS = {
        "deterministic_handler_router",
        "attachment_strategy_handler",
        "deterministic_solver",
        "python_calculator",
    }

    def __init__(self, answer_validator: AnswerValidator | None = None) -> None:
        self.answer_validator = answer_validator or AnswerValidator()

    def check_agent(
        self,
        *,
        target: AgentReasoningSummary,
        reasoning_steps: list[tuple[int, str]],
        evidence: dict[str, Any] | None = None,
    ) -> AgentEvidenceSupportSummary:
        """
        彙整指定 Agent 可使用的工具紀錄並判斷其證據支持狀態。

        Args:
         - target: Stage1 聚合後的 Agent reasoning summary。
         - reasoning_steps: ReasoningParser 已切分的推理步驟。
         - evidence: Evidence Prepare 產生的 evidence bundle。

        Returns:
         - AgentEvidenceSupportSummary: Agent 與每個步驟的支持分類。
        """
        records = self.collect_records(target=target, evidence=evidence or {})
        final_answer = self.answer_validator.clean(target.compressed_answer)
        trusted_finals = self._unique(
            record.value
            for record in records
            if (
                record.output_type == "final_answer"
                and record.trusted
                and record.evidence_valid
                and record.value
            )
        )
        trusted_final_conflict = self._has_conflicting_values(trusted_finals)
        matched_final_values = [
            value for value in trusted_finals if self._answers_equivalent(final_answer, value)
        ]

        step_results = [
            self._check_step(
                step_index=step_index,
                step_text=step_text,
                records=records,
                final_answer=final_answer,
                trusted_finals=trusted_finals,
                trusted_final_conflict=trusted_final_conflict,
            )
            for step_index, step_text in reasoning_steps
        ]
        status = self._agent_status(
            final_answer=final_answer,
            records=records,
            step_results=step_results,
            trusted_finals=trusted_finals,
            matched_final_values=matched_final_values,
            trusted_final_conflict=trusted_final_conflict,
        )
        failures = [record for record in records if record.output_type == "failed"]
        return AgentEvidenceSupportSummary(
            agent_id=target.agent_id,
            status=status,
            priority=self.SUPPORT_PRIORITY[status],
            step_results=step_results,
            evidence_records=records,
            matched_final_values=matched_final_values,
            trusted_final_answers=trusted_finals,
            tool_failure_count=len(failures),
            metadata={
                "trusted_final_conflict": trusted_final_conflict,
                "record_count": len(records),
                "supported_step_count": sum(
                    1 for item in step_results if item.status == "supported"
                ),
                "contradicted_step_count": sum(
                    1 for item in step_results if item.status == "contradicted"
                ),
            },
        )

    def collect_records(
        self,
        *,
        target: AgentReasoningSummary,
        evidence: dict[str, Any],
    ) -> list[ToolEvidenceRecord]:
        """
        將 Evidence Prepare 與目標 Agent 的工具結果正規化並去重。

        Args:
         - target: 目前要驗證的 Agent summary。
         - evidence: Evidence Prepare bundle。

        Returns:
         - list[ToolEvidenceRecord]: 僅包含附件、deterministic 與 Stage1 工具結果。
        """
        records: list[ToolEvidenceRecord] = []
        for item in evidence.get("tool_usage", []) or []:
            if isinstance(item, dict) and item.get("tool_name") == "search":
                records.extend(
                    self._search_evidence_records(
                        item,
                        source_scope="evidence_prepare",
                    )
                )
                continue
            record = self._normalize_record(item, source_scope="evidence_prepare")
            if record is not None:
                records.append(record)

        selected_runs = self._selected_runs(target)
        for run in selected_runs:
            for item in run.tool_results or []:
                if isinstance(item, dict) and item.get("tool_name") == "search":
                    records.extend(
                        self._search_evidence_records(
                            item,
                            source_scope="stage1_tool_use",
                            agent_id=target.agent_id,
                            run_index=run.run_index,
                        )
                    )
                    continue
                record = self._normalize_record(
                    item,
                    source_scope="stage1_tool_use",
                    agent_id=target.agent_id,
                    run_index=run.run_index,
                )
                if record is not None:
                    records.append(record)

        return self._deduplicate_records(records)

    def _search_evidence_records(
        self,
        item: dict[str, Any],
        *,
        source_scope: str,
        agent_id: str = "",
        run_index: int = 0,
    ) -> list[ToolEvidenceRecord]:
        """
        將 Search pipeline 最終選出的 evidence items 轉成可追溯的支持紀錄。

        Args:
         - item: SearchTool 或 EvidenceRunner 保存的工具結果。
         - source_scope: evidence_prepare 或 stage1_tool_use。

        Returns:
         - list[ToolEvidenceRecord]: 僅包含已轉成 evidence 的搜尋內容。
        """
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        evidence_items = raw.get("evidence_items") if isinstance(raw, dict) else []
        records: list[ToolEvidenceRecord] = []
        for evidence_item in evidence_items or []:
            if not isinstance(evidence_item, dict):
                continue
            text = str(evidence_item.get("text") or "").strip()
            if not text:
                continue
            records.append(
                ToolEvidenceRecord(
                    tool_name="search",
                    output_type="evidence_text",
                    trusted=False,
                    evidence_valid=True,
                    source_scope=source_scope,
                    agent_id=agent_id,
                    run_index=int(run_index or 0),
                    status="selected_evidence",
                    evidence_text=text,
                    metadata={
                        "evidence_id": str(evidence_item.get("evidence_id") or ""),
                        "source_id": str(evidence_item.get("source_id") or ""),
                        "source_title": str(evidence_item.get("title") or ""),
                        "evidence_bucket": str(evidence_item.get("evidence_bucket") or ""),
                        "sequence_tag": str(evidence_item.get("sequence_tag") or ""),
                        "url": str(evidence_item.get("url") or ""),
                    },
                )
            )
            useful_spans: list[str] = []
            for field_name in (
                "compatible_spans",
                "answer_support_spans",
                "bridge_spans",
                "matched_terms",
            ):
                useful_spans.extend(self._string_list(evidence_item.get(field_name)))
            for span in self._unique(useful_spans):
                records.append(
                    ToolEvidenceRecord(
                        tool_name="search",
                        output_type="intermediate_value",
                        value=span,
                        role="selected_evidence_span",
                        trusted=False,
                        evidence_valid=True,
                        source_scope=source_scope,
                        agent_id=agent_id,
                        run_index=int(run_index or 0),
                        status="selected_evidence",
                        evidence_text=text,
                        metadata={
                            "evidence_id": str(evidence_item.get("evidence_id") or ""),
                            "source_id": str(evidence_item.get("source_id") or ""),
                            "source_title": str(evidence_item.get("title") or ""),
                            "evidence_bucket": str(evidence_item.get("evidence_bucket") or ""),
                        },
                    )
                )
        return records

    def _selected_runs(self, target: AgentReasoningSummary) -> list[Any]:
        target_key = self._answer_key(target.compressed_answer)
        selected = [
            run
            for run in target.runs
            if target_key and self._answer_key(run.final_answer) == target_key
        ]
        return selected or list(target.runs)

    def _normalize_record(
        self,
        item: Any,
        *,
        source_scope: str,
        agent_id: str = "",
        run_index: int = 0,
    ) -> ToolEvidenceRecord | None:
        if not isinstance(item, dict):
            return None
        tool_name = str(item.get("tool_name") or "").strip()
        if not tool_name or tool_name in self._IGNORED_TOOLS:
            return None

        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        trust = item.get("handler_trust") if isinstance(item.get("handler_trust"), dict) else {}
        status = str(item.get("status") or raw.get("status") or "").strip()
        error = str(
            item.get("error")
            or item.get("error_message")
            or raw.get("error")
            or ""
        ).strip()
        missing_inputs = self._string_list(
            item.get("missing_inputs") or raw.get("missing_inputs")
        )
        ok = bool(item.get("ok", raw.get("ok", False)))
        evidence_valid = bool(item.get("evidence_valid", raw.get("evidence_valid", False)))
        trusted = bool(trust.get("trusted", ok and evidence_valid))

        output_type = str(
            item.get("output_type")
            or trust.get("output_type")
            or raw.get("output_type")
            or ""
        ).strip()
        if error or missing_inputs or status in {
            "error",
            "fatal",
            "missing_inputs",
            "retryable_failure",
            "unsupported",
        }:
            output_type = "failed"
        elif not output_type and tool_name == "deterministic_solver":
            output_type = (
                "final_answer"
                if raw.get("used_deterministic_solver") and self._candidate_value(item, raw, trust)
                else "failed"
            )
        elif not output_type and tool_name == "python_calculator":
            output_type = "intermediate_value"
        elif not output_type:
            output_type = "evidence_text"

        value = self._candidate_value(item, raw, trust)
        evidence_text = str(
            item.get("output_text")
            or trust.get("evidence_text")
            or raw.get("context")
            or raw.get("output_text")
            or ""
        ).strip()
        if output_type in {"final_answer", "intermediate_value"} and not value:
            value = self._short_value(evidence_text)
        role = str(
            item.get("semantic_role")
            or trust.get("semantic_role")
            or raw.get("semantic_role")
            or ""
        ).strip()

        if output_type == "failed":
            trusted = False
            evidence_valid = False
        elif tool_name in self._DETERMINISTIC_TOOLS and output_type == "final_answer":
            trusted = bool(trusted and value)

        return ToolEvidenceRecord(
            tool_name=tool_name,
            output_type=output_type,
            value=value,
            role=role,
            trusted=trusted,
            evidence_valid=evidence_valid,
            source_scope=source_scope,
            agent_id=agent_id,
            run_index=run_index,
            status=status,
            evidence_text=evidence_text,
            missing_inputs=missing_inputs,
            next_action_hint=str(
                item.get("next_action_hint")
                or item.get("retry_hint")
                or raw.get("next_action_hint")
                or ""
            ).strip(),
            error=error,
            metadata={
                "handler_name": str(item.get("handler_name") or raw.get("handler_name") or ""),
                "cache_hit": bool(item.get("cache_hit")),
            },
        )

    def _candidate_value(
        self,
        item: dict[str, Any],
        raw: dict[str, Any],
        trust: dict[str, Any],
    ) -> str:
        containers = [trust, raw, item]
        nested_evidence = raw.get("evidence")
        if isinstance(nested_evidence, dict):
            containers.append(nested_evidence)
        for container in containers:
            for key in (
                "answer",
                "answer_text",
                "final_answer",
                "candidate_answer",
                "final_answer_candidate",
                "value",
            ):
                value = container.get(key)
                if value is not None and str(value).strip():
                    return self.answer_validator.clean(value)
        return ""

    def _check_step(
        self,
        *,
        step_index: int,
        step_text: str,
        records: list[ToolEvidenceRecord],
        final_answer: str,
        trusted_finals: list[str],
        trusted_final_conflict: bool,
    ) -> StepSupportResult:
        matched = [
            record
            for record in records
            if (
                record.output_type in {"final_answer", "intermediate_value"}
                and record.value
                and record.evidence_valid
                and self._value_mentioned(record.value, step_text)
            )
        ]
        if matched:
            return StepSupportResult(
                step_index=step_index,
                step_text=step_text,
                status="supported",
                matched_tool_values=self._unique(record.value for record in matched),
                source_tools=self._unique(record.tool_name for record in matched),
                reason="reasoning_step_uses_tool_value",
            )

        grounded_text_records = [
            record
            for record in records
            if (
                record.output_type == "evidence_text"
                and record.evidence_valid
                and final_answer
                and self._answer_in_evidence(final_answer, record.evidence_text)
                and self._value_mentioned(final_answer, step_text)
            )
        ]
        if grounded_text_records:
            return StepSupportResult(
                step_index=step_index,
                step_text=step_text,
                status="supported",
                matched_tool_values=[final_answer],
                source_tools=self._unique(
                    record.tool_name for record in grounded_text_records
                ),
                reason="final_answer_grounded_in_selected_evidence",
            )

        if (
            trusted_finals
            and not trusted_final_conflict
            and final_answer
            and not any(self._answers_equivalent(final_answer, value) for value in trusted_finals)
            and self._value_mentioned(final_answer, step_text)
        ):
            return StepSupportResult(
                step_index=step_index,
                step_text=step_text,
                status="contradicted",
                matched_tool_values=list(trusted_finals),
                source_tools=self._unique(
                    record.tool_name
                    for record in records
                    if record.output_type == "final_answer" and record.trusted
                ),
                reason="step_asserts_final_answer_that_conflicts_with_trusted_tool",
            )

        failures = [record for record in records if record.output_type == "failed"]
        if failures and not any(
            record.evidence_valid and record.output_type != "failed" for record in records
        ):
            return StepSupportResult(
                step_index=step_index,
                step_text=step_text,
                status="tool_failed",
                source_tools=self._unique(record.tool_name for record in failures),
                reason="tools_failed_before_reasoning",
            )
        return StepSupportResult(
            step_index=step_index,
            step_text=step_text,
            status="unsupported",
            reason="no_tool_value_grounded_in_step",
        )

    def _agent_status(
        self,
        *,
        final_answer: str,
        records: list[ToolEvidenceRecord],
        step_results: list[StepSupportResult],
        trusted_finals: list[str],
        matched_final_values: list[str],
        trusted_final_conflict: bool,
    ) -> str:
        if matched_final_values:
            return "tool_final_supported"
        if (
            trusted_finals
            and not trusted_final_conflict
            and final_answer
            and not any(self._answers_equivalent(final_answer, value) for value in trusted_finals)
        ):
            return "contradicted"
        supporting_text_records = [
            record
            for record in records
            if (
                record.output_type == "evidence_text"
                and record.evidence_valid
                and self._answer_in_evidence(final_answer, record.evidence_text)
            )
        ]
        if supporting_text_records:
            if any(record.tool_name == "search" for record in supporting_text_records):
                return "search_evidence_supported"
            return "attachment_evidence_supported"
        if any(
            item.status == "supported"
            and any(
                record.output_type == "intermediate_value"
                and record.value in item.matched_tool_values
                for record in records
            )
            for item in step_results
        ):
            return "tool_intermediate_supported"
        valid_records = [
            record
            for record in records
            if record.output_type != "failed" and record.evidence_valid
        ]
        if records and not valid_records and any(
            record.output_type == "failed" for record in records
        ):
            return "tool_failed_model_only"
        return "no_support"

    def _answer_in_evidence(self, answer: str, evidence_text: str) -> bool:
        answer_key = self._answer_key(answer)
        evidence_key = self._answer_key(evidence_text)
        return bool(answer_key and evidence_key and self._contains_value(evidence_key, answer_key))

    def _value_mentioned(self, value: str, text: str) -> bool:
        value_key = self._answer_key(value)
        text_key = self._answer_key(text)
        return bool(value_key and text_key and self._contains_value(text_key, value_key))

    def _contains_value(self, text_key: str, value_key: str) -> bool:
        if len(value_key) == 1 and value_key.isalpha():
            return bool(re.search(rf"\b{re.escape(value_key)}\b", text_key))
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value_key):
            return bool(
                re.search(
                    rf"(?<![\d.]){re.escape(value_key)}(?!\d)(?!\.\d)",
                    text_key,
                )
            )
        return value_key in text_key

    def _answers_equivalent(self, left: str, right: str) -> bool:
        left_key = self._answer_key(left)
        right_key = self._answer_key(right)
        if not left_key or not right_key:
            return False
        if left_key == right_key:
            return True
        left_number = self._decimal_value(left_key)
        right_number = self._decimal_value(right_key)
        return (
            left_number is not None
            and right_number is not None
            and left_number == right_number
        )

    def _has_conflicting_values(self, values: list[str]) -> bool:
        if len(values) < 2:
            return False
        first = values[0]
        return any(not self._answers_equivalent(first, value) for value in values[1:])

    def _answer_key(self, value: Any) -> str:
        cleaned = self.answer_validator.clean(value)
        normalized = normalize_for_exact(cleaned)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _decimal_value(self, value: str) -> Decimal | None:
        compact = value.replace(",", "").strip()
        match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact)
        if not match:
            return None
        try:
            return Decimal(compact)
        except InvalidOperation:
            return None

    def _short_value(self, text: str) -> str:
        compact = " ".join(str(text or "").split())
        if 0 < len(compact) <= 160:
            return self.answer_validator.clean(compact)
        return ""

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _deduplicate_records(
        self,
        records: list[ToolEvidenceRecord],
    ) -> list[ToolEvidenceRecord]:
        seen: set[tuple[Any, ...]] = set()
        result: list[ToolEvidenceRecord] = []
        for record in records:
            key = (
                record.tool_name,
                record.output_type,
                self._answer_key(record.value),
                record.source_scope,
                record.agent_id,
                record.run_index,
                record.status,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(record)
        return result

    def _unique(self, values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = self._answer_key(text)
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    @staticmethod
    def summary_to_dict(summary: AgentEvidenceSupportSummary) -> dict[str, Any]:
        return {
            "agent_id": summary.agent_id,
            "status": summary.status,
            "priority": summary.priority,
            "step_results": [asdict(item) for item in summary.step_results],
            "evidence_records": [
                {
                    "tool_name": record.tool_name,
                    "output_type": record.output_type,
                    "value": record.value,
                    "role": record.role,
                    "trusted": record.trusted,
                    "evidence_valid": record.evidence_valid,
                    "source_scope": record.source_scope,
                    "agent_id": record.agent_id,
                    "run_index": record.run_index,
                    "status": record.status,
                    "evidence_preview": record.evidence_text[:300],
                    "missing_inputs": list(record.missing_inputs),
                    "next_action_hint": record.next_action_hint,
                    "error": record.error,
                    "metadata": dict(record.metadata),
                }
                for record in summary.evidence_records
            ],
            "matched_final_values": list(summary.matched_final_values),
            "trusted_final_answers": list(summary.trusted_final_answers),
            "tool_failure_count": summary.tool_failure_count,
            "metadata": dict(summary.metadata),
        }


__all__ = ["EvidenceSupportChecker"]
