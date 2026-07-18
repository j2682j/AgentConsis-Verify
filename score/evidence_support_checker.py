from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
import hashlib
import re
from typing import Any, Iterable

from core.config import (
    AgentEvidenceSupportSummary,
    AgentReasoningSummary,
    StepSupportResult,
    ToolEvidenceRecord,
)
from score.answer_validator import AnswerValidator
from score.evidence_support_context import EvidenceSupportContext
from score.evidence_support_level import support_level_for_status
from score.candidate_fact_verifier import (
    CandidateFactVerification,
    CandidateFactVerifier,
)
from score.numerical_derivation_verifier import (
    NumericalDerivationSummary,
    NumericalDerivationVerifier,
)
from utils.network_utils import normalize_for_exact
from tools.evidence.fact_extraction import (
    FactDerivationEngine,
    EvidenceFact,
    TaskFactCollector,
    TaskFactStore,
)


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
        "tool_failed_model_only": 1,
        "no_support": 1,
        "search_evidence_supported": 3,
        "attachment_evidence_supported": 3,
        "tool_intermediate_supported": 4,
        "derived_evidence_supported": 4,
        "tool_final_supported": 5,
    }
    _IGNORED_TOOLS = {
        "tool_planner",
        "attachment_strategy_planner",
        "attachment_strategy_reviewer",
        "attachment_fact_extractor",
        "semantic_fact_extractor",
    }
    _DETERMINISTIC_TOOLS = {
        "deterministic_handler_router",
        "attachment_strategy_handler",
        "deterministic_solver",
        "python_calculator",
    }

    def __init__(
        self,
        answer_validator: AnswerValidator | None = None,
        numerical_derivation_verifier: NumericalDerivationVerifier | None = None,
        candidate_fact_verifier: CandidateFactVerifier | None = None,
        fact_derivation_engine: FactDerivationEngine | None = None,
        fact_collector: TaskFactCollector | None = None,
    ) -> None:
        self.answer_validator = answer_validator or AnswerValidator()
        self.numerical_derivation_verifier = (
            numerical_derivation_verifier or NumericalDerivationVerifier()
        )
        self.candidate_fact_verifier = candidate_fact_verifier or CandidateFactVerifier(
            equivalence_fn=self._answers_equivalent,
        )
        self.fact_derivation_engine = fact_derivation_engine or FactDerivationEngine()
        self.fact_collector = fact_collector or TaskFactCollector()

    def check_agent(
        self,
        *,
        target: AgentReasoningSummary,
        reasoning_steps: list[tuple[int, str]],
        evidence: dict[str, Any] | None = None,
        question: str = "",
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
        final_answer = self.answer_validator.clean(target.compressed_answer)
        shared_fact_store = self.collect_fact_store(
            target=target,
            evidence=evidence or {},
        )
        answer_requirement = str((evidence or {}).get("answer_requirement") or question).strip()
        fact_derivation = self.fact_derivation_engine.derive(
            shared_fact_store,
            answer_requirement=answer_requirement,
        )
        if evidence is not None:
            evidence["fact_store"] = shared_fact_store.to_dict()
        # Candidate-dependent derivations must not contaminate other candidates.
        fact_store = TaskFactStore.from_dict(shared_fact_store.to_dict())
        fact_verification = self.candidate_fact_verifier.verify(
            candidate_answer=final_answer,
            fact_store=fact_store,
            answer_requirement=answer_requirement,
        )
        records = self.collect_records(target=target, evidence=evidence or {})
        records = self._deduplicate_records(
            [*records, *self._fact_store_records(fact_store)]
        )
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

        numerical_derivation = self.numerical_derivation_verifier.verify(
            question=question,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            records=records,
        )
        numerical_fact = self._numerical_derivation_fact(
            final_answer=final_answer,
            numerical_derivation=numerical_derivation,
            fact_store=fact_store,
            answer_requirement=answer_requirement,
        )
        if numerical_fact is not None and fact_store.add(numerical_fact):
            records = self._deduplicate_records(
                [*records, *self._fact_store_records(fact_store)]
            )
            fact_verification = self.candidate_fact_verifier.verify(
                candidate_answer=final_answer,
                fact_store=fact_store,
                answer_requirement=answer_requirement,
            )
        numerical_by_step = {
            item.step_index: item for item in numerical_derivation.step_results
        }
        step_results = []
        for step_index, step_text in reasoning_steps:
            direct_result = self._check_step(
                step_index=step_index,
                step_text=step_text,
                records=records,
                final_answer=final_answer,
                trusted_finals=trusted_finals,
                trusted_final_conflict=trusted_final_conflict,
            )
            step_results.append(
                self._merge_numerical_support(
                    direct_result,
                    numerical_by_step.get(step_index),
                )
            )
        status = self._agent_status(
            final_answer=final_answer,
            records=records,
            step_results=step_results,
            trusted_finals=trusted_finals,
            matched_final_values=matched_final_values,
            trusted_final_conflict=trusted_final_conflict,
            numerical_derivation=numerical_derivation,
            fact_verification=fact_verification,
            fact_store=fact_store,
        )
        failures = [record for record in records if record.output_type == "failed"]
        return AgentEvidenceSupportSummary(
            agent_id=target.agent_id,
            status=status,
            priority=self.SUPPORT_PRIORITY[status],
            support_level=support_level_for_status(status).value,
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
                "numerical_derivation": numerical_derivation.to_dict(),
                "candidate_fact_verification": fact_verification.to_dict(),
                "fact_derivation": fact_derivation.to_dict(),
                "fact_store": fact_store.to_dict(),
            },
        )

    def prepare_context(
        self,
        *,
        evidence: dict[str, Any] | None,
        question: str = "",
        evidence_revision: int = 0,
    ) -> EvidenceSupportContext:
        """Prepare immutable task evidence before evaluating candidate paths."""

        payload = dict(evidence or {})
        source_store = payload.get("_fact_store")
        if isinstance(source_store, TaskFactStore):
            store = TaskFactStore.from_dict(source_store.to_dict())
        else:
            store = TaskFactStore.from_dict(payload.get("fact_store"))
        self.fact_collector.collect_many(
            store,
            list(payload.get("tool_usage") or []),
            question=question,
            source_scope="evidence_prepare",
        )
        payload.pop("_fact_store", None)
        payload["fact_store"] = store.to_dict()
        routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
        return EvidenceSupportContext(
            base_fact_store=store,
            answer_requirement=str(payload.get("answer_requirement") or question).strip(),
            answer_role=str(payload.get("answer_role") or "").strip(),
            task_route=str(routing.get("primary_route") or "").strip(),
            evidence_revision=int(evidence_revision or 0),
            evidence_payload=payload,
            metadata={
                "base_fact_count": len(store.all()),
                "evidence_revision": int(evidence_revision or 0),
            },
        )

    def check_path(
        self,
        *,
        context: EvidenceSupportContext,
        target: AgentReasoningSummary,
        candidate_answer: str,
        reasoning_steps: list[tuple[int, str]],
        tool_results: list[dict[str, Any]] | None = None,
        question: str = "",
    ) -> AgentEvidenceSupportSummary:
        """Evaluate one path against an isolated clone of the task evidence."""

        selected_runs = list(target.runs)
        if selected_runs and tool_results is not None:
            selected_runs[0] = replace(
                selected_runs[0],
                tool_results=list(tool_results),
            )
        candidate_target = replace(
            target,
            runs=selected_runs,
            compressed_answer=str(candidate_answer or "").strip(),
            run_validity_labels=list(target.run_validity_labels),
            aggregation_metadata=dict(target.aggregation_metadata),
            self_review_metadata=dict(target.self_review_metadata),
        )
        return self.check_agent(
            target=candidate_target,
            reasoning_steps=list(reasoning_steps),
            evidence=context.candidate_evidence(),
            question=question,
        )

    def collect_fact_store(
        self,
        *,
        target: AgentReasoningSummary,
        evidence: dict[str, Any],
    ) -> TaskFactStore:
        """彙整 Evidence Prepare 與 Stage1 Tool Use 產生的任務事實。"""

        store = evidence.get("_fact_store")
        if not isinstance(store, TaskFactStore):
            store = TaskFactStore.from_dict(evidence.get("fact_store"))
        self.fact_collector.collect_many(
            store,
            list(evidence.get("tool_usage") or []),
            question="",
            source_scope="evidence_prepare",
        )
        for run in self._selected_runs(target):
            self.fact_collector.collect_many(
                store,
                list(run.tool_results or []),
                question="",
                source_scope="stage1_tool_use",
            )
        evidence["_fact_store"] = store
        evidence["fact_store"] = store.to_dict()
        return store

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
            records.extend(
                self._semantic_fact_records(
                    item,
                    source_scope="evidence_prepare",
                )
            )
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
                records.extend(
                    self._semantic_fact_records(
                        item,
                        source_scope="stage1_tool_use",
                        agent_id=target.agent_id,
                        run_index=run.run_index,
                    )
                )
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
            direct_contracts = [
                dict(contract)
                for contract in list(evidence_item.get("direct_contracts") or [])
                if isinstance(contract, dict)
            ]
            goal_ids = self._unique(
                str(contract.get("goal_id") or "")
                for contract in direct_contracts
            )
            answer_spans = self._unique(
                str(contract.get("answer_span") or "")
                for contract in direct_contracts
            )
            records.extend(
                self._semantic_fact_records(
                    {
                        "tool_name": "search",
                        "raw_result": {
                            "semantic_facts": list(
                                evidence_item.get("semantic_facts") or []
                            )
                        },
                    },
                    source_scope=source_scope,
                    agent_id=agent_id,
                    run_index=run_index,
                )
            )
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
                        "goal_ids": goal_ids,
                        "answer_spans": answer_spans,
                        "direct_contracts": direct_contracts,
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
            useful_spans.extend(answer_spans)
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
                            "goal_ids": self._unique(
                                str(contract.get("goal_id") or "")
                                for contract in direct_contracts
                                if self._answers_equivalent(
                                    span,
                                    str(contract.get("answer_span") or ""),
                                )
                            ),
                            "answer_spans": [span],
                        },
                    )
                )
        return records

    def _semantic_fact_records(
        self,
        item: Any,
        *,
        source_scope: str,
        agent_id: str = "",
        run_index: int = 0,
    ) -> list[ToolEvidenceRecord]:
        if not isinstance(item, dict):
            return []
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        facts = list(item.get("semantic_facts") or raw.get("semantic_facts") or [])
        tool_name = str(item.get("tool_name") or "semantic_fact_extractor").strip()
        records: list[ToolEvidenceRecord] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("grounding_status") or "").strip().lower() != "grounded":
                continue
            value = str(fact.get("object") or "").strip()
            role = str(fact.get("role") or "CONTEXT").strip().upper()
            evidence_spans = self._string_list(fact.get("evidence_spans"))
            qualifiers = dict(fact.get("qualifiers") or {})
            evidence_text = str(fact.get("context") or "").strip()
            if not evidence_text:
                evidence_text = " ".join(evidence_spans).strip()
            if not value or not evidence_text:
                continue
            records.append(
                ToolEvidenceRecord(
                    tool_name=tool_name,
                    output_type="evidence_fact",
                    value=value,
                    role=role,
                    trusted=False,
                    evidence_valid=True,
                    source_scope=source_scope,
                    agent_id=agent_id,
                    run_index=int(run_index or 0),
                    status="grounded_fact",
                    evidence_text=evidence_text,
                    metadata={
                        "fact_id": str(fact.get("fact_id") or ""),
                        "evidence_id": str(qualifiers.get("evidence_id") or ""),
                        "subject": str(fact.get("subject") or ""),
                        "relation": str(fact.get("relation") or ""),
                        "object": value,
                        "qualifiers": qualifiers,
                        "polarity": str(fact.get("polarity") or "positive"),
                        "goal_ids": self._string_list(fact.get("goal_id")),
                        "source_id": str(fact.get("source_id") or ""),
                        "source_type": str(fact.get("source_type") or ""),
                        "source_title": str(fact.get("source_title") or ""),
                        "grounding_status": "grounded",
                        "extraction_method": str(
                            fact.get("extraction_method") or ""
                        ),
                        "evidence_spans": evidence_spans,
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
                record.output_type in {
                    "final_answer",
                    "intermediate_value",
                    "evidence_fact",
                }
                and record.value
                and record.evidence_valid
                and not (
                    record.output_type == "evidence_fact"
                    and str(record.metadata.get("polarity") or "positive") == "negative"
                )
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
                and self._record_directly_supports_answer(record, final_answer)
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
        numerical_derivation: NumericalDerivationSummary,
        fact_verification: CandidateFactVerification,
        fact_store: TaskFactStore,
    ) -> str:
        if fact_verification.status == "contradicted":
            return "contradicted"
        if fact_verification.status == "supported":
            if fact_verification.support_kind == "derived":
                return "derived_evidence_supported"
            supporting_facts = [
                fact_store.get(fact_id)
                for fact_id in fact_verification.supporting_fact_ids
            ]
            if any(
                fact is not None and fact.source_type in {"search", "web"}
                for fact in supporting_facts
            ):
                return "search_evidence_supported"
            if any(
                fact is not None and fact.source_type in {"handler", "stage1_tool"}
                for fact in supporting_facts
            ):
                return "tool_final_supported"
            return "attachment_evidence_supported"
        if numerical_derivation.status == "contradicted":
            return "contradicted"
        if matched_final_values:
            return "tool_final_supported"
        if numerical_derivation.final_supported:
            return "derived_evidence_supported"
        supporting_fact_records = [
            record
            for record in records
            if (
                record.output_type == "evidence_fact"
                and record.evidence_valid
                and record.role == "ANSWER_SUPPORT"
                and str(record.metadata.get("polarity") or "positive") == "positive"
                and self._answers_equivalent(final_answer, record.value)
            )
        ]
        if supporting_fact_records:
            if any(
                record.tool_name == "search"
                or str(record.metadata.get("source_type") or "") == "web"
                for record in supporting_fact_records
            ):
                return "search_evidence_supported"
            return "attachment_evidence_supported"
        contradicted_fact_records = [
            record
            for record in records
            if (
                record.output_type == "evidence_fact"
                and record.evidence_valid
                and record.role == "ANSWER_SUPPORT"
                and str(record.metadata.get("polarity") or "positive") == "negative"
                and self._answers_equivalent(final_answer, record.value)
            )
        ]
        if contradicted_fact_records:
            return "contradicted"
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
                and self._record_directly_supports_answer(record, final_answer)
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

    def _fact_store_records(
        self,
        fact_store: TaskFactStore,
    ) -> list[ToolEvidenceRecord]:
        records: list[ToolEvidenceRecord] = []
        for fact in fact_store.all():
            if fact.grounding_status != "grounded" or not fact.object:
                continue
            records.append(
                ToolEvidenceRecord(
                    tool_name=fact.source_type or "fact_store",
                    output_type="evidence_fact",
                    value=fact.object,
                    role=fact.role,
                    trusted=False,
                    evidence_valid=True,
                    source_scope="task_fact_store",
                    status=("derived_fact" if fact.parent_fact_ids else "grounded_fact"),
                    evidence_text=fact.context or " ".join(fact.evidence_spans),
                    metadata={
                        "fact_id": fact.fact_id,
                        "evidence_id": fact.qualifiers.get("evidence_id", ""),
                        "subject": fact.subject,
                        "relation": fact.relation,
                        "object": fact.object,
                        "qualifiers": dict(fact.qualifiers),
                        "polarity": fact.polarity,
                        "goal_ids": self._string_list(fact.goal_id),
                        "source_id": fact.source_id,
                        "source_type": fact.source_type,
                        "source_title": fact.source_title,
                        "grounding_status": fact.grounding_status,
                        "extraction_method": fact.extraction_method,
                        "evidence_spans": list(fact.evidence_spans),
                        "parent_fact_ids": list(fact.parent_fact_ids),
                        "derivation_type": fact.derivation_type,
                    },
                )
            )
        return records

    def _numerical_derivation_fact(
        self,
        *,
        final_answer: str,
        numerical_derivation: NumericalDerivationSummary,
        fact_store: TaskFactStore,
        answer_requirement: str,
    ) -> EvidenceFact | None:
        if not numerical_derivation.final_supported or not final_answer:
            return None
        provenance = set(numerical_derivation.provenance_ids)
        parent_fact_ids = [
            fact.fact_id
            for fact in fact_store.all()
            if str(fact.qualifiers.get("evidence_id") or "") in provenance
            or fact.fact_id in provenance
        ]
        if not parent_fact_ids:
            return None
        payload = "\x1f".join([final_answer, *parent_fact_ids])
        fact_id = "derived-numeric-" + hashlib.sha1(
            payload.encode("utf-8")
        ).hexdigest()[:16]
        contexts = [
            fact.context
            for fact_id_value in parent_fact_ids
            if (fact := fact_store.get(fact_id_value)) is not None and fact.context
        ]
        return EvidenceFact(
            fact_id=fact_id,
            subject=answer_requirement or "requested result",
            relation="has verified numerical result",
            object=final_answer,
            qualifiers={
                "terminal_value": numerical_derivation.terminal_value,
                "goal_ids": ",".join(numerical_derivation.goal_ids),
                "answer_binding": "direct",
                "answer_requirement": answer_requirement,
            },
            polarity="positive",
            role="ANSWER_SUPPORT",
            goal_id=(numerical_derivation.goal_ids[0] if numerical_derivation.goal_ids else ""),
            evidence_spans=[final_answer],
            context="\n".join(contexts),
            source_id="numerical_derivation",
            source_type="derived",
            source_title="Verified numerical derivation",
            grounding_status="grounded",
            extraction_method="numerical_derivation_verifier",
            parent_fact_ids=parent_fact_ids,
            derivation_type="numerical_calculation",
        )

    def _record_directly_supports_answer(
        self,
        record: ToolEvidenceRecord,
        final_answer: str,
    ) -> bool:
        """Require an answer-bound evidence contract instead of text occurrence."""
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        contracts = [
            item
            for item in list(metadata.get("direct_contracts") or [])
            if isinstance(item, dict)
        ]
        for contract in contracts:
            answer_span = str(contract.get("answer_span") or "").strip()
            context = str(contract.get("context") or record.evidence_text or "").strip()
            requirement = str(contract.get("answer_requirement") or "").strip()
            if not answer_span or not requirement:
                continue
            if not self._answers_equivalent(final_answer, answer_span):
                continue
            if not self._answer_in_evidence(answer_span, context):
                continue
            if contract.get("relation_resolved") is False:
                continue
            return True
        return False

    def _merge_numerical_support(
        self,
        direct_result: StepSupportResult,
        numerical_result: Any | None,
    ) -> StepSupportResult:
        if numerical_result is None:
            return direct_result
        numerical_metadata = {
            "support_kind": "numerical_derivation",
            "derivation": numerical_result.to_dict(),
        }
        if numerical_result.status == "contradicted":
            return StepSupportResult(
                step_index=direct_result.step_index,
                step_text=direct_result.step_text,
                status="contradicted",
                matched_tool_values=list(numerical_result.matched_values),
                source_tools=list(numerical_result.source_tools),
                reason=numerical_result.reason,
                metadata=numerical_metadata,
            )
        if numerical_result.status == "derived_supported":
            return StepSupportResult(
                step_index=direct_result.step_index,
                step_text=direct_result.step_text,
                status="supported",
                matched_tool_values=self._unique(
                    [
                        *direct_result.matched_tool_values,
                        *numerical_result.matched_values,
                        numerical_result.claimed_value,
                    ]
                ),
                source_tools=self._unique(
                    [*direct_result.source_tools, *numerical_result.source_tools]
                ),
                reason="evidence_grounded_calculation_verified",
                metadata=numerical_metadata,
            )
        if direct_result.status == "unsupported":
            return StepSupportResult(
                step_index=direct_result.step_index,
                step_text=direct_result.step_text,
                status="unsupported",
                matched_tool_values=list(numerical_result.matched_values),
                source_tools=list(numerical_result.source_tools),
                reason=numerical_result.reason,
                metadata=numerical_metadata,
            )
        direct_result.metadata.update(numerical_metadata)
        return direct_result

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
            metadata = record.metadata if isinstance(record.metadata, dict) else {}
            key = (
                record.tool_name,
                record.output_type,
                self._answer_key(record.value),
                str(metadata.get("evidence_id") or ""),
                str(metadata.get("source_id") or ""),
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
            "support_level": summary.support_level,
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
