from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.attachment_strategy import AttachmentStrategyExecutor
from tools.attachment_workspace import AttachmentWorkspace
from tools.deterministic_handlers import DeterministicHandlerRouter, HandlerTrustGate
from tools.evidence.fact_extraction import (
    SemanticFactExtractor,
    SemanticSourceUnit,
    TaskFactCollector,
    TaskFactStore,
)
from tools.evidence.evidence_readiness import (
    EvidenceReadiness,
    EvidenceReadinessEvaluator,
    EvidenceReadinessStatus,
)
from tools.search_result_builder.evidence import (
    BestEffortReferenceSelector,
    EvidenceConverter,
    EvidenceSelectionContract,
    SpanBuilder,
)
from tools.search_result_builder.retrieval_control import WebRetrievalControl
from tools.search_result_builder.source_analyze import PROJECT_LABELER_CHECKPOINT
from tools.system_routing_contract import SystemRoutingContract
from tools.validation import ToolResultValidator
from utils.network_utils import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvidenceRunner:
    """
    在 Stage1 開始前準備共享 evidence，包含 attachment、search 與 deterministic solver 結果。

    Args:
        - question: 使用者輸入的問題。
        - attachment: 題目附檔資訊或已解析內容。
        - tool_manager: 可執行 search 等工具的管理器。
        - search_result: 外部預先提供的 search evidence。
        - attachment_result: 外部預先提供的 attachment evidence。
        - routing_contract: 判斷 Stage1 是否需要使用工具的 routing contract。
        - attachment_evidence_builder: 建立 attachment evidence 的 builder。
        - deterministic_solver: 嘗試解決 deterministic 類型問題的 solver。

    Returns:
        - dict[str, Any]: 包含 search_result、attachment_result、solver_result、routing 與 tool_usage。
        - 空字串欄位: 對於未使用或失敗的 evidence 類型回傳空字串。
    """

    def __init__(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        tool_manager: Any | None = None,
        search_result: str = "",
        attachment_result: str = "",
        routing_contract: SystemRoutingContract | None = None,
        attachment_evidence_builder: AttachmentEvidenceBuilder | None = None,
        attachment_strategy_executor: AttachmentStrategyExecutor | None = None,
        deterministic_solver: Any | None = None,
        deterministic_handler_router: DeterministicHandlerRouter | None = None,
        tool_result_validator: ToolResultValidator | None = None,
        handler_trust_gate: HandlerTrustGate | None = None,
        span_builder: SpanBuilder | None = None,
        evidence_converter: EvidenceConverter | None = None,
        compact_search_evidence: bool = False,
        enable_evidence_driven_search: bool = True,
        bypass_search_labeler: bool = False,
        max_parallel_next_hop_queries: int = 2,
        attachment_workspace: AttachmentWorkspace | None = None,
        semantic_fact_extractor: SemanticFactExtractor | None = None,
        fact_store: TaskFactStore | None = None,
        fact_collector: TaskFactCollector | None = None,
        evidence_readiness_evaluator: EvidenceReadinessEvaluator | None = None,
        best_effort_reference_selector: BestEffortReferenceSelector | None = None,
    ) -> None:
        self.question = question
        self.attachment = attachment or {}
        self.tool_manager = tool_manager
        self.search_result = search_result
        self.attachment_result = attachment_result
        self.attachment_workspace = attachment_workspace
        self.routing_contract = routing_contract or SystemRoutingContract()
        self.attachment_evidence_builder = attachment_evidence_builder or AttachmentEvidenceBuilder()
        self.deterministic_solver = deterministic_solver
        self.deterministic_handler_router = deterministic_handler_router or DeterministicHandlerRouter()
        self.tool_result_validator = tool_result_validator or ToolResultValidator()
        self.handler_trust_gate = handler_trust_gate or HandlerTrustGate()
        self.attachment_strategy_executor = attachment_strategy_executor or AttachmentStrategyExecutor(
            attachment_builder=self.attachment_evidence_builder,
            handler_router=self.deterministic_handler_router,
            trust_gate=self.handler_trust_gate,
        )
        self.span_builder = span_builder or SpanBuilder()
        self.evidence_converter = evidence_converter or EvidenceConverter(
            span_builder=self.span_builder,
        )
        self.compact_search_evidence = compact_search_evidence
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.bypass_search_labeler = bool(bypass_search_labeler)
        self.max_parallel_next_hop_queries = max(0, max_parallel_next_hop_queries)
        self.semantic_fact_extractor = semantic_fact_extractor
        self.fact_store = fact_store or TaskFactStore()
        self.fact_collector = fact_collector or TaskFactCollector()
        self.evidence_readiness_evaluator = (
            evidence_readiness_evaluator or EvidenceReadinessEvaluator()
        )
        self.best_effort_reference_selector = (
            best_effort_reference_selector or BestEffortReferenceSelector()
        )

    def run(self) -> dict[str, Any]:
        """
        執行 evidence routing，並平行準備 attachment 與 search evidence。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: Stage1 可使用的 evidence 與工具使用紀錄。
        """
        routing = self._route_stage1_tools()
        routing["primary_route"] = self._primary_route_from_routing(routing)
        route_transitions: list[dict[str, Any]] = [
            {
                "from": "task_received",
                "to": routing["primary_route"],
                "reason": "system_routing_contract",
            }
        ]
        tool_usage: list[dict[str, Any]] = []
        attachment_result = self._resolve_attachment_result()
        search_result = self.search_result.strip()
        solver_result = ""
        if str(routing.get("question_encoding") or "") == "reversed":
            solver_result, decode_usage = self._decode_reversed_question()
            tool_usage.extend(decode_usage)
        attachment_strategy_metadata: dict[str, Any] = {}
        attachment_answer_requirement = ""
        attachment_strategy_executed = False
        if self._has_attachment_metadata():
            attachment_strategy_executed = True
            attachment_was_prepared = bool(attachment_result.strip())
            strategy_result = self.attachment_strategy_executor.run(
                question=self.question,
                attachment=self.attachment,
                existing_attachment_context=attachment_result,
                search_context=search_result,
            )
            if self.attachment_workspace is not None:
                self.attachment_workspace.seed_from_strategy_result(
                    strategy_result,
                    reader_executed=not attachment_was_prepared,
                )
            attachment_result, strategy_usage = self._validate_attachment_strategy_output(
                strategy_result,
                fallback_context=attachment_result,
            )
            solver_result = (
                strategy_result.solver_context
                if strategy_result.handler_status == "success"
                else ""
            )
            tool_usage.extend(strategy_usage)
            attachment_strategy_metadata = strategy_result.to_dict()
            active_strategy = strategy_result.revised_strategy or strategy_result.strategy
            attachment_answer_requirement = (
                active_strategy.expected_answer
                if strategy_result.strategy_status == "success"
                else ""
            )
            routing["attachment_strategy_loop"] = {
                "enabled": True,
                "reader_status": strategy_result.reader_status,
                "strategy_status": strategy_result.strategy_status,
                "handler_status": strategy_result.handler_status,
                "strategy": strategy_result.strategy.to_dict(),
                "revised_strategy": (
                    strategy_result.revised_strategy.to_dict()
                    if strategy_result.revised_strategy
                    else None
                ),
                "needs_search": bool(strategy_result.metadata.get("needs_search")),
                "next_capability": (
                    active_strategy.next_capability
                    or ("search" if active_strategy.needs_search else "")
                ),
                "final_answer_candidate": strategy_result.final_answer_candidate,
            }
            if strategy_result.metadata.get("needs_search"):
                routing["use_search"] = True
                routing["search_allowed"] = True
                routing["search_policy"] = "deferred"
        search_deferred = self._should_defer_search(
            primary_route=str(routing.get("primary_route", "")),
            routing=routing,
            tool_usage=tool_usage,
        )
        search_skip_reason = (
            "blocked_by_metadata_first_route"
            if search_deferred and routing.get("search_allowed") is False
            else "deferred_until_non_search_tools_complete"
            if search_deferred
            else ""
        )

        evidence_tasks = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if not attachment_result and routing.get("use_attachment"):
                evidence_tasks[executor.submit(self._build_attachment_evidence)] = "attachment"
            if not search_result and routing.get("use_search") and not search_deferred:
                evidence_tasks[executor.submit(self._build_search_evidence)] = "search"

            for future in as_completed(evidence_tasks):
                task_name = evidence_tasks[future]
                try:
                    result_text, result_usage = future.result()
                except Exception as exc:
                    result_text = ""
                    result_usage = [
                        {
                            "ok": False,
                            "tool_name": task_name,
                            "output_text": "",
                            "raw_result": None,
                            "error": str(exc),
                        }
                    ]

                if task_name == "attachment":
                    attachment_result = result_text
                elif task_name == "search":
                    search_result = result_text
                tool_usage.extend(result_usage)

        if (
            not solver_result
            and not attachment_strategy_executed
            and (routing.get("use_deterministic_solver") or routing.get("use_python_solver"))
        ):
            solver_result, solver_usage = self._build_deterministic_handler_evidence(
                attachment_context=attachment_result,
                search_context=search_result,
            )
            tool_usage.extend(solver_usage)
            solver_result, retry_usage = self._retry_deterministic_after_gap(
                solver_result=solver_result,
                tool_usage=solver_usage,
                attachment_result=attachment_result,
                search_result=search_result,
                allow_search=str(routing.get("search_policy") or "fallback").lower()
                != "forbidden",
            )
            if retry_usage:
                if any(item.get("tool_name") == "attachment_reader" for item in retry_usage):
                    attachment_result = self._last_output_for_tool(retry_usage, "attachment_reader") or attachment_result
                if any(item.get("tool_name") == "search" for item in retry_usage):
                    search_result = self._last_output_for_tool(retry_usage, "search") or search_result
                tool_usage.extend(retry_usage)

        self._collect_facts(tool_usage)
        readiness = self._evaluate_readiness(
            routing=routing,
            tool_usage=tool_usage,
            attachment_result=attachment_result,
            search_result=search_result,
            solver_result=solver_result,
        )
        route_transitions.append(
            {
                "from": routing["primary_route"],
                "to": readiness.status.value,
                "reason": readiness.reason,
            }
        )

        if search_deferred and not search_result.strip():
            search_policy = str(routing.get("search_policy") or "fallback").lower()
            if readiness.is_sufficient:
                search_skip_reason = "non_search_direct_evidence_sufficient"
            elif (
                readiness.status == EvidenceReadinessStatus.NEEDS_EXTERNAL
                and readiness.next_capability == "search"
                and search_policy != "forbidden"
            ):
                search_text, search_usage = self._build_search_evidence()
                search_result = search_text
                tool_usage.extend(search_usage)
                self._collect_facts(search_usage)
                search_skip_reason = ""
                route_transitions.append(
                    {
                        "from": readiness.status.value,
                        "to": "search",
                        "reason": "readiness_requested_search",
                    }
                )
                readiness = self._evaluate_readiness(
                    routing=routing,
                    tool_usage=tool_usage,
                    attachment_result=attachment_result,
                    search_result=search_result,
                    solver_result=solver_result,
                )
                route_transitions.append(
                    {
                        "from": "search",
                        "to": readiness.status.value,
                        "reason": readiness.reason,
                    }
                )
            else:
                search_skip_reason = "readiness_did_not_request_search"

        deterministic_gap = self._deterministic_gap_from_usage(tool_usage)
        if deterministic_gap:
            routing["deterministic_tool_gap"] = deterministic_gap
        routing["route_transitions"] = route_transitions[:6]
        routing["final_evidence_state"] = readiness.to_dict()
        routing["search_decision"] = self._search_decision_trace(
            primary_route=str(routing.get("primary_route", "")),
            search_allowed=routing.get("search_allowed"),
            search_executed=bool(search_result.strip()),
            search_skipped=search_deferred and not search_result.strip(),
            skip_reason=search_skip_reason,
            tool_usage=tool_usage,
        )
        return self._finalize_evidence_bundle({
            "search_result": search_result.strip(),
            "attachment_result": attachment_result.strip(),
            "solver_result": solver_result.strip(),
            "answer_requirement": attachment_answer_requirement.strip(),
            "routing": routing,
            "tool_usage": tool_usage,
            "evidence_readiness": readiness.to_dict(),
            "attachment_profile": dict(
                attachment_strategy_metadata.get("attachment_profile") or {}
            ),
            "attachment_strategy": attachment_strategy_metadata,
        })

    def _finalize_evidence_bundle(
        self,
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        """將證據準備階段的各類來源寫入任務級事實庫。"""

        self.fact_collector.collect_many(
            self.fact_store,
            list(bundle.get("tool_usage") or []),
            question=self.question,
            source_scope="evidence_prepare",
        )
        self._attach_search_contract_state(bundle)
        bundle["fact_store"] = self.fact_store.to_dict()
        return bundle

    def _collect_facts(self, tool_usage: list[dict[str, Any]]) -> None:
        """Collect newly available tool facts before readiness evaluation."""

        self.fact_collector.collect_many(
            self.fact_store,
            list(tool_usage or []),
            question=self.question,
            source_scope="evidence_prepare",
        )

    def _evaluate_readiness(
        self,
        *,
        routing: dict[str, Any],
        tool_usage: list[dict[str, Any]],
        attachment_result: str,
        search_result: str,
        solver_result: str,
    ) -> EvidenceReadiness:
        return self.evidence_readiness_evaluator.evaluate(
            fact_store=self.fact_store,
            tool_usage=tool_usage,
            routing=routing,
            attachment_result=attachment_result,
            search_result=search_result,
            solver_result=solver_result,
        )

    def _attach_search_contract_state(self, bundle: dict[str, Any]) -> None:
        """將搜尋階段的 final intent/relation state 提升到共用 evidence bundle。"""

        for usage in list(bundle.get("tool_usage") or []):
            if not isinstance(usage, dict) or usage.get("tool_name") != "search":
                continue
            raw_result = usage.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            diagnostics = raw_result.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            relation_plan = diagnostics.get("relation_plan")
            if not isinstance(relation_plan, dict):
                relation_plan = raw_result.get("relation_plan")
            search_intent_plan = diagnostics.get("search_intent_plan")
            if isinstance(relation_plan, dict):
                bundle["relation_plan"] = relation_plan
            if isinstance(search_intent_plan, dict):
                bundle["search_intent_plan"] = search_intent_plan
                bundle.setdefault(
                    "answer_role",
                    normalize_text(str(search_intent_plan.get("answer_role") or "")),
                )
                bundle.setdefault(
                    "answer_target",
                    normalize_text(str(search_intent_plan.get("target") or "")),
                )
            return

    def _primary_route_from_routing(self, routing: dict[str, Any]) -> str:
        initial_route = str(routing.get("initial_route") or "").strip().lower()
        if initial_route == "attachment_first":
            return "attachment"
        if initial_route == "deterministic_first":
            return "deterministic"
        if initial_route == "search_first":
            return "factual_search"
        if initial_route == "agent_direct":
            return "unknown"
        task_type = str(routing.get("task_type") or "").strip().lower()
        if task_type in {"hybrid_search_and_solver"}:
            return "hybrid"
        if task_type in {"factual_search"}:
            return "factual_search"
        if task_type in {"attachment_evidence", "closed_world_attachment"}:
            return "attachment"
        if task_type in {"deterministic_solver", "closed_world_puzzle", "attachment_deterministic_solver"}:
            return "deterministic"
        if routing.get("use_attachment"):
            return "attachment"
        if routing.get("use_deterministic_solver") or routing.get("use_python_solver"):
            return "deterministic"
        if routing.get("use_search"):
            return "factual_search"
        return "unknown"

    def _should_defer_search(
        self,
        *,
        primary_route: str,
        routing: dict[str, Any],
        tool_usage: list[dict[str, Any]],
    ) -> bool:
        if routing.get("search_allowed") is False and not self._deterministic_gap_requires_search(tool_usage):
            return True
        if primary_route in {"factual_search", "hybrid"}:
            return False
        if self._deterministic_gap_requires_search(tool_usage):
            return False
        if not routing.get("use_search"):
            return True
        return primary_route in {"attachment", "deterministic", "media", "unknown"}

    def _has_trusted_deterministic_final(self, tool_usage: list[dict[str, Any]]) -> bool:
        for item in reversed(tool_usage):
            if item.get("tool_name") not in {"deterministic_handler_router", "attachment_strategy_handler"}:
                continue
            trust = item.get("handler_trust") if isinstance(item.get("handler_trust"), dict) else {}
            if (
                item.get("ok")
                and item.get("evidence_valid")
                and item.get("output_type") == "final_answer"
                and trust.get("trusted", True)
            ):
                return True
        return False

    def _attachment_strategy_needs_search(self, routing: dict[str, Any]) -> bool:
        loop = routing.get("attachment_strategy_loop")
        if not isinstance(loop, dict):
            return False
        return bool(loop.get("needs_search"))

    def _deterministic_gap_requires_search(self, tool_usage: list[dict[str, Any]]) -> bool:
        gap = self._deterministic_gap_from_usage(tool_usage)
        if not gap:
            return False
        missing = {str(item).strip() for item in gap.get("missing_inputs", []) or []}
        return bool(missing & self._SEARCH_GAP_INPUTS)

    def _search_decision_trace(
        self,
        *,
        primary_route: str,
        search_allowed: Any,
        search_executed: bool,
        search_skipped: bool,
        skip_reason: str,
        tool_usage: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "primary_route": primary_route,
            "search_allowed": search_allowed,
            "search_executed": bool(search_executed),
            "search_skipped": bool(search_skipped),
            "skip_reason": skip_reason,
            "non_search_sufficient": self._has_trusted_deterministic_final(tool_usage),
            "deterministic_gap_requires_search": self._deterministic_gap_requires_search(tool_usage),
            "tools_seen": [
                str(item.get("tool_name") or "")
                for item in tool_usage
                if str(item.get("tool_name") or "")
            ],
        }


    _ATTACHMENT_GAP_INPUTS = {
        "table_rows",
        "source_text",
        "grid",
        "candidate_words",
        "edges",
        "date_values",
        "numbers",
        "list_items",
        "quoted_or_inline_text",
        "two_coordinate_pairs",
    }
    _SEARCH_GAP_INPUTS = {
        "source_text",
        "date_values",
        "numbers",
        "matching_text",
        "connected_path",
    }

    def _retry_deterministic_after_gap(
        self,
        *,
        solver_result: str,
        tool_usage: list[dict[str, Any]],
        attachment_result: str,
        search_result: str,
        executed: set[str] | None = None,
        handler_name: str = "",
        handler_plan: dict[str, Any] | None = None,
        allow_search: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        gap = self._deterministic_gap_from_usage(tool_usage)
        if not gap or solver_result.strip():
            return solver_result, []

        executed = executed or set()
        retry_usage: list[dict[str, Any]] = []
        missing = set(gap.get("missing_inputs", []) or [])
        updated_attachment = attachment_result
        updated_search = search_result

        if (
            not updated_attachment.strip()
            and self.attachment
            and "attachment_reader" not in executed
            and missing & self._ATTACHMENT_GAP_INPUTS
        ):
            result_text, result_usage = self._build_attachment_evidence()
            updated_attachment = result_text
            retry_usage.extend(self._mark_gap_recovery_usage(result_usage, gap, "attachment_reader"))
            executed.add("attachment_reader")

        if (
            not updated_search.strip()
            and allow_search
            and "search" not in executed
            and missing & self._SEARCH_GAP_INPUTS
        ):
            result_text, result_usage = self._build_search_evidence()
            updated_search = result_text
            retry_usage.extend(self._mark_gap_recovery_usage(result_usage, gap, "search"))
            executed.add("search")

        if updated_attachment == attachment_result and updated_search == search_result:
            return solver_result, retry_usage

        retry_result, deterministic_usage = self._build_deterministic_handler_evidence(
            attachment_context=updated_attachment,
            search_context=updated_search,
            handler_name=handler_name,
            handler_plan=handler_plan or {},
        )
        for item in deterministic_usage:
            item["gap_recovery_retry"] = True
            item["previous_deterministic_gap"] = gap
        retry_usage.extend(deterministic_usage)
        return retry_result or solver_result, retry_usage

    def _mark_gap_recovery_usage(
        self,
        usage: list[dict[str, Any]],
        gap: dict[str, Any],
        tool_name: str,
    ) -> list[dict[str, Any]]:
        marked: list[dict[str, Any]] = []
        for item in usage:
            copied = dict(item)
            copied["gap_recovery_for"] = "deterministic_handler"
            copied["gap_recovery_tool"] = tool_name
            copied["previous_deterministic_gap"] = gap
            marked.append(copied)
        return marked

    def _last_output_for_tool(
        self,
        usage: list[dict[str, Any]],
        tool_name: str,
    ) -> str:
        for item in reversed(usage):
            if item.get("tool_name") == tool_name:
                return str(item.get("output_text", "") or "")
        return ""

    def _deterministic_gap_from_usage(
        self,
        tool_usage: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for item in reversed(tool_usage):
            if item.get("tool_name") != "deterministic_handler_router":
                continue
            if item.get("ok"):
                return {}
            missing = list(item.get("missing_inputs") or [])
            if not missing:
                raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
                missing = list(raw.get("missing_inputs") or [])
            if not missing:
                raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
                selected = (raw.get("structured_result") or {}).get("selected_match") or {}
                missing = list(selected.get("missing_inputs") or [])
            if not missing:
                continue
            raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
            return {
                "handler_name": item.get("handler_name") or raw.get("handler_name", ""),
                "status": item.get("status") or raw.get("status", ""),
                "missing_inputs": missing,
                "next_action_hint": item.get("next_action_hint") or raw.get("next_action_hint", ""),
                "error": item.get("error") or raw.get("error", ""),
                "selected_match": (raw.get("structured_result") or {}).get("selected_match", {}),
            }
        return {}

    def _route_stage1_tools(self) -> dict[str, Any]:
        """
        根據問題與附檔狀態決定 Stage1 evidence 準備需要哪些工具。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: routing decision 與預先提供 evidence 的標記。
        """
        decision = self.routing_contract.route(
            question=self.question,
            stage="stage1_round0",
            has_attachment=bool(self.attachment),
            attachment_type=self._attachment_type(),
        )
        routing = decision.to_dict()
        if self.search_result:
            routing["use_search"] = False
            routing["search_allowed"] = False
            routing["provided_search_result"] = True
        if self.attachment_result:
            routing["use_attachment"] = False
            routing["provided_attachment_result"] = True
        return routing

    def _decode_reversed_question(self) -> tuple[str, list[dict[str, Any]]]:
        """
        將反寫題目字元反轉成明文，作為可信的中間 context 提供給 Stage1。

        Args:
            - 無。

        Returns:
            - str: 附說明的解碼文字，放入 solver_result。
            - list[dict[str, Any]]: 解碼工具使用紀錄。
        """
        decoded = str(self.question or "")[::-1].strip()
        context = (
            "Decoded_Question (the original question text is written in "
            "reverse; read this decoded version instead):\n"
            f"{decoded}"
        )
        usage = {
            "ok": True,
            "tool_name": "reversed_text_decoder",
            "status": "success",
            "output_type": "intermediate_value",
            "output_text": context,
            "value": decoded,
            "trusted": True,
            "evidence_valid": True,
            "raw_result": {"decoded_question": decoded, "encoding": "reversed"},
            "error": None,
        }
        return context, [usage]

    def _attachment_type(self) -> str | None:
        """
        從 attachment metadata 或 file_path 推斷附檔副檔名。

        Args:
            - 無。

        Returns:
            - str | None: 附檔類型；無法判斷時回傳 None。
        """
        extension = str(self.attachment.get("extension", "") or "").strip().lower()
        if extension.startswith("."):
            return extension[1:]
        if extension:
            return extension
        file_path = str(self.attachment.get("file_path", "") or "")
        if "." in file_path:
            return file_path.rsplit(".", 1)[-1].lower()
        return None

    def _has_attachment_metadata(self) -> bool:
        """
        判斷任務是否帶有附件 metadata。

        Args:
         - 無。

        Returns:
         - bool: 只要有附件路徑、名稱、類型或既有內容，即視為附件題。
        """
        if not self.attachment:
            return False
        for key in ("file_path", "path", "file_name", "extension", "context", "attachment_context"):
            if str(self.attachment.get(key) or "").strip():
                return True
        return False

    def _resolve_attachment_result(self) -> str:
        """
        優先使用外部提供的 attachment evidence，否則從 attachment dict 取已解析內容。

        Args:
            - 無。

        Returns:
            - str: 可直接放入 Stage1 context 的 attachment evidence。
        """
        if self.attachment_result:
            return self.attachment_result
        for key in ("context", "attachment_context", "content", "text"):
            value = self.attachment.get(key)
            if value:
                return str(value)
        return ""

    def _build_attachment_evidence(self) -> tuple[str, list[dict[str, Any]]]:
        """
        呼叫 AttachmentEvidenceBuilder 讀取附檔並建立 evidence。

        Args:
            - 無。

        Returns:
            - str: attachment context 文字。
            - list[dict[str, Any]]: attachment reader 的工具使用紀錄。
        """
        try:
            result = self.attachment_evidence_builder.build(self.question, self.attachment)
        except Exception as exc:
            return self._validate_tool_context("attachment_reader", "", [
                {
                    "ok": False,
                    "tool_name": "attachment_reader",
                    "output_text": "",
                    "raw_result": None,
                    "error": str(exc),
                }
            ])
        return self._validate_tool_context(
            "attachment_reader",
            str(result.get("context", "") or ""),
            list(result.get("tool_usage", []) or []),
        )


    def _build_search_evidence(self) -> tuple[str, list[dict[str, Any]]]:
        """
        使用 tool_manager 執行 search，建立 Stage1 可用的外部查詢 evidence。

        Args:
            - 無。

        Returns:
            - str: search evidence 文字。
            - list[dict[str, Any]]: search tool 的執行結果紀錄。
        """
        try:
            controller = WebRetrievalControl(
                max_queries=3,
                max_results_per_query=5,
                max_pages_to_fetch=6,
                max_chunks_per_url=20,
                max_corpus_records=120,
                max_iter=5 if self.enable_evidence_driven_search else 1,
                top_k=16,
                min_retrieval_score=0.0,
                relative_score_margin=1.0,
                embedding_batch_size=8,
                bypass_labeler=self.bypass_search_labeler,
            )
            output = controller.run(
                self.question,
                output_dir=self._web_retrieval_output_dir(),
            )
            output_dict = self._dataclass_to_dict(output)
            contract = self._evidence_selection_contract(output_dict)
            evidence_items = self._web_retrieval_evidence_items(
                output_dict,
                contract=contract,
            )
            unverified_references = self._web_retrieval_unverified_references(
                output_dict,
                evidence_items=evidence_items,
            )
            answer_candidates: list[dict[str, Any]] = []
            summary = self._render_web_retrieval_evidence(
                evidence_items,
                unverified_references=unverified_references,
                answer_candidates=answer_candidates,
                contract=contract,
            )
            result = {
                "ok": bool(summary.strip()),
                "tool_name": "search",
                "output_text": summary,
                "raw_result": self._web_retrieval_raw_result(
                    output_dict=output_dict,
                    evidence_items=evidence_items,
                    unverified_references=unverified_references,
                    answer_candidates=answer_candidates,
                    contract=contract,
                ),
                "error": None,
            }
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": "search",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }
        return self._validate_tool_context(
            "search",
            str(result.get("output_text", "") or ""),
            [result],
        )

    def _web_retrieval_output_dir(self) -> Path:
        """
        建立單題 WebRetrievalControl 的輸出目錄。

        Args:
            - 無。

        Returns:
            - Path: 本題 corpus、embedding 與 trace 的輸出目錄。
        """
        digest = hashlib.sha1(
            normalize_text(self.question).encode("utf-8")
        ).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        return (
            PROJECT_ROOT
            / "outputs"
            / "web_retrieval_runtime"
            / f"{digest}_{timestamp}"
        )

    def _web_retrieval_evidence_items(
        self,
        output_dict: dict[str, Any],
        *,
        max_items: int = 5,
        max_chars: int = 450,
        contract: EvidenceSelectionContract | None = None,
    ) -> list[dict[str, Any]]:
        """
        從 WebRetrievalControl trace 選出可傳給 Stage1 Agent 的 evidence chunks。

        Args:
            - output_dict: WebRetrievalControl 的 JSON-safe trace。
            - max_items: 最多輸出多少 evidence items。
            - max_chars: 單筆 evidence 最大字元數。

        Returns:
            - list[dict[str, Any]]: 精簡後的 evidence items。
        """
        self.evidence_converter.max_items = max(1, max_items)
        self.evidence_converter.max_chars = max(120, max_chars)
        return self.evidence_converter.convert_web_retrieval_output(
            output_dict,
            contract=contract or self._evidence_selection_contract(output_dict),
            fact_store=self.fact_store,
        )

    def _evidence_selection_contract(
        self,
        output_dict: dict[str, Any],
    ) -> EvidenceSelectionContract:
        """
        從搜尋前處理結果建立 EvidenceConverter 使用的簡單 evidence contract。

        Args:
            - output_dict: WebRetrievalControl 的 JSON-safe trace。

        Returns:
            - EvidenceSelectionContract: 不暴露 SearchIntentPlan 細節的 evidence selection contract。

        """
        diagnostics = output_dict.get("diagnostics") or {}
        query_plan = diagnostics.get("query_plan") or {}
        state = diagnostics.get("search_intent_plan") or {}
        plan_sources = self._contract_plan_sources(diagnostics, query_plan, state)
        answer_requirement = self._first_contract_text(
            plan_sources,
            "answer_requirement",
            "answer_role",
            "role_text",
        )
        answer_target = self._first_contract_text(
            plan_sources,
            "answer_target",
            "target",
        )
        must_include = []
        for source in self._contract_list_values(plan_sources, "must_include"):
            if isinstance(source, list):
                must_include.extend(source)
        return EvidenceSelectionContract.from_parts(
            question=self.question,
            answer_requirement=answer_requirement,
            answer_target=answer_target,
            must_include=must_include,
        )

    def _web_retrieval_unverified_references(
        self,
        output_dict: dict[str, Any],
        *,
        evidence_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        references = self.best_effort_reference_selector.select(
            output_dict,
            strict_evidence_items=evidence_items,
        )
        return [reference.to_dict() for reference in references]

    def _contract_plan_sources(
        self,
        diagnostics: dict[str, Any],
        query_plan: dict[str, Any],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for source in (state, query_plan, diagnostics):
            if isinstance(source, dict):
                sources.append(source)
                for key in ("query_state", "search_intent_plan", "answer_role"):
                    nested = source.get(key)
                    if isinstance(nested, dict):
                        sources.append(nested)
        return sources

    def _first_contract_text(
        self,
        sources: list[dict[str, Any]],
        *field_names: str,
    ) -> str:
        for source in sources:
            for field_name in field_names:
                text = normalize_text(str(source.get(field_name, "") or ""))
                if text and text.casefold() not in {"unknown", "none", "null", "n/a"}:
                    return text
        return ""

    def _contract_list_values(
        self,
        sources: list[dict[str, Any]],
        field_name: str,
    ) -> list[Any]:
        values: list[Any] = []
        for source in sources:
            value = source.get(field_name)
            if isinstance(value, list):
                values.append(value)
        return values

    def _render_web_retrieval_evidence(
        self,
        evidence_items: list[dict[str, Any]],
        unverified_references: list[dict[str, Any]] | None = None,
        answer_candidates: list[dict[str, Any]] | None = None,
        contract: EvidenceSelectionContract | None = None,
    ) -> str:
        """
        將 WebRetrievalControl evidence items 轉成 Stage1 prompt context。

        Args:
            - evidence_items: 精簡後的 evidence items。

        Returns:
            - str: 只包含 source title 與 evidence 內容的 prompt 區塊。
        """
        lines = ["Evidence:"]
        lines.extend(self._render_answer_requirement_lines(contract))
        if evidence_items:
            for index, item in enumerate(evidence_items, start=1):
                lines.extend(
                    [
                        f"[E{index}]",
                        f"Source Title: {item.get('title') or item.get('source_id') or 'Unknown'}",
                        f"Evidence: {item.get('text', '')}",
                    ]
                )
        else:
            lines.append("None")

        references = list(unverified_references or [])
        if references:
            lines.extend(
                [
                    "",
                    "Unverified References:",
                    "These retrieved passages may be incomplete or irrelevant and are not verified answer support.",
                ]
            )
            if not evidence_items:
                lines.append(
                    "Extract the required values or rows from the references below before "
                    "answering. Do not guess from memory."
                )
            for index, item in enumerate(references, start=1):
                lines.extend(
                    [
                        f"Reference {index}:",
                        f"Source Title: {item.get('title') or item.get('source_id') or 'Unknown'}",
                        f"Content: {item.get('text', '')}",
                    ]
                )
        candidates = list(answer_candidates or [])
        if candidates:
            lines.extend(["", "Candidate Answers:"])
            for index, candidate in enumerate(candidates[:5], start=1):
                text = normalize_text(candidate.get("text", ""))
                if text:
                    lines.append(f"{index}. {text}")
        return "\n".join(lines).strip()

    def _render_answer_requirement_lines(
        self,
        contract: EvidenceSelectionContract | None,
    ) -> list[str]:
        if contract is None:
            return []
        lines: list[str] = []
        if contract.answer_requirement:
            lines.append(f"Answer Requirement: {contract.answer_requirement}")
        if contract.answer_target:
            lines.append(f"Answer Target: {contract.answer_target}")
        if lines:
            lines.append("")
        return lines

    def _web_retrieval_raw_result(
        self,
        *,
        output_dict: dict[str, Any],
        evidence_items: list[dict[str, Any]],
        unverified_references: list[dict[str, Any]] | None = None,
        answer_candidates: list[dict[str, Any]] | None = None,
        contract: EvidenceSelectionContract | None = None,
    ) -> dict[str, Any]:
        """
        建立 GAIA log 可讀且與舊 search summary 大致相容的 raw_result。

        Args:
            - output_dict: WebRetrievalControl 的完整 trace。
            - evidence_items: Stage1 實際收到的 evidence items。

        Returns:
            - dict[str, Any]: 可序列化 search raw result。
        """
        web_searches = list(output_dict.get("web_searches") or [])
        retrieval = output_dict.get("retrieval") or {}
        rounds = list(retrieval.get("rounds") or [])
        searched_queries = list(retrieval.get("searched_queries") or [])
        next_queries = [
            normalize_text(round_info.get("next_query", ""))
            for round_info in rounds
            if normalize_text(round_info.get("next_query", ""))
        ]
        source_count = sum(
            int(search.get("result_count", 0) or 0)
            for search in web_searches
        )
        blocked_source_count = int(
            (output_dict.get("diagnostics") or {}).get(
                "blocked_source_count",
                0,
            )
            or 0
        )
        references = list(unverified_references or [])
        diagnostics = {
            **dict(output_dict.get("diagnostics") or {}),
            "initial_web_preprocessing": {
                "source_pipeline": (
                    (output_dict.get("diagnostics") or {}).get("corpus_pipeline")
                    or "web_search->seer_source_filter->full_page_fetch->clean->chunk->dedup->e5_faiss"
                ),
                "labeler_stage": {
                    "position": "after_faiss_retrieval_before_filter",
                    "adapter": "EfficientRAGLabelerAdapter",
                    "implementation": "efficientrag_pretrained_sequence_token_model",
                    "checkpoint": str(PROJECT_LABELER_CHECKPOINT),
                    "device": "cpu",
                },
                "source_count": source_count,
                "filtered_source_count": (output_dict.get("diagnostics") or {}).get(
                    "filtered_source_count",
                    0,
                ),
                "blocked_source_count": blocked_source_count,
                "corpus_record_count": output_dict.get("corpus_record_count", 0),
                "fetched_pages": (output_dict.get("diagnostics") or {}).get(
                    "fetched_page_count",
                    0,
                ),
                "source_filter": (output_dict.get("diagnostics") or {}).get(
                    "source_filter",
                    {},
                ),
                "full_page_fetch": (output_dict.get("diagnostics") or {}).get(
                    "full_page_fetch",
                    {},
                ),
                "sufficiency_decision": "delegated_to_web_retrieval_control",
            },
            "initial_retrieval_decision": {
                "controller": "WebRetrievalControl",
                "initial_retrieval_query": (output_dict.get("diagnostics") or {}).get(
                    "initial_retrieval_query",
                    "",
                ),
                "stop_reason": retrieval.get("stop_reason", ""),
            },
            "final_retrieval_decision": {
                "controller": "WebRetrievalControl",
                "stop_reason": retrieval.get("stop_reason", ""),
                "searched_queries": searched_queries,
            },
            "sufficiency_method": {
                "controller": "WebRetrievalControl",
                "signals": [
                    "query_generation",
                    "seer_source_filter",
                    "e5_faiss_retrieval",
                    (
                        "span_role_classifier"
                        if self.bypass_search_labeler
                        else "efficientrag_labeler"
                    ),
                    "efficientrag_filter",
                ],
            },
            "web_retrieval_control": {
                "enabled": True,
                "round_count": len(rounds),
                "searched_queries": searched_queries,
                "retrieval_stop_reason": retrieval.get("stop_reason", ""),
                "unique_document_count": retrieval.get(
                    "unique_document_count",
                    0,
                ),
            },
            "coverage_summary": (output_dict.get("diagnostics") or {}).get(
                "coverage_summary",
                {},
            ),
            "evidence_conversion": self._dataclass_to_dict(
                self.evidence_converter.last_diagnostics,
            ),
            "best_effort_evidence": {
                "enabled": True,
                "triggered": bool(references),
                "fallback_reason": (
                    "strict_evidence_empty" if references else ""
                ),
                "strict_evidence_count": len(evidence_items),
                "unverified_reference_count": len(references),
                "selected_reference_ids": [
                    str(item.get("reference_id") or "") for item in references
                ],
            },
            "evidence_selection_contract": (
                contract.to_dict() if contract else {}
            ),
            "evidence_driven_search": {
                "enabled": self.enable_evidence_driven_search,
                "triggered": len(searched_queries) > 1 or bool(next_queries),
                "mode": "web_retrieval_control_iterative_filter",
                "labeler_bypassed": self.bypass_search_labeler,
                "queries": next_queries,
                "query_ids": [
                    f"H{index}"
                    for index, _ in enumerate(next_queries, start=1)
                ],
                "parallel_query_count": 0,
                "max_parallel_queries": self.max_parallel_next_hop_queries,
                "evidence_gain": len(evidence_items),
                "stop_reason": retrieval.get("stop_reason", ""),
            },
            "final_counts": {
                "query_count": len(output_dict.get("generated_queries") or []),
                "retrieval_query_count": len(searched_queries),
                "source_count": source_count,
                "evidence_count": len(evidence_items),
                "strict_evidence_count": len(evidence_items),
                "unverified_reference_count": len(references),
                "stage1_search_context_empty": not bool(
                    evidence_items or references
                ),
                "answer_candidate_count": len(answer_candidates or []),
                "blocked_source_count": blocked_source_count,
                "corpus_record_count": output_dict.get(
                    "corpus_record_count",
                    0,
                ),
            },
        }
        diagnostics["pipeline_failure_stage"] = self._web_retrieval_failure_stage(
            diagnostics,
            evidence_items=evidence_items,
            retrieval=retrieval,
        )
        return {
            "question": output_dict.get("question", ""),
            "queries": output_dict.get("generated_queries", []),
            "salient_spans": output_dict.get("salient_spans", []),
            "web_searches": web_searches,
            "sources": [],
            "evidence_items": evidence_items,
            "verified_evidence_items": evidence_items,
            "unverified_references": references,
            "answer_candidates": list(answer_candidates or []),
            "summary": self._render_web_retrieval_evidence(
                evidence_items,
                unverified_references=references,
                answer_candidates=answer_candidates,
                contract=contract,
            ),
            "diagnostics": diagnostics,
            "blocked_sources": self._web_retrieval_blocked_sources(output_dict),
            "corpus_path": output_dict.get("corpus_path", ""),
            "embedding_path": output_dict.get("embedding_path", ""),
            "retrieval": retrieval,
        }

    def _web_retrieval_failure_stage(
        self,
        diagnostics: dict[str, Any],
        *,
        evidence_items: list[dict[str, Any]],
        retrieval: dict[str, Any],
    ) -> str:
        final_counts = diagnostics.get("final_counts") or {}
        preprocessing = diagnostics.get("initial_web_preprocessing") or {}
        if int(final_counts.get("source_count", 0) or 0) <= 0:
            return "no_sources"
        if int(preprocessing.get("filtered_source_count", 0) or 0) <= 0:
            return "all_sources_blocked"
        if int(preprocessing.get("corpus_record_count", 0) or 0) <= 0:
            return "no_corpus_chunks"
        if not (retrieval.get("rounds") or []):
            return "no_retrieval_rounds"
        if not evidence_items:
            stop_reason = str(retrieval.get("stop_reason", "") or "")
            if stop_reason:
                return f"evidence_conversion_empty:{stop_reason}"
            return "evidence_conversion_empty"
        return ""

    def _web_retrieval_blocked_sources(
        self,
        output_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        將 WebRetrievalControl 擋下的 source 轉成 log 可讀的精簡資料。

        Args:
            - output_dict: WebRetrievalResult 轉成的 dict。

        Returns:
            - list[dict[str, Any]]: blocked source detail。
        """
        blocked_sources: list[dict[str, Any]] = []
        for source in output_dict.get("blocked_sources") or []:
            if not isinstance(source, dict):
                continue
            snippet = self._truncate_text(str(source.get("snippet", "") or ""), max_chars=500)
            raw_preview = self._truncate_text(str(source.get("raw_content", "") or ""), max_chars=500)
            blocked_sources.append(
                {
                    "source_id": source.get("source_id", ""),
                    "query_id": source.get("query_id", ""),
                    "rank": source.get("rank", 0),
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "domain": source.get("domain", ""),
                    "snippet": snippet,
                    "raw_content_preview": raw_preview,
                    "block_reason": source.get("block_reason", ""),
                    "filter_reasons": list(source.get("filter_reasons") or []),
                }
            )
        return blocked_sources

    def _dataclass_to_dict(self, value: Any) -> Any:
        """
        遞迴轉換 dataclass 與 Path，讓 log 可以直接 JSON 序列化。

        Args:
            - value: 任意 Python 物件。

        Returns:
            - Any: JSON-safe value。
        """
        if is_dataclass(value):
            return self._dataclass_to_dict(asdict(value))
        if isinstance(value, list):
            return [self._dataclass_to_dict(item) for item in value]
        if isinstance(value, tuple):
            return [self._dataclass_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._dataclass_to_dict(item)
                for key, item in value.items()
            }
        if isinstance(value, Path):
            return str(value)
        return value

    def _truncate_text(self, text: str, *, max_chars: int) -> str:
        cleaned = normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + " ..."

    def _validate_attachment_strategy_output(
        self,
        strategy_result: Any,
        *,
        fallback_context: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """驗證 Strategy Executor 內部的 attachment_reader 輸出並保留工具順序。"""
        original_usage = list(getattr(strategy_result, "tool_usage", []) or [])
        attachment_usage = [
            item for item in original_usage if item.get("tool_name") == "attachment_reader"
        ]
        strategy_context = str(getattr(strategy_result, "attachment_context", "") or "")
        if not attachment_usage:
            return strategy_context.strip() or fallback_context.strip(), original_usage

        validated_context, validated_attachment_usage = self._validate_tool_context(
            "attachment_reader",
            strategy_context,
            attachment_usage,
        )
        validated_iter = iter(validated_attachment_usage)
        merged_usage: list[dict[str, Any]] = []
        for item in original_usage:
            if item.get("tool_name") == "attachment_reader":
                merged_usage.append(next(validated_iter))
            else:
                merged_usage.append(item)
        return validated_context.strip() or fallback_context.strip(), merged_usage

    def _validate_tool_context(
        self,
        tool_name: str,
        output_text: str,
        usage: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        對一般工具輸出做 light validation，避免無效輸出進入 Agent context。

        Args:
            - tool_name: 主要工具名稱。
            - output_text: 準備放入 context 的文字。
            - usage: 工具執行紀錄。

        Returns:
            - tuple[str, list[dict[str, Any]]]: 通過驗證的 context 與補上 validation 的 usage。
        """
        items = [dict(item) for item in usage or []]
        if not items:
            items = [
                {
                    "ok": bool(str(output_text or "").strip()),
                    "tool_name": tool_name,
                    "output_text": output_text,
                    "raw_result": None,
                    "error": None,
                }
            ]

        validated_items: list[dict[str, Any]] = []
        primary_valid = False
        for item in items:
            item_tool = str(item.get("tool_name") or tool_name)
            item_output = str(item.get("output_text") or "")
            if item_tool == tool_name and not item_output:
                item_output = output_text
            validation = self.tool_result_validator.validate(
                tool_name=item_tool,
                raw_result=item.get("raw_result"),
                output_text=item_output,
                metadata=item,
            )
            item["validation"] = validation.to_dict()
            item["evidence_valid"] = validation.valid
            if item_tool == tool_name and validation.valid:
                primary_valid = True
            validated_items.append(item)

        if not any(str(item.get("tool_name") or "") == tool_name for item in validated_items):
            validation = self.tool_result_validator.validate(
                tool_name=tool_name,
                raw_result=None,
                output_text=output_text,
                metadata={"ok": bool(str(output_text or "").strip())},
            )
            validated_items.insert(
                0,
                {
                    "ok": validation.valid,
                    "tool_name": tool_name,
                    "output_text": output_text if validation.valid else "",
                    "raw_result": None,
                    "error": None,
                    "validation": validation.to_dict(),
                    "evidence_valid": validation.valid,
                },
            )
            primary_valid = validation.valid

        return (output_text if primary_valid else ""), validated_items

    def _build_solver_evidence(
        self,
        *,
        attachment_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        使用 deterministic solver 嘗試產生可直接支持答案的 evidence。

        Args:
            - attachment_context: 已解析的附檔內容，供 solver 處理表格或封閉資料問題。

        Returns:
            - str: solver evidence 文字。
            - list[dict[str, Any]]: solver 執行紀錄。
        """
        return self._build_deterministic_handler_evidence(
            attachment_context=attachment_context,
            search_context="",
        )

    def _build_deterministic_handler_evidence(
        self,
        *,
        attachment_context: str,
        search_context: str,
        handler_name: str = "",
        handler_plan: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Run the deterministic handler router after attachment/search evidence is available.
        """
        try:
            result = self.deterministic_handler_router.run(
                question=self.question,
                attachment=self.attachment,
                attachment_result=attachment_context,
                search_result=search_context,
                handler_name=handler_name,
                required_handler_role=str((handler_plan or {}).get("required_handler_role") or ""),
            )
            payload = result.to_dict()
            trust = self.handler_trust_gate.validate(
                result,
                question=self.question,
                handler_plan=handler_plan or {},
            )
            context = trust.evidence_text if trust.trusted else ""
            if (
                result.ok
                and result.output_type == "intermediate_value"
                and result.semantic_role
                and result.supporting_inputs
            ):
                context = result.evidence_text
            evidence_valid = bool(trust.trusted and result.output_type == "final_answer")
            next_action_hint = result.next_action_hint or (
                "Recover missing deterministic inputs: " + ", ".join(result.missing_inputs)
                if result.missing_inputs
                else ""
            )
            return context, [
                {
                    "ok": evidence_valid,
                    "tool_name": "deterministic_handler_router",
                    "handler_name": result.handler_name,
                    "planned_handler_name": handler_name,
                    "required_handler_role": str((handler_plan or {}).get("required_handler_role") or ""),
                    "handler_plan": handler_plan or {},
                    "handler_trust": trust.to_dict(),
                    "status": result.status,
                    "output_type": result.output_type,
                    "semantic_role": result.semantic_role,
                    "supporting_inputs": list(result.supporting_inputs or []),
                    "output_text": context,
                    "raw_result": payload,
                    "missing_inputs": list(result.missing_inputs or []),
                    "next_action_hint": next_action_hint,
                    "next_capability": result.next_capability,
                    "evidence_valid": evidence_valid,
                    "error": result.error or None,
                }
            ]
        except Exception as exc:
            return "", [
                {
                    "ok": False,
                    "tool_name": "deterministic_handler_router",
                    "output_text": "",
                    "raw_result": None,
                    "error": str(exc),
                }
            ]


__all__ = ["EvidenceRunner"]
