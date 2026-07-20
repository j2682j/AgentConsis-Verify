from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from tools.evidence.fact_extraction import TaskFactStore
from utils.network_utils import normalize_text


class EvidenceReadinessStatus(str, Enum):
    """描述目前證據是否足以進入回答階段。"""

    PARSED = "parsed"
    PARTIAL = "partial"
    DIRECT_SUPPORTED = "direct_supported"
    NEEDS_EXTERNAL = "needs_external"
    FAILED = "failed"


@dataclass(frozen=True)
class EvidenceReadiness:
    """
    保存證據準備完成度與下一個需要執行的能力。

    Args:
     - status: 目前證據完成狀態。
     - known_information: 已成功取得的資訊類型。
     - missing_information: 尚未取得的必要資訊。
     - next_capability: 下一步應使用的能力，例如 search 或 attachment。
     - direct_fact_ids: 可直接驗證答案的 fact ids。
     - source_tools: 已成功提供資訊的工具。
     - reason: 狀態判定原因。

    Returns:
     - EvidenceReadiness: 可供 EvidenceRunner 決定是否切換工具的狀態。
    """

    status: EvidenceReadinessStatus
    known_information: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    next_capability: str = ""
    direct_fact_ids: list[str] = field(default_factory=list)
    source_tools: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_sufficient(self) -> bool:
        return self.status == EvidenceReadinessStatus.DIRECT_SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["is_sufficient"] = self.is_sufficient
        return payload


class EvidenceReadinessEvaluator:
    """
    依據 Fact Store 與工具契約判斷證據完成度，不使用加權分數。

    Args:
     - None.

    Returns:
     - EvidenceReadinessEvaluator: 證據狀態判定器。
    """

    _SEARCH_INPUTS = {
        "source_text",
        "date_values",
        "numbers",
        "matching_text",
        "connected_path",
        "external_fact",
        "web_source",
    }
    _ATTACHMENT_INPUTS = {
        "table_rows",
        "grid",
        "candidate_words",
        "edges",
        "list_items",
        "quoted_or_inline_text",
        "two_coordinate_pairs",
        "attachment_content",
    }

    def evaluate(
        self,
        *,
        fact_store: TaskFactStore,
        tool_usage: list[dict[str, Any]],
        routing: dict[str, Any] | None = None,
        attachment_result: str = "",
        search_result: str = "",
        solver_result: str = "",
    ) -> EvidenceReadiness:
        routing = dict(routing or {})
        direct_facts = fact_store.verifiable_answer_facts()
        trusted_handler_ids = self._trusted_handler_finals(tool_usage)
        direct_ids = [fact.fact_id for fact in direct_facts] + trusted_handler_ids
        sources = self._successful_tools(tool_usage)
        known = self._known_information(
            attachment_result=attachment_result,
            search_result=search_result,
            solver_result=solver_result,
            source_tools=sources,
        )

        if direct_ids:
            return EvidenceReadiness(
                status=EvidenceReadinessStatus.DIRECT_SUPPORTED,
                known_information=known,
                direct_fact_ids=self._dedupe(direct_ids),
                source_tools=sources,
                reason="verifiable_direct_fact_or_trusted_handler_final",
            )

        missing = self._missing_inputs(tool_usage)
        next_capability = self._explicit_next_capability(tool_usage, routing)
        if not next_capability:
            next_capability = self._infer_next_capability(missing)

        if next_capability:
            return EvidenceReadiness(
                status=EvidenceReadinessStatus.NEEDS_EXTERNAL,
                known_information=known,
                missing_information=missing,
                next_capability=next_capability,
                source_tools=sources,
                reason="explicit_missing_information_requires_capability",
            )

        search_policy = normalize_text(routing.get("search_policy")).lower()
        primary_route = normalize_text(routing.get("primary_route")).lower()
        if (
            not known
            and search_policy == "fallback"
            and primary_route == "unknown"
        ):
            return EvidenceReadiness(
                status=EvidenceReadinessStatus.NEEDS_EXTERNAL,
                missing_information=["answer_support"],
                next_capability="search",
                source_tools=sources,
                reason="unknown_route_without_usable_evidence",
            )

        if known:
            return EvidenceReadiness(
                status=EvidenceReadinessStatus.PARTIAL,
                known_information=known,
                missing_information=missing,
                source_tools=sources,
                reason="content_available_but_no_direct_answer_support",
            )

        if any(not bool(item.get("ok")) for item in tool_usage):
            return EvidenceReadiness(
                status=EvidenceReadinessStatus.FAILED,
                missing_information=missing,
                source_tools=sources,
                reason="tools_failed_without_usable_evidence",
            )

        return EvidenceReadiness(
            status=EvidenceReadinessStatus.PARSED,
            source_tools=sources,
            reason="routing_completed_without_direct_evidence",
        )

    def _trusted_handler_finals(self, tool_usage: list[dict[str, Any]]) -> list[str]:
        result: list[str] = []
        for index, item in enumerate(tool_usage, start=1):
            if item.get("tool_name") not in {
                "deterministic_handler_router",
                "attachment_strategy_handler",
            }:
                continue
            trust = item.get("handler_trust")
            trust = trust if isinstance(trust, dict) else {}
            if (
                item.get("ok")
                and item.get("evidence_valid")
                and item.get("output_type") == "final_answer"
                and trust.get("trusted", True)
            ):
                handler_name = normalize_text(item.get("handler_name")) or "handler"
                result.append(f"trusted:{handler_name}:{index}")
        return result

    def _successful_tools(self, tool_usage: list[dict[str, Any]]) -> list[str]:
        return self._dedupe(
            normalize_text(item.get("tool_name"))
            for item in tool_usage
            if item.get("ok") and normalize_text(item.get("tool_name"))
        )

    def _known_information(
        self,
        *,
        attachment_result: str,
        search_result: str,
        solver_result: str,
        source_tools: list[str],
    ) -> list[str]:
        known: list[str] = []
        if normalize_text(attachment_result):
            known.append("attachment_content")
        if normalize_text(search_result):
            known.append("search_content")
        if normalize_text(solver_result):
            known.append("solver_output")
        known.extend(f"tool:{name}" for name in source_tools)
        return self._dedupe(known)

    def _missing_inputs(self, tool_usage: list[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        for item in tool_usage:
            raw = item.get("raw_result")
            raw = raw if isinstance(raw, dict) else {}
            for value in list(item.get("missing_inputs") or raw.get("missing_inputs") or []):
                cleaned = normalize_text(value)
                if cleaned:
                    values.append(cleaned)
        return self._dedupe(values)

    def _explicit_next_capability(
        self,
        tool_usage: list[dict[str, Any]],
        routing: dict[str, Any],
    ) -> str:
        for item in reversed(tool_usage):
            raw = item.get("raw_result")
            raw = raw if isinstance(raw, dict) else {}
            value = normalize_text(
                item.get("next_capability") or raw.get("next_capability")
            ).lower()
            if value:
                return value
        loop = routing.get("attachment_strategy_loop")
        if isinstance(loop, dict):
            value = normalize_text(loop.get("next_capability")).lower()
            if value:
                return value
            if loop.get("needs_search"):
                return "search"
        return ""

    def _infer_next_capability(self, missing: list[str]) -> str:
        values = {normalize_text(value).lower() for value in missing}
        if values & self._SEARCH_INPUTS:
            return "search"
        if values & self._ATTACHMENT_INPUTS:
            return "attachment"
        if any(value.startswith("handler:") for value in values):
            return "agent"
        return ""

    @staticmethod
    def _dedupe(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text(value)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output


__all__ = [
    "EvidenceReadiness",
    "EvidenceReadinessEvaluator",
    "EvidenceReadinessStatus",
]
