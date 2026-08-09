from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from context.context_builder import ContextPacket


@dataclass
class SearchGateDecision:
    """Describe whether one Stage1 search request may reach the search backend."""

    allowed: bool
    reason: str
    query: str
    missing_information: str = ""
    use_existing_evidence: bool = False
    cached_result: dict[str, Any] | None = None


@dataclass
class Stage1SearchAccessState:
    """Maintain task-scoped Stage1 search access and supplemental evidence."""

    prepared_status: str = "not_prepared"
    prepared_evidence_available: bool = False
    prepared_queries: list[str] = field(default_factory=list)
    prepared_evidence_ids: list[str] = field(default_factory=list)
    refinement_budget: int = 1
    # Refinement searches every Agent may make before the shared budget applies.
    # The budget is task-scoped but requested by 3 Agents x 3 runs, so without a
    # floor it is simply whoever asks first: on level1_final_08 it was exhausted
    # on 24 of 53 tasks, gemma was refused 93 times and granted none, and every
    # refusal costs one of a run's few tool turns. Losing them all is what pushed
    # gemma through the tool-less repair prompt on 100% of its runs. A floor
    # grant does not draw on the shared budget, so the task ceiling is
    # per_agent_floor * agents + refinement_budget. Set to 0 to restore a purely
    # shared budget.
    per_agent_refinement_floor: int = 1
    direct_evidence_available: bool = False
    supplemental_evidence_max_items: int = 3
    request_count: int = 0
    physical_execution_count: int = 0
    cache_hit_count: int = 0
    refinement_used: int = 0
    shared_refinement_used: int = 0
    agent_refinement_used: dict[str, int] = field(default_factory=dict)
    executed_queries: list[str] = field(default_factory=list)
    blocked_requests: list[dict[str, Any]] = field(default_factory=list)
    _prepared_query_keys: set[str] = field(default_factory=set, repr=False)
    _reserved_query_keys: set[str] = field(default_factory=set, repr=False)
    _supplemental_results: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _supplemental_packets: list[ContextPacket] = field(default_factory=list, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def from_evidence(
        cls,
        evidence: dict[str, Any],
        *,
        refinement_budget: int = 1,
        supplemental_evidence_max_items: int = 3,
        per_agent_refinement_floor: int = 1,
    ) -> "Stage1SearchAccessState":
        search_result = str(evidence.get("search_result", "") or "").strip()
        routing = evidence.get("routing") if isinstance(evidence.get("routing"), dict) else {}
        tool_usage = [
            item
            for item in list(evidence.get("tool_usage") or [])
            if isinstance(item, dict)
        ]
        search_usage = [
            item for item in tool_usage if str(item.get("tool_name", "") or "") == "search"
        ]
        valid_search_usage = any(
            bool(item.get("evidence_valid"))
            and bool(
                str(item.get("output_text", "") or "").strip()
                or cls._raw_evidence_items(item.get("raw_result"))
            )
            for item in search_usage
        )
        provided_search = bool(routing.get("provided_search_result")) or bool(
            search_result and not search_usage
        )
        prepared_available = bool(search_result and (valid_search_usage or provided_search))
        if prepared_available:
            prepared_status = "prepared_usable"
        elif search_usage or routing.get("use_search") or routing.get("search_decision"):
            prepared_status = "prepared_failed"
        else:
            prepared_status = "not_prepared"

        prepared_queries: list[str] = []
        evidence_ids: list[str] = []
        for item in search_usage:
            raw = item.get("raw_result")
            prepared_queries.extend(cls._raw_queries(raw))
            evidence_ids.extend(cls._raw_evidence_ids(raw))

        direct_evidence_available = any(
            bool(item.get("trusted"))
            and bool(item.get("evidence_valid"))
            and str(item.get("output_type", "") or "") == "final_answer"
            for item in tool_usage
        )
        state = cls(
            prepared_status=prepared_status,
            prepared_evidence_available=prepared_available,
            prepared_queries=cls._unique_text(prepared_queries),
            prepared_evidence_ids=cls._unique_text(evidence_ids),
            refinement_budget=int(refinement_budget),
            per_agent_refinement_floor=max(0, int(per_agent_refinement_floor)),
            direct_evidence_available=direct_evidence_available,
            supplemental_evidence_max_items=max(1, int(supplemental_evidence_max_items)),
        )
        state._prepared_query_keys = {
            state.normalize_query(query) for query in state.prepared_queries if query
        }
        return state

    @property
    def enabled(self) -> bool:
        return self.refinement_budget >= 0

    def authorize(
        self,
        *,
        query: str,
        missing_information: str,
        agent_id: str = "",
    ) -> SearchGateDecision:
        query = str(query or "").strip()
        missing_information = str(missing_information or "").strip()
        query_key = self.normalize_query(query)
        agent_key = str(agent_id or "").strip().casefold()
        with self._lock:
            self.request_count += 1
            if not self.enabled or (
                not self.prepared_evidence_available and not self.direct_evidence_available
            ):
                if query_key:
                    self._reserved_query_keys.add(query_key)
                    self.executed_queries.append(query)
                return SearchGateDecision(True, "legacy_search_access", query, missing_information)

            if self.direct_evidence_available:
                return self._block(
                    reason="trusted_direct_evidence_available",
                    query=query,
                    missing_information=missing_information,
                )
            if not missing_information:
                return self._block(
                    reason="missing_information_required",
                    query=query,
                    missing_information=missing_information,
                )
            if not query_key:
                return self._block(
                    reason="empty_refinement_query",
                    query=query,
                    missing_information=missing_information,
                )
            if query_key in self._prepared_query_keys:
                return self._block(
                    reason="query_already_prepared",
                    query=query,
                    missing_information=missing_information,
                    use_existing_evidence=True,
                )
            cached = self._supplemental_results.get(query_key)
            if cached is not None:
                self.cache_hit_count += 1
                return SearchGateDecision(
                    allowed=False,
                    reason="shared_stage1_search_result",
                    query=query,
                    missing_information=missing_information,
                    use_existing_evidence=True,
                    cached_result=self._cached_result(cached),
                )
            if query_key in self._reserved_query_keys:
                return self._block(
                    reason="query_inflight",
                    query=query,
                    missing_information=missing_information,
                )
            agent_used = self.agent_refinement_used.get(agent_key, 0)
            # A budget of 0 disables refinement outright, so the per-agent floor
            # has nothing to apportion: it shares out an enabled budget fairly,
            # it does not override a closed one.
            within_floor = (
                bool(agent_key)
                and self.refinement_budget > 0
                and agent_used < self.per_agent_refinement_floor
            )
            within_shared = self.shared_refinement_used < self.refinement_budget
            if not within_floor and not within_shared:
                return self._block(
                    reason="refinement_budget_exhausted",
                    query=query,
                    missing_information=missing_information,
                )

            if not within_floor:
                self.shared_refinement_used += 1
            if agent_key:
                self.agent_refinement_used[agent_key] = agent_used + 1
            self.refinement_used += 1
            self._reserved_query_keys.add(query_key)
            self.executed_queries.append(query)
            return SearchGateDecision(
                True,
                "refinement_allowed_by_floor" if within_floor else "refinement_allowed",
                query,
                missing_information,
            )

    def complete(self, *, query: str, result: dict[str, Any]) -> None:
        query_key = self.normalize_query(query)
        if not query_key:
            return
        with self._lock:
            self._reserved_query_keys.discard(query_key)
            stored = dict(result or {})
            self._supplemental_results[query_key] = stored
            if not stored.get("cache_hit"):
                self.physical_execution_count += 1
            if not stored.get("evidence_valid"):
                return
            output = str(stored.get("output_text", "") or "").strip()
            if not output or len(self._supplemental_packets) >= self.supplemental_evidence_max_items:
                return
            self._supplemental_packets.append(
                ContextPacket(
                    packet_type="search_result",
                    content=output,
                    priority=75,
                    metadata={
                        "source": "stage1_tool_use",
                        "query": query,
                    },
                )
            )

    def supplemental_packets(self) -> list[ContextPacket]:
        with self._lock:
            return list(self._supplemental_packets)

    def format_prompt(self) -> str:
        with self._lock:
            if not self.enabled:
                return "Mode: unrestricted (Stage1 search gate disabled)."
            if self.direct_evidence_available:
                return (
                    "Mode: search locked.\n"
                    "Reason: trusted direct evidence is already available.\n"
                    "Instruction: use the existing evidence or another non-search tool."
                )
            if not self.prepared_evidence_available:
                return "Mode: normal search access; no usable prepared search evidence is available."
            remaining = max(0, self.refinement_budget - self.shared_refinement_used)
            evidence_ids = ", ".join(self.prepared_evidence_ids) or "available in Search_Result"
            return (
                "Mode: refinement only.\n"
                "Prepared search evidence: available.\n"
                f"Prepared evidence IDs: {evidence_ids}.\n"
                f"Shared refinement searches remaining: {remaining}.\n"
                f"Reserved for each agent: {self.per_agent_refinement_floor}.\n"
                "Instruction: use Search_Result first; request search only for one specific missing fact."
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            reason_counts: dict[str, int] = {}
            for item in self.blocked_requests:
                reason = str(item.get("reason", "") or "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            return {
                "prepared_status": self.prepared_status,
                "prepared_evidence_available": self.prepared_evidence_available,
                "prepared_queries": list(self.prepared_queries),
                "prepared_evidence_ids": list(self.prepared_evidence_ids),
                "direct_evidence_available": self.direct_evidence_available,
                "enabled": self.enabled,
                "refinement_budget": self.refinement_budget,
                "refinement_used": self.refinement_used,
                "per_agent_refinement_floor": self.per_agent_refinement_floor,
                "shared_refinement_used": self.shared_refinement_used,
                "agent_refinement_used": dict(self.agent_refinement_used),
                "refinement_remaining": (
                    -1
                    if not self.enabled
                    else max(0, self.refinement_budget - self.shared_refinement_used)
                ),
                "request_count": self.request_count,
                "physical_execution_count": self.physical_execution_count,
                "cache_hit_count": self.cache_hit_count,
                "blocked_count": len(self.blocked_requests),
                "blocked_reasons": reason_counts,
                "executed_queries": list(self.executed_queries),
                "supplemental_evidence_count": len(self._supplemental_packets),
            }

    def blocked_result(self, decision: SearchGateDecision) -> dict[str, Any]:
        if decision.cached_result is not None:
            return dict(decision.cached_result)
        retry_hint = self._retry_hint(decision)
        return {
            "ok": False,
            "tool_name": "search",
            "status": "already_available" if decision.use_existing_evidence else "search_restricted",
            "output_text": "",
            "raw_result": {"stage1_search_gate": self.snapshot()},
            "error_code": decision.reason,
            "error_message": f"search request blocked: {decision.reason}. {retry_hint}",
            "error": f"search request blocked: {decision.reason}",
            "retryable": False,
            "retry_hint": retry_hint,
            "evidence_valid": bool(decision.use_existing_evidence),
            "cache_hit": False,
            "duplicate_request": bool(decision.use_existing_evidence),
            "search_gate_reason": decision.reason,
        }

    @staticmethod
    def _retry_hint(decision: SearchGateDecision) -> str:
        if decision.reason == "missing_information_required":
            return (
                'Retry the same search now with tool_args including exactly this key: '
                '"missing_information": "<the one specific fact still missing>". '
                "Do not resend the request without that key."
            )
        if decision.reason == "empty_refinement_query":
            return (
                'Retry with a non-empty "input" query string, and include '
                '"missing_information": "<the one specific fact still missing>" in tool_args.'
            )
        if decision.reason in {"query_already_prepared", "shared_stage1_search_result"}:
            return "This query is already covered by Search_Result; read it instead of searching again."
        if decision.reason == "trusted_direct_evidence_available":
            return "A trusted answer is already available; use it or a non-search tool instead of searching."
        if decision.reason == "refinement_budget_exhausted":
            return "No search refinements remain; answer from the existing evidence or use a non-search tool."
        if decision.reason == "query_inflight":
            return "That query is already running; wait for its result or use a non-search tool."
        return "Use the prepared evidence or choose a non-search tool."

    def _block(
        self,
        *,
        reason: str,
        query: str,
        missing_information: str,
        use_existing_evidence: bool = False,
    ) -> SearchGateDecision:
        self.blocked_requests.append(
            {
                "reason": reason,
                "query": query,
                "missing_information": missing_information,
            }
        )
        return SearchGateDecision(
            allowed=False,
            reason=reason,
            query=query,
            missing_information=missing_information,
            use_existing_evidence=use_existing_evidence,
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        text = unicodedata.normalize("NFKC", str(query or "")).casefold()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    @staticmethod
    def _cached_result(result: dict[str, Any]) -> dict[str, Any]:
        cached = dict(result)
        cached.update(
            {
                "status": "already_available",
                "cache_hit": True,
                "duplicate_request": True,
                "retryable": False,
                "retry_hint": "Use the shared Stage1 search result; do not search again.",
                "search_gate_reason": "shared_stage1_search_result",
            }
        )
        return cached

    @staticmethod
    def _raw_evidence_items(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, dict):
            return []
        return [item for item in list(raw.get("evidence_items") or []) if isinstance(item, dict)]

    @classmethod
    def _raw_evidence_ids(cls, raw: Any) -> list[str]:
        values = []
        for item in cls._raw_evidence_items(raw):
            value = str(item.get("evidence_id") or item.get("source_id") or item.get("id") or "").strip()
            if value:
                values.append(value)
        return values

    @staticmethod
    def _raw_queries(raw: Any) -> list[str]:
        if not isinstance(raw, dict):
            return []
        values = [str(item).strip() for item in list(raw.get("queries") or []) if str(item).strip()]
        retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        values.extend(
            str(item).strip()
            for item in list(retrieval.get("searched_queries") or [])
            if str(item).strip()
        )
        diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
        control = (
            diagnostics.get("web_retrieval_control")
            if isinstance(diagnostics.get("web_retrieval_control"), dict)
            else {}
        )
        values.extend(
            str(item).strip()
            for item in list(control.get("searched_queries") or [])
            if str(item).strip()
        )
        return Stage1SearchAccessState._unique_text(values)

    @staticmethod
    def _unique_text(values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            output.append(text)
        return output


__all__ = ["SearchGateDecision", "Stage1SearchAccessState"]
