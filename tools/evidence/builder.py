from __future__ import annotations

from typing import Any

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.system_routing_contract import SystemRoutingContract
from utils.network_utils import should_use_calculator, should_use_search


class EvidenceBuilder:
    """Build the tool evidence currently used by the local agent network."""

    def __init__(
        self,
        tool_manager: Any | None = None,
        runtime: Any | None = None,
        *,
        search_query_planner: Any | None = None,
        evidence_searcher: Any | None = None,
        initialize_search_helpers: bool = True,
    ) -> None:
        """
        ??????????????
        
        Args:
            - ????????????
        
        Returns:
            - None?
        """
        self.tool_manager = tool_manager
        self.runtime = runtime
        self.search_query_planner = search_query_planner
        self.evidence_searcher = evidence_searcher
        self.attachment_evidence_builder = AttachmentEvidenceBuilder()
        self.system_routing_contract = SystemRoutingContract()

        if initialize_search_helpers:
            self._ensure_evidence_searcher()

    def build(
        self,
        question: str,
        agent_id: str = "unknown_agent",
        stage: str = "stage1",
        router_model_name: str | None = None,
        shared_search_bundle: dict[str, Any] | None = None,
        include_routed_tools: bool = True,
        include_attachment: bool = True,
    ) -> dict[str, Any]:
        """
        ?????????????????????
        
        Args:
            - ????????????
        
        Returns:
            - ?????????
        """
        routing = (
            self._route_tools(question, stage=stage)
            if include_routed_tools
            else self._empty_routing()
        )
        attachment = self._build_attachment_evidence(question) if include_attachment else self._empty_tool_result()

        if attachment["used"] and not self._question_requires_web(question):
            routing["use_search"] = False

        calc = (
            self._build_calculator_evidence(question, agent_id, stage)
            if routing.get("use_calculator")
            else self._empty_tool_result()
        )
        solver = (
            self._build_deterministic_solver_evidence(
                question,
                agent_id,
                stage,
                attachment_context=attachment["context"],
            )
            if routing.get("use_deterministic_solver") or routing.get("use_python_solver")
            else self._empty_context_result()
        )
        if solver.get("used"):
            routing["use_search"] = False

        search = (
            self._build_search_evidence(question, agent_id, stage)
            if routing.get("use_search")
            else self._empty_tool_result()
        )

        tool_usage = []
        tool_usage.extend(calc.get("tool_usage", []))
        tool_usage.extend(solver.get("tool_usage", []))
        tool_usage.extend(search.get("tool_usage", []))
        tool_usage.extend(attachment.get("tool_usage", []))

        tool_context = self._join_contexts(
            [
                self._select_primary_tool_context(
                    attachment=attachment,
                    calc=calc,
                    solver=solver,
                    search=search,
                )
            ]
        )

        return {
            "tool_usage": tool_usage,
            "tool_context": tool_context,
            "attachment_context": attachment["context"],
            "calculator_context": calc["context"],
            "search_context": search["context"],
            "solver_context": solver["context"],
            "best_verified_candidate": search.get("best_verified_candidate"),
            "deterministic_solver_result": solver.get("deterministic_solver_result"),
            "used_attachment": attachment["used"],
            "used_calculator": calc["used"],
            "used_search": search["used"],
            "used_python_solver": solver["used"],
            "routing": routing,
        }

    def should_enable_stage1_routed_tools(self, question: str) -> bool:
        """
        ???????????????
        
        Args:
            - ????????????
        
        Returns:
            - ???????
        """
        decision = self.system_routing_contract.route(
            question=question,
            stage="stage1_round0",
            has_attachment=self._current_attachment() is not None,
            attachment_type=self._attachment_type(),
        )
        return decision.needs_routed_tool

    def _route_tools(self, question: str, *, stage: str) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        decision = self.system_routing_contract.route(
            question=question,
            stage=stage,
            has_attachment=self._current_attachment() is not None,
            attachment_type=self._attachment_type(),
        )
        routing = decision.to_dict()
        routing["use_calculator"] = bool(routing.get("use_calculator") or should_use_calculator(question))
        routing["use_search"] = bool(routing.get("use_search") or should_use_search(question))
        return routing

    def _empty_routing(self) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        return {
            "use_calculator": False,
            "use_search": False,
            "use_deterministic_solver": False,
            "use_python_solver": False,
            "use_attachment": False,
            "calculator_expression": None,
            "task_type": "manual",
            "trigger_terms": [],
            "tool_policy": {"prefer": [], "optional": [], "avoid": []},
            "routing_reasons": [],
        }

    def _empty_tool_result(self) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        return {"tool_usage": [], "context": "", "used": False}

    def _empty_context_result(self) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        return {"tool_usage": [], "context": "", "used": False}

    def _current_attachment(self) -> dict[str, Any] | None:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        attachment = getattr(self.runtime, "current_attachment", None)
        return attachment if isinstance(attachment, dict) and attachment else None

    def _attachment_type(self) -> str | None:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        attachment = self._current_attachment() or {}
        extension = str(attachment.get("extension", "") or "").strip().lower()
        if extension:
            return extension.lstrip(".")
        path = str(attachment.get("file_path", "") or attachment.get("path", "") or "")
        if "." in path:
            return path.rsplit(".", 1)[-1].lower()
        return None

    def _build_attachment_evidence(self, question: str) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        attachment = self._current_attachment()
        if not attachment:
            return self._empty_tool_result()
        result = self.attachment_evidence_builder.build(question, attachment)
        return {
            "tool_usage": result.get("tool_usage", []),
            "context": result.get("context", ""),
            "used": bool(result.get("used")),
        }

    def _build_calculator_evidence(self, question: str, agent_id: str, stage: str) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if self.tool_manager is None:
            return self._empty_tool_result()
        try:
            result = self.tool_manager.execute_tool(
                "python_calculator",
                {"input": question},
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": "python_calculator",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }
        output = str(result.get("output_text", "") or "").strip()
        return {
            "tool_usage": [result],
            "context": f"Calculator evidence:\n{output}" if output else "",
            "used": bool(result.get("ok") and output),
        }

    def _build_deterministic_solver_evidence(
        self,
        question: str,
        agent_id: str,
        stage: str,
        *,
        attachment_context: str = "",
    ) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if self.tool_manager is None:
            return self._empty_context_result()
        try:
            result = self.tool_manager.execute_tool(
                "deterministic_solver",
                {"input": question, "attachment_context": attachment_context},
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": "deterministic_solver",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }

        parsed = result.get("raw_result") if isinstance(result.get("raw_result"), dict) else {}
        answer_text = str(parsed.get("answer_text", "") or parsed.get("answer", "") or "").strip()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        used = bool(parsed.get("used_deterministic_solver") and answer_text)
        context = ""
        if used:
            context = (
                "Deterministic solver evidence:\n"
                f"Answer text: {answer_text}\n"
                f"Confidence: {confidence}\n"
                "Instruction: use Answer text exactly for deterministic closed-world tasks."
            )
        return {
            "tool_usage": [result],
            "context": context,
            "used": used,
            "deterministic_solver_result": parsed or None,
        }

    def _build_search_evidence(self, question: str, agent_id: str, stage: str) -> dict[str, Any]:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        if self.tool_manager is None:
            return self._empty_tool_result()

        searcher = self._ensure_evidence_searcher()
        try:
            output = searcher.search(
                question,
                max_queries=3,
                max_results_per_query=5,
                max_full_page_results=2,
                agent_id=agent_id,
                stage=stage,
            )
            context = output.summary
            tool_usage = output.tool_usage
            query_plan = searcher.to_dict(output)
            queries = [query.query for query in output.queries]
            best_verified_candidate = output.candidates[0].__dict__ if output.candidates else None
        except Exception as exc:
            context = ""
            tool_usage = [
                {
                    "ok": False,
                    "tool_name": "search",
                    "output_text": "",
                    "raw_result": None,
                    "error": str(exc),
                }
            ]
            query_plan = {}
            queries = []
            best_verified_candidate = None

        return {
            "tool_usage": tool_usage,
            "context": context,
            "used": bool(context),
            "queries": queries,
            "query_plan": query_plan,
            "search_runs": [],
            "best_verified_candidate": best_verified_candidate,
        }

    def _ensure_evidence_searcher(self) -> Any:
        """
        建立或重用新版 EvidenceSearcher。

        Args:
            - 無。

        Returns:
            - Any: 可執行 evidence-oriented search 的 EvidenceSearcher。
        """
        if self.evidence_searcher is None:
            from tools.search_result_builder.evidence_searcher import EvidenceSearcher

            self.evidence_searcher = EvidenceSearcher(
                tool_manager=self.tool_manager,
                query_planner=self.search_query_planner,
            )
        return self.evidence_searcher

    def _question_requires_web(self, question: str) -> bool:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        normalized = str(question or "").lower()
        return any(
            marker in normalized
            for marker in (
                "website",
                "web site",
                "webpage",
                "url",
                "internet",
                "search",
                "latest",
                "current",
                "today",
                "online",
            )
        )

    def _join_contexts(self, contexts: list[str]) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        valid = [ctx for ctx in contexts if ctx and ctx.strip()]
        return "\n\n".join(valid).strip()

    def _select_primary_tool_context(
        self,
        *,
        attachment: dict[str, Any],
        calc: dict[str, Any],
        solver: dict[str, Any],
        search: dict[str, Any],
    ) -> str:
        """
        ????????????
        
        Args:
            - ????????????
        
        Returns:
            - ????????????
        """
        for item in (solver, attachment, search, calc):
            if item.get("used") and str(item.get("context", "") or "").strip():
                return str(item.get("context", "") or "").strip()
        return ""
