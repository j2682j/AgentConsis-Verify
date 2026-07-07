from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.search_result_builder.evidence import EvidenceConverter
from tools.search_result_builder.retrieval_control import WebRetrievalControl
from tools.system_routing_contract import SystemRoutingContract
from utils.network_utils import normalize_text, should_use_calculator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvidenceBuilder:
    """
    建立本地 Agent Network 在推理前可使用的工具證據上下文。

    Args:
        - tool_manager: 負責執行 calculator、search、deterministic_solver 等工具的管理器。
        - runtime: 可提供目前 attachment 狀態的執行期物件。

    Returns:
        - EvidenceBuilder: 可依題目 routing 組裝工具 evidence 的建構器。
    """

    def __init__(
        self,
        tool_manager: Any | None = None,
        runtime: Any | None = None,
        *,
        search_query_planner: Any | None = None,
        web_retrieval_control: Any | None = None,
        initialize_search_helpers: bool = True,
    ) -> None:
        """
        初始化 EvidenceBuilder 需要的工具、routing 與 search evidence helper。

        Args:
            - tool_manager: 負責執行工具的 ToolManager，沒有提供時會略過工具呼叫。
            - runtime: 提供 current_attachment 等執行期狀態的物件。
            - search_query_planner: 可選的搜尋 query 規劃器。
            - web_retrieval_control: 可選的 WebRetrievalControl 相容 search 主入口。
            - initialize_search_helpers: 是否在初始化時建立 search helper。

        Returns:
            - None。
        """
        self.tool_manager = tool_manager
        self.runtime = runtime
        self.search_query_planner = search_query_planner
        self.web_retrieval_control = web_retrieval_control
        self.attachment_evidence_builder = AttachmentEvidenceBuilder()
        self.system_routing_contract = SystemRoutingContract()

        if initialize_search_helpers:
            self._ensure_web_retrieval_control()

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
        根據題目與 routing 結果建立主要工具上下文。

        Args:
            - question: 原始任務問題。
            - agent_id: 呼叫工具的 Agent id，用於 trace。
            - stage: 呼叫工具所屬階段，例如 stage1。
            - router_model_name: 保留參數，目前未使用。
            - shared_search_bundle: 保留參數，目前未使用。
            - include_routed_tools: 是否根據 routing 啟用 calculator、search、solver。
            - include_attachment: 是否讀取目前 attachment。

        Returns:
            - dict[str, Any]: 包含 tool_context、各類 evidence context、tool_usage 與 routing 的結果。
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
            "best_candidate": search.get("best_candidate"),
            "deterministic_solver_result": solver.get("deterministic_solver_result"),
            "used_attachment": attachment["used"],
            "used_calculator": calc["used"],
            "used_search": search["used"],
            "used_python_solver": solver["used"],
            "routing": routing,
        }

    def should_enable_stage1_routed_tools(self, question: str) -> bool:
        """
        判斷 Stage1 是否需要啟用 routed tools。

        Args:
            - question: 原始任務問題。

        Returns:
            - bool: 若 routing contract 判定需要工具，回傳 True。
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
        依據題目與階段產生工具 routing 決策。

        Args:
            - question: 原始任務問題。
            - stage: 目前工具 routing 所屬階段。

        Returns:
            - dict[str, Any]: 包含 use_search、use_calculator、use_attachment 等欄位的 routing dict。
        """
        decision = self.system_routing_contract.route(
            question=question,
            stage=stage,
            has_attachment=self._current_attachment() is not None,
            attachment_type=self._attachment_type(),
        )
        routing = decision.to_dict()
        routing["use_calculator"] = bool(routing.get("use_calculator") or should_use_calculator(question))
        return routing

    def _empty_routing(self) -> dict[str, Any]:
        """
        建立不啟用任何工具的空 routing 結果。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 所有工具皆為 False 的 routing dict。
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
        建立未使用工具時的空工具結果。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 包含空 tool_usage、context 與 used=False 的結果。
        """
        return {"tool_usage": [], "context": "", "used": False}

    def _empty_context_result(self) -> dict[str, Any]:
        """
        建立沒有 context 的空 evidence 結果。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: 包含空 tool_usage、context 與 used=False 的結果。
        """
        return {"tool_usage": [], "context": "", "used": False}

    def _current_attachment(self) -> dict[str, Any] | None:
        """
        從 runtime 取得目前任務的 attachment metadata。

        Args:
            - 無。

        Returns:
            - dict[str, Any] | None: 若 runtime.current_attachment 存在且有效則回傳，否則回傳 None。
        """
        attachment = getattr(self.runtime, "current_attachment", None)
        return attachment if isinstance(attachment, dict) and attachment else None

    def _attachment_type(self) -> str | None:
        """
        從目前 attachment metadata 推斷附檔類型。

        Args:
            - 無。

        Returns:
            - str | None: 附檔副檔名，例如 pdf、xlsx、png；無法判斷時回傳 None。
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
        讀取目前 attachment 並建立可放入 Agent prompt 的 evidence context。

        Args:
            - question: 原始任務問題，用於讓 attachment reader 做問題導向摘要。

        Returns:
            - dict[str, Any]: 包含 attachment context、tool_usage 與 used 狀態。
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
        呼叫 python_calculator 工具建立計算證據。

        Args:
            - question: 原始任務問題或計算描述。
            - agent_id: 呼叫工具的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - dict[str, Any]: 包含 calculator context、tool_usage 與 used 狀態。
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
        呼叫 deterministic_solver 建立封閉型任務的確定性答案證據。

        Args:
            - question: 原始任務問題。
            - agent_id: 呼叫工具的 Agent id。
            - stage: 工具呼叫所屬階段。
            - attachment_context: attachment reader 產生的文字內容，供 solver 使用。

        Returns:
            - dict[str, Any]: 包含 solver context、tool_usage、solver 原始結果與 used 狀態。
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
        使用 WebRetrievalControl 建立 evidence-oriented search context。

        Args:
            - question: 原始任務問題。
            - agent_id: 呼叫 search 的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - dict[str, Any]: 包含 search context、query plan、tool_usage、候選答案與 used 狀態。
        """
        if self.tool_manager is None:
            return self._empty_tool_result()

        controller = self._ensure_web_retrieval_control()
        try:
            output = controller.run(
                question,
                output_dir=self._web_retrieval_output_dir(question),
            )
            output_dict = self._dataclass_to_dict(output)
            converter = EvidenceConverter()
            evidence_items = converter.convert_web_retrieval_output(
                output_dict,
                question=question,
            )
            context = self._render_evidence_items(evidence_items)
            query_plan = {
                **output_dict,
                "evidence_items": evidence_items,
                "evidence_conversion": self._dataclass_to_dict(
                    converter.last_diagnostics,
                ),
            }
            tool_usage = [
                {
                    "ok": bool(context.strip()),
                    "tool_name": "search",
                    "output_text": context,
                    "raw_result": query_plan,
                    "error": None,
                }
            ]
            queries = list(output_dict.get("generated_queries") or [])
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

        return {
            "tool_usage": tool_usage,
            "context": context,
            "used": bool(context),
            "queries": queries,
            "query_plan": query_plan,
            "search_runs": [],
            "best_candidate": None,
        }

    def _ensure_web_retrieval_control(self) -> Any:
        """
        建立或重用 WebRetrievalControl。

        Args:
            - 無。

        Returns:
            - Any: 可執行 evidence-oriented search 的 WebRetrievalControl。
        """
        if self.web_retrieval_control is None:
            self.web_retrieval_control = WebRetrievalControl(
                max_queries=3,
                max_results_per_query=5,
                max_pages_to_fetch=6,
                max_chunks_per_url=10,
                max_corpus_records=120,
                max_iter=3,
                top_k=10,
                min_retrieval_score=0.0,
                relative_score_margin=1.0,
                embedding_batch_size=8,
            )
        return self.web_retrieval_control

    def _web_retrieval_output_dir(self, question: str) -> Path:
        digest = hashlib.sha1(
            normalize_text(question).encode("utf-8")
        ).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        return (
            PROJECT_ROOT
            / "outputs"
            / "web_retrieval_runtime"
            / f"{digest}_{timestamp}"
        )

    def _render_evidence_items(self, evidence_items: list[dict[str, Any]]) -> str:
        lines = ["Evidence:"]
        if not evidence_items:
            lines.append("None")
            return "\n".join(lines)
        for index, item in enumerate(evidence_items, start=1):
            lines.extend(
                [
                    f"[E{index}]",
                    f"Source Title: {item.get('title') or item.get('source_id') or 'Unknown'}",
                    f"Evidence: {item.get('text', '')}",
                ]
            )
        return "\n".join(lines).strip()

    def _dataclass_to_dict(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._dataclass_to_dict(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): self._dataclass_to_dict(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in value]
        return value

    def _question_requires_web(self, question: str) -> bool:
        """
        判斷題目是否明確需要網路搜尋。

        Args:
            - question: 原始任務問題。

        Returns:
            - bool: 題目包含 website、latest、current、online 等線索時回傳 True。
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
        將多段 evidence context 合併成單一 prompt 文字。

        Args:
            - contexts: 多段可能為空的 evidence context。

        Returns:
            - str: 以空行分隔後的合併 context。
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
        從多種工具 evidence 中選出最適合作為主要 prompt context 的內容。

        Args:
            - attachment: attachment evidence 結果。
            - calc: calculator evidence 結果。
            - solver: deterministic solver evidence 結果。
            - search: search evidence 結果。

        Returns:
            - str: 優先順序為 solver、attachment、search、calculator 的主要 context。
        """
        for item in (solver, attachment, search, calc):
            if item.get("used") and str(item.get("context", "") or "").strip():
                return str(item.get("context", "") or "").strip()
        return ""
