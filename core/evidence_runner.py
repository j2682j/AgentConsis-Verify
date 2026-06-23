from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.deterministic_solver import DeterministicSolver
from tools.search_result_builder.evidence_searcher import EvidenceSearcher
from tools.system_routing_contract import SystemRoutingContract


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
        deterministic_solver: DeterministicSolver | None = None,
        compact_search_evidence: bool = False,
        enable_evidence_driven_search: bool = True,
        max_parallel_next_hop_queries: int = 2,
    ) -> None:
        self.question = question
        self.attachment = attachment or {}
        self.tool_manager = tool_manager
        self.search_result = search_result
        self.attachment_result = attachment_result
        self.routing_contract = routing_contract or SystemRoutingContract()
        self.attachment_evidence_builder = attachment_evidence_builder or AttachmentEvidenceBuilder()
        self.deterministic_solver = deterministic_solver or DeterministicSolver()
        self.compact_search_evidence = compact_search_evidence
        self.enable_evidence_driven_search = enable_evidence_driven_search
        self.max_parallel_next_hop_queries = max(0, max_parallel_next_hop_queries)

    def run(self) -> dict[str, Any]:
        """
        執行 evidence routing，並平行準備 attachment 與 search evidence。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: Stage1 可使用的 evidence 與工具使用紀錄。
        """
        routing = self._route_stage1_tools()
        tool_usage: list[dict[str, Any]] = []

        attachment_result = self._resolve_attachment_result()
        search_result = self.search_result.strip()

        evidence_tasks = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if not attachment_result and routing.get("use_attachment"):
                evidence_tasks[executor.submit(self._build_attachment_evidence)] = "attachment"
            if not search_result and routing.get("use_search"):
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

        solver_result = ""
        if routing.get("use_deterministic_solver") or routing.get("use_python_solver"):
            solver_result, solver_usage = self._build_solver_evidence(
                attachment_context=attachment_result,
            )
            tool_usage.extend(solver_usage)

        return {
            "search_result": search_result.strip(),
            "attachment_result": attachment_result.strip(),
            "solver_result": solver_result.strip(),
            "routing": routing,
            "tool_usage": tool_usage,
        }

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
            routing["provided_search_result"] = True
        if self.attachment_result:
            routing["use_attachment"] = False
            routing["provided_attachment_result"] = True
        return routing

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
            return "", [
                {
                    "ok": False,
                    "tool_name": "attachment_reader",
                    "output_text": "",
                    "raw_result": None,
                    "error": str(exc),
                }
            ]
        return str(result.get("context", "") or ""), list(result.get("tool_usage", []) or [])

    def _build_search_evidence(self) -> tuple[str, list[dict[str, Any]]]:
        """
        使用 tool_manager 執行 search，建立 Stage1 可用的外部查詢 evidence。

        Args:
            - 無。

        Returns:
            - str: search evidence 文字。
            - list[dict[str, Any]]: search tool 的執行結果紀錄。
        """
        if self.tool_manager is None or not hasattr(self.tool_manager, "execute_tool"):
            return "", [
                {
                    "ok": False,
                    "tool_name": "search",
                    "output_text": "",
                    "raw_result": None,
                    "error": "tool_manager with execute_tool is not available",
                }
            ]

        try:
            searcher = EvidenceSearcher(
                tool_manager=self.tool_manager,
                enable_evidence_driven_search=self.enable_evidence_driven_search,
                max_parallel_next_hop_queries=self.max_parallel_next_hop_queries,
            )
            output = searcher.search(
                self.question,
                max_queries=3,
                max_results_per_query=5,
                max_full_page_results=2,
                agent_id="network_shared",
                stage="stage1_evidence",
            )
            result = {
                "ok": bool(output.summary.strip()),
                "tool_name": "search",
                "output_text": output.summary,
                "raw_result": searcher.to_dict(output),
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
        return str(result.get("output_text", "") or ""), [result]

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
        try:
            result = self.deterministic_solver.solve(
                self.question,
                attachment_context=attachment_context,
            )
            payload = result.to_dict()
        except Exception as exc:
            payload = {
                "used_deterministic_solver": False,
                "answer_text": "",
                "confidence": 0.0,
                "error": str(exc),
            }

        used = bool(payload.get("used_deterministic_solver"))
        answer_text = str(payload.get("answer_text", "") or payload.get("answer", "") or "")
        confidence = payload.get("confidence", 0.0)
        context = ""
        if used and answer_text:
            context = (
                "Deterministic solver evidence:\n"
                f"Answer text: {answer_text}\n"
                f"Confidence: {confidence}\n"
                "Instruction: prefer this exact answer for deterministic closed-world tasks."
            )
        return context, [
            {
                "ok": used,
                "tool_name": "deterministic_solver",
                "output_text": context,
                "raw_result": payload,
                "error": payload.get("error"),
            }
        ]


__all__ = ["EvidenceRunner"]
