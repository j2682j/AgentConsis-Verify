from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import ToolRegistry
from .tool_gap_detector import ToolGapDetector
from .tool_result import ToolExecutionResult, failure_result


class ToolManager:
    """
    管理本地工具註冊、啟用狀態、執行結果正規化與工具使用 trace。

    Args:
        - 無。

    Returns:
        - ToolManager: 可供 EvidenceRunner 與 Stage1ToolUseRunner 呼叫的工具管理器。
    """

    def __init__(self) -> None:
        """
        初始化工具 registry、啟用工具集合與 trace 紀錄，並註冊預設工具。

        Args:
            - 無。

        Returns:
            - None。
        """
        self.registry = ToolRegistry()
        self.tools: dict[str, Any] = {}
        self.enabled_tools: set[str] = set()
        self.tool_traces: list[dict[str, Any]] = []
        self.register_default_tools()
        self.gap_detector = ToolGapDetector(self.registry)

    def register_tool(self, tool: Any, auto_expand: bool = True) -> None:
        """
        註冊單一工具到 ToolManager 與底層 ToolRegistry。

        Args:
            - tool: 具有 name 與 run 方法的工具物件。
            - auto_expand: 是否允許 registry 展開 expandable tool。

        Returns:
            - None。
        """
        self.tools[tool.name] = tool
        self.registry.register_tool(tool, auto_expand=auto_expand)

    def register_default_tools(self) -> None:
        """
        註冊系統預設工具，例如 calculator、deterministic_solver 與 search。

        Args:
            - 無。

        Returns:
            - None。
        """
        from .calculator import CalculatorTool
        from .attachment_reader_tool import AttachmentReaderTool
        from .deterministic_solver_tool import DeterministicSolverTool
        from .video_evidence_tool import VideoEvidenceTool
        from .video_transcript_tool import VideoTranscriptTool

        calculator = CalculatorTool()
        self.register_tool(calculator)
        self.enabled_tools.add(calculator.name)

        attachment_reader = AttachmentReaderTool()
        self.register_tool(attachment_reader)
        self.enabled_tools.add(attachment_reader.name)

        deterministic_solver = DeterministicSolverTool()
        self.register_tool(deterministic_solver)
        self.enabled_tools.add(deterministic_solver.name)

        video_evidence = VideoEvidenceTool()
        self.register_tool(video_evidence)
        self.enabled_tools.add(video_evidence.name)

        video_transcript = VideoTranscriptTool()
        self.register_tool(video_transcript)
        self.enabled_tools.add(video_transcript.name)

        try:
            from .search_tool import SearchTool

            search_tool = SearchTool()
            self.register_tool(search_tool)
            self.enabled_tools.add(search_tool.name)
        except Exception as exc:
            print(f"[WARN] SearchTool initialization failed: {exc}")

    def set_enabled_tools(self, tool_names: list[str] | set[str]) -> None:
        """
        設定目前允許執行的工具名稱集合。

        Args:
            - tool_names: 要啟用的工具名稱清單或集合。

        Returns:
            - None。
        """
        self.enabled_tools = set(tool_names)

    def is_tool_enabled(self, tool_name: str) -> bool:
        return tool_name in self.enabled_tools and self.registry.get_tool(tool_name) is not None

    def describe_enabled_tools(self) -> str:
        lines: list[str] = []
        for tool_name in sorted(self.enabled_tools):
            tool = self.registry.get_tool(tool_name)
            if tool is None:
                continue
            parameters = ", ".join(
                f"{item.name}:{item.type}{'*' if item.required else ''}"
                for item in tool.get_parameters()
            )
            capabilities = ", ".join(sorted(tool.capabilities)) or "none"
            lines.append(
                f"- {tool.name}: {tool.description} "
                f"capabilities=[{capabilities}] args=[{parameters or 'none'}]"
            )
        return "\n".join(lines) if lines else "No enabled tools."

    def detect_tool_gap(
        self,
        question: str,
        *,
        attachment_type: str | None = None,
        requested_tool_name: str = "",
    ) -> dict[str, Any]:
        return self.gap_detector.detect(
            question,
            attachment_type=attachment_type,
            enabled_tools=self.enabled_tools,
            requested_tool_name=requested_tool_name,
        ).to_dict()

    def format_tool_gap(
        self,
        question: str,
        *,
        attachment_type: str | None = None,
    ) -> str:
        report = self.detect_tool_gap(question, attachment_type=attachment_type)
        required = [
            item.get("capability", "")
            for item in report.get("required", [])
            if item.get("capability")
        ]
        matched = report.get("matched", {})
        missing = report.get("missing", [])
        return (
            f"Required capabilities: {required or ['none']}\n"
            f"Matched capabilities: {matched or {}}\n"
            f"Missing capabilities: {missing or ['none']}"
        )

    def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        agent_id: str | None = None,
        stage: str | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        執行指定工具並回傳標準化工具結果，同時記錄 tool trace。

        Args:
            - tool_name: 要執行的工具名稱。
            - parameters: 傳給工具的參數。
            - agent_id: 發起工具呼叫的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - dict[str, Any]: 包含 ok、tool_name、output_text、raw_result 與 error 的工具結果。
        """
        tool_name, parameters = self._reroute_local_media_tool(tool_name, parameters)
        if tool_name not in self.enabled_tools:
            gap = self.detect_tool_gap(
                str(parameters.get("input") or parameters.get("question") or ""),
                requested_tool_name=tool_name,
            )
            result = failure_result(
                tool_name,
                status="unsupported",
                error_code="tool_not_enabled",
                error_message=f"tool '{tool_name}' is not enabled",
                retry_hint="Choose a tool listed as enabled.",
                raw_result={"tool_gap": gap},
            )
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        tool = self.tools.get(tool_name) or self.registry.get_tool(tool_name)
        if tool is None:
            gap = self.detect_tool_gap(
                str(parameters.get("input") or parameters.get("question") or ""),
                requested_tool_name=tool_name,
            )
            result = failure_result(
                tool_name,
                status="unsupported",
                error_code="tool_not_found",
                error_message=f"tool '{tool_name}' not found",
                retry_hint="Choose a registered tool or report a missing capability.",
                raw_result={"tool_gap": gap},
            )
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        try:
            contextual_runner = getattr(tool, "run_with_context", None)
            if callable(contextual_runner) and runtime_context:
                raw = contextual_runner(parameters, runtime_context)
            else:
                raw = tool.run(parameters)
            result = self.normalize_result(tool_name, raw)
        except Exception as exc:
            result = failure_result(
                tool_name,
                status="retryable_failure",
                error_code="tool_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                retryable=True,
                retry_hint="Retry only after changing the input or selecting another tool.",
            )

        self._record_trace(tool_name, parameters, result, agent_id, stage)
        return result

    _LOCAL_MEDIA_EXTENSIONS = {
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
    }

    def _reroute_local_media_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if str(tool_name or "") not in {"video_transcript", "video_evidence"}:
            return tool_name, parameters

        params = dict(parameters or {})
        attachment = params.get("attachment")
        if isinstance(attachment, dict):
            extension = self._attachment_extension(attachment)
            if extension in self._LOCAL_MEDIA_EXTENSIONS:
                routed = dict(params)
                routed.setdefault("file_path", attachment.get("file_path") or attachment.get("path"))
                routed.setdefault(
                    "question",
                    params.get("question") or params.get("input") or params.get("query") or "",
                )
                routed["attachment"] = dict(attachment)
                routed["rerouted_from"] = str(tool_name or "")
                return "attachment_reader", routed

        value = str(
            params.get("file_path")
            or params.get("path")
            or params.get("url")
            or params.get("input")
            or params.get("query")
            or ""
        ).strip()
        if self._looks_like_local_media_path(value):
            routed = dict(params)
            routed.setdefault("file_path", value)
            routed.setdefault(
                "question",
                params.get("question") or params.get("input") or params.get("query") or "",
            )
            routed["rerouted_from"] = str(tool_name or "")
            return "attachment_reader", routed

        return tool_name, parameters

    def _looks_like_local_media_path(self, value: str) -> bool:
        candidate = str(value or "").strip()
        if not candidate:
            return False
        if candidate.lower().startswith(("http://", "https://")):
            return False
        return Path(candidate).suffix.lower() in self._LOCAL_MEDIA_EXTENSIONS

    def _attachment_extension(self, attachment: dict[str, Any]) -> str:
        extension = str(attachment.get("extension", "") or "").strip().lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if extension:
            return extension
        file_path = str(attachment.get("file_path", "") or attachment.get("path", "") or "")
        return Path(file_path).suffix.lower()

    def normalize_result(self, tool_name: str, raw_result: Any) -> dict[str, Any]:
        """
        將工具原始輸出轉成系統統一的工具結果格式。

        Args:
            - tool_name: 工具名稱。
            - raw_result: 工具原始回傳值。

        Returns:
            - dict[str, Any]: 標準化後的工具結果。
        """
        if isinstance(raw_result, dict):
            return self._normalize_dict_result(tool_name, raw_result)
        output_text = str(raw_result or "").strip()
        return ToolExecutionResult(
            ok=bool(output_text),
            tool_name=tool_name,
            status="success" if output_text else "partial",
            output_text=output_text,
            raw_result=raw_result,
            evidence_valid=bool(output_text),
        ).to_dict()

    def _normalize_dict_result(
        self,
        tool_name: str,
        raw_result: dict[str, Any],
    ) -> dict[str, Any]:
        if {"status", "ok", "evidence_valid"}.issubset(raw_result):
            result = dict(raw_result)
            result.setdefault("tool_name", tool_name)
            result.setdefault("output_text", str(raw_result.get("raw_result", "") or ""))
            result.setdefault("error_message", str(result.get("error", "") or ""))
            result["error"] = result.get("error_message") or None
            return result

        if tool_name == "attachment_reader":
            return self._normalize_attachment_result(raw_result)
        if tool_name == "deterministic_solver":
            return self._normalize_solver_result(raw_result)
        if tool_name == "search":
            return self._normalize_search_result(raw_result)

        output_text = str(raw_result)
        return ToolExecutionResult(
            ok=True,
            tool_name=tool_name,
            status="success",
            output_text=output_text,
            raw_result=raw_result,
            evidence_valid=bool(output_text.strip()),
        ).to_dict()

    def _normalize_attachment_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        context = str(raw_result.get("context", "") or "").strip()
        nested_usage = [
            item for item in raw_result.get("tool_usage", []) or [] if isinstance(item, dict)
        ]
        nested_failures = [item for item in nested_usage if not item.get("ok", False)]
        metadata = raw_result.get("metadata") if isinstance(raw_result.get("metadata"), dict) else {}
        reader = str(metadata.get("reader", "") or "")
        valid_context = bool(
            context
            and context.lower() not in {"none", "extracted content:\nnone"}
            and reader != "error_reader"
        )
        if nested_failures or not valid_context:
            messages = [
                str(item.get("error", "") or "").strip()
                for item in nested_failures
                if str(item.get("error", "") or "").strip()
            ]
            error_message = "; ".join(messages) or "attachment produced no usable evidence"
            return failure_result(
                "attachment_reader",
                status="retryable_failure",
                error_code="attachment_read_failed",
                error_message=error_message,
                retryable=True,
                retry_hint="Use another compatible attachment reader or reduce media input size.",
                raw_result=raw_result,
            )
        return ToolExecutionResult(
            ok=True,
            tool_name="attachment_reader",
            status="success",
            output_text=context,
            raw_result=raw_result,
            evidence_valid=True,
        ).to_dict()

    def _normalize_solver_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        used = bool(raw_result.get("used_deterministic_solver"))
        answer = str(raw_result.get("answer_text") or raw_result.get("answer") or "").strip()
        if not used or not answer:
            error_message = str(raw_result.get("error", "") or "no deterministic handler matched")
            result = failure_result(
                "deterministic_solver",
                status="unsupported",
                error_code="deterministic_handler_not_found",
                error_message=error_message,
                retry_hint=(
                    str(raw_result.get("next_action_hint", "") or "")
                    or "Use another registered capability or report a deterministic tool gap."
                ),
                raw_result=raw_result,
            )
            result["missing_inputs"] = list(raw_result.get("missing_inputs") or [])
            result["next_action_hint"] = str(raw_result.get("next_action_hint", "") or "")
            return result
        return ToolExecutionResult(
            ok=True,
            tool_name="deterministic_solver",
            status="success",
            output_text=answer,
            raw_result=raw_result,
            evidence_valid=True,
        ).to_dict()

    def _normalize_search_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        results = raw_result.get("results") if isinstance(raw_result.get("results"), list) else []
        notices = [str(item) for item in raw_result.get("notices", []) or []]
        if not results:
            return ToolExecutionResult(
                ok=True,
                tool_name="search",
                status="partial",
                output_text=str(raw_result),
                raw_result=raw_result,
                error_code="search_no_results",
                error_message="search returned no results",
                retryable=True,
                retry_hint="Change the query terms before retrying.",
                evidence_valid=False,
            ).to_dict()
        return ToolExecutionResult(
            ok=True,
            tool_name="search",
            status="success",
            output_text=str(raw_result),
            raw_result=raw_result,
            error_message="; ".join(notices),
            evidence_valid=True,
        ).to_dict()

    def _record_trace(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        agent_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        """
        記錄一次工具呼叫的輸入、輸出、Agent 與階段資訊。

        Args:
            - tool_name: 工具名稱。
            - parameters: 工具呼叫參數。
            - result: 標準化工具結果。
            - agent_id: 發起工具呼叫的 Agent id。
            - stage: 工具呼叫所屬階段。

        Returns:
            - None。
        """
        self.tool_traces.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "agent_id": agent_id,
                "stage": stage,
                "ok": result.get("ok", False),
                "status": result.get("status", ""),
                "error_code": result.get("error_code", ""),
                "evidence_valid": result.get("evidence_valid", False),
                "retryable": result.get("retryable", False),
                "retry_hint": result.get("retry_hint", ""),
                "output_text": result.get("output_text", ""),
                "error": result.get("error"),
            }
        )
