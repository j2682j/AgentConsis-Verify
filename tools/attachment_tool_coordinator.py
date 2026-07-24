from __future__ import annotations

from typing import Any

from tools.attachment_strategy import AttachmentStrategyExecutor
from tools.attachment_workspace import AttachmentWorkspace


class AttachmentToolCoordinator:
    """Coordinate Stage1 attachment reuse, strategy planning, and handler execution."""

    def __init__(
        self,
        *,
        strategy_executor: AttachmentStrategyExecutor | None = None,
    ) -> None:
        self.strategy_executor = strategy_executor or AttachmentStrategyExecutor()

    def run(
        self,
        *,
        question: str,
        information_need: str,
        attachment: dict[str, Any],
        workspace: AttachmentWorkspace,
        search_context: str = "",
    ) -> dict[str, Any]:
        need = str(information_need or question or "").strip()
        decision = workspace.decide(need)
        if decision.cached_result is not None:
            result = dict(decision.cached_result)
            result["attachment_reuse"] = {
                "action": decision.action,
                "reason": decision.reason,
            }
            return result
        if decision.action == "blocked_duplicate":
            return self._failure(
                status="duplicate_blocked",
                error_code=decision.reason,
                message="The same failed attachment request will not be executed again.",
                workspace=workspace,
            )

        artifact = workspace.artifact()
        strategy_result = self.strategy_executor.run(
            question=question,
            attachment=attachment,
            search_context=search_context,
            prepared_artifact=(artifact if decision.action == "reuse_prepared_payload" else None),
            information_need=need,
        )
        if artifact is None:
            workspace.seed_from_strategy_result(strategy_result, reader_executed=True)

        handler_items = [
            item
            for item in list(strategy_result.tool_usage or [])
            if isinstance(item, dict)
            and item.get("tool_name") == "attachment_strategy_handler"
        ]
        selected = handler_items[-1] if handler_items else {}
        handler_name = str(selected.get("handler_name") or "")
        handler_trust = selected.get("handler_trust") or {}
        trusted = bool(handler_trust.get("trusted"))
        usable_as_intermediate = bool(handler_trust.get("usable_as_intermediate"))
        handler_evidence = str(selected.get("output_text") or "").strip()
        attachment_context = str(strategy_result.attachment_context or "").strip()
        output_text = handler_evidence or attachment_context
        evidence_valid = bool(
            handler_evidence and (trusted or usable_as_intermediate)
        ) or bool(
            attachment_context and strategy_result.reader_status == "success"
        )
        output_type = str(selected.get("output_type") or "evidence_text")
        if not handler_evidence:
            output_type = "evidence_text"

        result = {
            "ok": evidence_valid,
            "tool_name": "attachment_reader",
            "status": (
                "success"
                if handler_evidence and (trusted or usable_as_intermediate)
                else "partial"
                if evidence_valid
                else "failed"
            ),
            "output_type": output_type,
            "value": self._handler_value(selected),
            "evidence_text": output_text,
            "output_text": output_text,
            "handler_name": handler_name,
            "trusted": trusted,
            "usable_as_intermediate": usable_as_intermediate,
            "evidence_valid": evidence_valid,
            "missing_inputs": list(selected.get("missing_inputs") or []),
            "next_action_hint": str(selected.get("next_action_hint") or ""),
            "error_code": "" if evidence_valid else "attachment_evidence_unavailable",
            "error_message": str(selected.get("error") or ""),
            "error": selected.get("error"),
            "retryable": False,
            "retry_hint": str(selected.get("next_action_hint") or ""),
            "cache_hit": False,
            "duplicate_request": False,
            "raw_result": {
                "strategy_result": strategy_result.to_dict(),
                "attachment_workspace": workspace.snapshot(),
            },
            "attachment_reuse": {
                "action": decision.action,
                "reason": decision.reason,
            },
        }
        workspace.record_result(
            need,
            result,
            handler_name=handler_name,
            handler_executed=bool(handler_items),
        )
        return result

    @staticmethod
    def _handler_value(handler_item: dict[str, Any]) -> str:
        raw = handler_item.get("raw_result")
        if not isinstance(raw, dict):
            return ""
        return str(raw.get("answer") or "").strip()

    @staticmethod
    def _failure(
        *,
        status: str,
        error_code: str,
        message: str,
        workspace: AttachmentWorkspace,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "tool_name": "attachment_reader",
            "status": status,
            "output_type": "failed",
            "value": "",
            "evidence_text": "",
            "output_text": "",
            "handler_name": "",
            "trusted": False,
            "evidence_valid": False,
            "missing_inputs": [],
            "next_action_hint": "Use existing attachment evidence or continue reasoning.",
            "error_code": error_code,
            "error_message": message,
            "error": message,
            "retryable": False,
            "retry_hint": "Use existing attachment evidence or continue reasoning.",
            "cache_hit": False,
            "duplicate_request": True,
            "raw_result": {"attachment_workspace": workspace.snapshot()},
        }


__all__ = ["AttachmentToolCoordinator"]
