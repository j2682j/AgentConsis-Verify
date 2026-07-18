from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.attachment_reader import AttachmentEvidenceBuilder
from tools.attachment_workspace import PreparedAttachmentArtifact
from tools.deterministic_handlers import DeterministicHandlerRouter, HandlerTrustGate
from tools.evidence.fact_extraction import (
    AttachmentFactExtractor,
    EvidenceFact,
    SemanticExtractionResult,
    render_attachment_facts,
)

from .models import AttachmentStrategy, AttachmentStrategyResult
from .parser import AttachmentStrategyParser
from .planner import AttachmentStrategyPlanner
from .reviewer import AttachmentStrategyReviewer


class AttachmentStrategyExecutor:
    """
    執行附件策略迴圈：模型規劃、附件讀取、handler 執行、失敗回饋。

    Args:
     - attachment_builder: 實際讀取附件並建立上下文的 builder。
     - handler_router: 執行 deterministic handlers 的 router。
     - trust_gate: 驗證 handler 輸出是否可作為 final evidence。
     - planner: 附件策略規劃器。
     - reviewer: 失敗時的一次性策略修正器。

    Returns:
     - AttachmentStrategyExecutor: 回傳 AttachmentStrategyResult 的執行器。
    """

    def __init__(
        self,
        *,
        attachment_builder: AttachmentEvidenceBuilder | None = None,
        handler_router: DeterministicHandlerRouter | None = None,
        trust_gate: HandlerTrustGate | None = None,
        planner: AttachmentStrategyPlanner | None = None,
        reviewer: AttachmentStrategyReviewer | None = None,
        attachment_fact_extractor: AttachmentFactExtractor | None = None,
    ) -> None:
        self.attachment_builder = attachment_builder or AttachmentEvidenceBuilder()
        self.handler_router = handler_router or DeterministicHandlerRouter()
        self.trust_gate = trust_gate or HandlerTrustGate()
        self.attachment_fact_extractor = (
            attachment_fact_extractor or AttachmentFactExtractor()
        )
        parser = AttachmentStrategyParser(set(self.allowed_handlers()))
        self.planner = planner or AttachmentStrategyPlanner(parser=parser)
        self.reviewer = reviewer or AttachmentStrategyReviewer(parser=parser)

    def run(
        self,
        *,
        question: str,
        attachment: dict[str, Any],
        existing_attachment_context: str = "",
        search_context: str = "",
        prepared_artifact: PreparedAttachmentArtifact | None = None,
        information_need: str = "",
    ) -> AttachmentStrategyResult:
        registered_handlers = self.allowed_handlers()
        planning_question = str(question or "").strip()
        if information_need.strip():
            planning_question = (
                f"{planning_question}\n\nSpecific attachment information need: "
                f"{information_need.strip()}"
            ).strip()
        attachment_context = existing_attachment_context.strip()
        if prepared_artifact is not None:
            attachment_context = str(prepared_artifact.context or attachment_context).strip()
            profile = dict(prepared_artifact.profile or {})
            parsed_payload = dict(prepared_artifact.parsed_payload or {})
            tool_usage = []
        elif attachment_context:
            file_path = Path(
                str(attachment.get("file_path") or attachment.get("path") or attachment.get("file_name") or "")
            )
            extension = str(attachment.get("extension") or file_path.suffix or "")
            provided_payload = self.attachment_builder.payload_builder.from_text(
                content=attachment_context,
                provenance={
                    "source": "provided_attachment_context",
                    "file_path": str(file_path),
                    "file_type": extension,
                    "parse_status": "success",
                },
            )
            profile = self.attachment_builder.profile_builder.build(
                file_path=file_path,
                extension=extension,
                read_ok=True,
                reader="provided_attachment_context",
                content=attachment_context,
                warnings=[],
                parsed_payload=provided_payload,
            ).to_dict()
            parsed_payload = {
                **provided_payload.to_dict(),
                "content": attachment_context,
                "reader": "provided_attachment_context",
                "reader_metadata": {},
            }
            tool_usage: list[dict[str, Any]] = []
        else:
            read_result = self.attachment_builder.build(question, attachment)
            attachment_context = str(read_result.get("context") or "").strip()
            profile = dict(read_result.get("profile") or {})
            parsed_payload = dict(read_result.get("parsed_payload") or {})
            tool_usage = list(read_result.get("tool_usage", []) or [])

        fact_tool_usage: dict[str, Any] | None = None
        fact_result = self._extract_attachment_facts(
            question=planning_question,
            answer_requirement=information_need,
            parsed_payload=parsed_payload,
        )
        if fact_result is not None:
            parsed_payload["semantic_facts"] = [
                fact.to_dict() for fact in fact_result.facts
            ]
            fact_context = render_attachment_facts(fact_result.facts)
            if fact_context:
                attachment_context = "\n\n".join(
                    part for part in (attachment_context, fact_context) if part.strip()
                )
            profile.setdefault("content_types", [])
            profile.setdefault("available_inputs", [])
            profile.setdefault("structure_summary", {})
            if fact_result.facts:
                if "semantic_facts" not in profile["content_types"]:
                    profile["content_types"].append("semantic_facts")
                if "semantic_facts" not in profile["available_inputs"]:
                    profile["available_inputs"].append("semantic_facts")
            profile["structure_summary"]["semantic_fact_count"] = len(
                fact_result.facts
            )
            fact_tool_usage = {
                    "ok": bool(fact_result.facts),
                    "tool_name": "attachment_fact_extractor",
                    "output_text": fact_context,
                    "output_type": "evidence_text",
                    "raw_result": {
                        "semantic_facts": [
                            fact.to_dict() for fact in fact_result.facts
                        ],
                        "diagnostics": dict(fact_result.diagnostics),
                    },
                    "evidence_valid": bool(
                        any(
                            fact.grounding_status == "grounded"
                            for fact in fact_result.facts
                        )
                    ),
                    "status": (
                        "success"
                        if fact_result.diagnostics.get("success", True)
                        else "partial"
                    ),
                    "error": fact_result.diagnostics.get("error") or None,
                }

        reader_status = str(profile.get("parse_status") or "failed")
        capability_metadata = {
            "attachment_profile": profile,
            "parsed_payload": parsed_payload,
            "require_attachment_provenance": True,
        }
        capabilities, capability_diagnostics = self.handler_router.eligible_capabilities(
            question=planning_question,
            attachment=attachment,
            attachment_result=attachment_context,
            search_result=search_context,
            metadata=capability_metadata,
        )
        allowed_handler_capabilities = [item.to_dict() for item in capabilities]
        eligible_handler_names = [item.handler_name for item in capabilities]
        capability_status_counts: dict[str, int] = {}
        for item in capability_diagnostics:
            capability_status_counts[item.status] = (
                capability_status_counts.get(item.status, 0) + 1
            )
        strategy = AttachmentStrategy()
        raw_plan = ""
        strategy_error = ""
        try:
            strategy, raw_plan = self.planner.plan(
                question=planning_question,
                attachment_profile=profile,
                allowed_handlers=allowed_handler_capabilities,
            )
        except Exception as exc:
            strategy_error = f"{type(exc).__name__}: {exc}"
            strategy = AttachmentStrategy(missing_inputs=["strategy_planner_failed"])
        strategy_status = self._strategy_status(strategy, strategy_error=strategy_error)
        tool_usage.append(
            {
                "ok": strategy_status == "success",
                "tool_name": "attachment_strategy_planner",
                "output_text": self._render_strategy(strategy),
                "raw_result": {
                    "raw_reply": raw_plan,
                    "strategy": strategy.to_dict(),
                    "attachment_profile": profile,
                    "allowed_handlers": allowed_handler_capabilities,
                },
                "status": strategy_status,
                "error": strategy_error or (
                    "invalid_strategy_json"
                    if strategy_status == "invalid_output"
                    else None
                ),
            }
        )
        if fact_tool_usage is not None:
            tool_usage.append(fact_tool_usage)

        try:
            solver_context, handler_usage = self._run_handlers(
                question=planning_question,
                attachment=attachment,
                attachment_context=attachment_context,
                search_context=search_context,
                strategy=strategy,
                attachment_profile=profile,
                parsed_payload=parsed_payload,
                eligible_handler_names=eligible_handler_names,
            )
        except Exception as exc:
            solver_context = ""
            handler_usage = [
                {
                    "ok": False,
                    "tool_name": "attachment_strategy_handler",
                    "status": "failed",
                    "output_text": "",
                    "raw_result": {},
                    "evidence_valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        tool_usage.extend(handler_usage)
        handler_status = self._handler_status(strategy, handler_usage)

        revised_strategy = None
        final_answer_candidate = ""
        if strategy.required_handler and not solver_context.strip():
            raw_review = ""
            review_error = ""
            try:
                revised_strategy, final_answer_candidate, raw_review = self.reviewer.review(
                    question=planning_question,
                    strategy=strategy,
                    handler_results=self._review_feedback(handler_usage),
                    attachment_profile=profile,
                    allowed_handlers=allowed_handler_capabilities,
                )
            except Exception as exc:
                review_error = f"{type(exc).__name__}: {exc}"
            tool_usage.append(
                {
                    "ok": bool(revised_strategy or final_answer_candidate),
                    "tool_name": "attachment_strategy_reviewer",
                    "output_text": self._render_strategy(revised_strategy)
                    if revised_strategy
                    else final_answer_candidate,
                    "raw_result": {
                        "raw_reply": raw_review,
                        "revised_strategy": revised_strategy.to_dict() if revised_strategy else None,
                        "final_answer_candidate": final_answer_candidate,
                    },
                    "error": review_error or None,
                }
            )
            if revised_strategy and revised_strategy.required_handler:
                try:
                    retry_context, retry_usage = self._run_handlers(
                        question=planning_question,
                        attachment=attachment,
                        attachment_context=attachment_context,
                        search_context=search_context,
                        strategy=revised_strategy,
                        attachment_profile=profile,
                        parsed_payload=parsed_payload,
                        eligible_handler_names=eligible_handler_names,
                        retry=True,
                    )
                except Exception as exc:
                    retry_context = ""
                    retry_usage = [
                        {
                            "ok": False,
                            "tool_name": "attachment_strategy_handler",
                            "status": "failed",
                            "output_text": "",
                            "raw_result": {},
                            "evidence_valid": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ]
                if retry_context:
                    solver_context = retry_context
                tool_usage.extend(retry_usage)
                handler_status = self._handler_status(revised_strategy, retry_usage)

        return AttachmentStrategyResult(
            strategy=strategy,
            revised_strategy=revised_strategy,
            final_answer_candidate=final_answer_candidate,
            attachment_context=attachment_context,
            solver_context=solver_context.strip(),
            attachment_profile=profile,
            parsed_payload=parsed_payload,
            tool_usage=tool_usage,
            reader_status=reader_status,
            strategy_status=strategy_status,
            handler_status=handler_status,
            metadata={
                "registered_handlers": registered_handlers,
                "eligible_handlers": eligible_handler_names,
                "handler_capabilities": allowed_handler_capabilities,
                "capability_status_counts": capability_status_counts,
                "needs_search": bool(
                    (revised_strategy.needs_search if revised_strategy else strategy.needs_search)
                ),
                "handler_count": int(bool(strategy.required_handler)),
                "parse_status": str(profile.get("parse_status") or "failed"),
                "reader_status": reader_status,
                "strategy_status": strategy_status,
                "handler_status": handler_status,
            },
        )

    def _extract_attachment_facts(
        self,
        *,
        question: str,
        answer_requirement: str,
        parsed_payload: dict[str, Any],
    ) -> SemanticExtractionResult:
        existing = [
            EvidenceFact.from_dict(item)
            for item in list(parsed_payload.get("semantic_facts") or [])
            if isinstance(item, dict)
        ]
        if existing:
            return SemanticExtractionResult(
                facts=existing,
                diagnostics={
                    "success": True,
                    "reused_existing_facts": True,
                    "stored_fact_count": len(existing),
                },
            )
        try:
            return self.attachment_fact_extractor.extract(
                question=question,
                answer_requirement=answer_requirement,
                parsed_payload=parsed_payload,
            )
        except Exception as exc:
            return SemanticExtractionResult(
                diagnostics={
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    @staticmethod
    def _strategy_status(
        strategy: AttachmentStrategy,
        *,
        strategy_error: str = "",
    ) -> str:
        if strategy_error:
            return "failed"
        if "invalid_strategy_json" in strategy.missing_inputs:
            return "invalid_output"
        if not (
            strategy.information_need
            or strategy.required_handler
            or strategy.expected_answer
            or strategy.needs_search
        ):
            return "not_required"
        return "success"

    @staticmethod
    def _handler_status(
        strategy: AttachmentStrategy,
        usage: list[dict[str, Any]],
    ) -> str:
        if not strategy.required_handler:
            return "not_required"
        handler_items = [
            item
            for item in usage
            if item.get("tool_name") == "attachment_strategy_handler"
        ]
        if any(item.get("evidence_valid") for item in handler_items):
            return "success"
        statuses = {str(item.get("status") or "") for item in handler_items}
        if "missing_inputs" in statuses:
            return "missing_inputs"
        trust_statuses = {
            str((item.get("handler_trust") or {}).get("status") or "")
            for item in handler_items
            if isinstance(item.get("handler_trust"), dict)
        }
        if trust_statuses - {"", "missing_input"}:
            return "rejected"
        return "failed"

    def allowed_handlers(self) -> list[str]:
        registry = getattr(self.handler_router, "registry", None)
        names: set[str] = set()
        if registry is not None:
            names.update(str(handler.name) for handler in registry.list_handlers())
            role_aliases = getattr(registry, "_role_aliases", {})
            if isinstance(role_aliases, dict):
                names.update(str(role) for role in role_aliases)
        return sorted(name for name in names if name)

    def _run_handlers(
        self,
        *,
        question: str,
        attachment: dict[str, Any],
        attachment_context: str,
        search_context: str,
        strategy: AttachmentStrategy,
        attachment_profile: dict[str, Any],
        parsed_payload: dict[str, Any],
        eligible_handler_names: list[str],
        retry: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        handler_ref = str(strategy.required_handler or "").strip()
        if not handler_ref:
            return "", []

        handler_name, handler_role = self._handler_selector(
            handler_ref,
            eligible_handler_names=eligible_handler_names,
        )
        preflight_metadata = {
            "attachment_strategy": strategy.to_dict(),
            "attachment_profile": attachment_profile,
            "parsed_payload": parsed_payload,
            "require_attachment_provenance": True,
        }
        preflight = self.handler_router.preflight(
            question=question,
            attachment=attachment,
            attachment_result=attachment_context,
            search_result=search_context,
            metadata=preflight_metadata,
            handler_name=handler_name,
            required_handler_role=handler_role,
            eligible_handler_names=eligible_handler_names,
        )
        selected_handler_name = preflight.handler_name or handler_name
        available_inputs = {name: True for name in preflight.available_inputs}
        handler_plan = {
            "handler_name": selected_handler_name,
            "required_handler_role": handler_role,
            "reason": strategy.information_need,
            "required_inputs": list(preflight.required_inputs or strategy.required_inputs),
            "available_inputs": available_inputs,
            "input_provenance": dict(preflight.input_provenance),
            "attachment_profile": attachment_profile,
            "status": preflight.status,
            "preflight": preflight.to_dict(),
            "strategy": strategy.to_dict(),
            "retry": retry,
        }
        if not preflight.ready:
            return "", [
                {
                    "ok": False,
                    "tool_name": "attachment_strategy_handler",
                    "handler_name": selected_handler_name,
                    "planned_handler_name": handler_name or handler_role,
                    "required_handler_role": handler_role,
                    "handler_plan": handler_plan,
                    "handler_trust": {
                        "trusted": False,
                        "status": preflight.status,
                        "reasons": [preflight.reason or preflight.status],
                    },
                    "status": preflight.status,
                    "output_type": "intermediate_value",
                    "semantic_role": "",
                    "supporting_inputs": [],
                    "output_text": "",
                    "raw_result": {},
                    "missing_inputs": list(preflight.missing_inputs),
                    "next_action_hint": (
                        "Provide unambiguous typed attachment inputs or continue with Stage1 reasoning."
                    ),
                    "evidence_valid": False,
                    "error": "planned handler did not pass capability preflight",
                }
            ]

        result = self.handler_router.run(
            question=question,
            attachment=attachment,
            attachment_result=attachment_context,
            search_result=search_context,
            handler_name=selected_handler_name,
            required_handler_role=handler_role,
            metadata={
                **preflight_metadata,
                "eligible_handler_names": eligible_handler_names,
            },
        )
        trust = self.trust_gate.validate(
            result,
            question=question,
            handler_plan=handler_plan,
        )
        context = trust.evidence_text if trust.trusted else ""
        usage = [
            {
                "ok": bool(trust.trusted),
                "tool_name": "attachment_strategy_handler",
                "handler_name": result.handler_name,
                "planned_handler_name": selected_handler_name,
                "required_handler_role": handler_role,
                "handler_plan": handler_plan,
                "handler_trust": trust.to_dict(),
                "status": result.status,
                "output_type": result.output_type,
                "semantic_role": result.semantic_role,
                "supporting_inputs": list(result.supporting_inputs or []),
                "output_text": context,
                "raw_result": result.to_dict(),
                "missing_inputs": list(result.missing_inputs or []),
                "next_action_hint": result.next_action_hint,
                "evidence_valid": bool(trust.trusted),
                "error": result.error or None,
            }
        ]
        return context.strip(), usage

    def _review_feedback(self, usage: list[dict[str, Any]]) -> list[dict[str, Any]]:
        feedback: list[dict[str, Any]] = []
        for item in usage:
            trust = item.get("handler_trust")
            trust = trust if isinstance(trust, dict) else {}
            feedback.append(
                {
                    "handler_name": str(item.get("handler_name") or ""),
                    "status": str(item.get("status") or ""),
                    "output_type": str(item.get("output_type") or ""),
                    "missing_inputs": list(item.get("missing_inputs") or []),
                    "trust_status": str(trust.get("status") or ""),
                    "trust_reasons": list(trust.get("reasons") or []),
                    "next_action_hint": str(item.get("next_action_hint") or ""),
                }
            )
        return feedback

    def _handler_selector(
        self,
        handler_ref: str,
        *,
        eligible_handler_names: list[str] | None = None,
    ) -> tuple[str, str]:
        text = str(handler_ref or "").strip()
        registry = getattr(self.handler_router, "registry", None)
        eligible = {
            str(name).strip()
            for name in list(eligible_handler_names or [])
            if str(name).strip()
        }
        if (
            registry is not None
            and registry.get(text) is not None
            and (not eligible or text in eligible)
        ):
            return text, ""
        if registry is not None and registry.find_by_role(text):
            return "", text
        return "", text

    def _render_strategy(self, strategy: AttachmentStrategy | None) -> str:
        if strategy is None:
            return ""
        lines = [
            "Attachment Strategy:",
            f"- information_need: {strategy.information_need or '(unspecified)'}",
            f"- required_handler: {strategy.required_handler or '(none)'}",
            f"- required_inputs: {', '.join(strategy.required_inputs) or '(none)'}",
            f"- expected_answer: {strategy.expected_answer or '(unspecified)'}",
            f"- needs_search: {strategy.needs_search}",
        ]
        if strategy.missing_inputs:
            lines.append(f"- missing_inputs: {', '.join(strategy.missing_inputs)}")
        return "\n".join(lines)


__all__ = ["AttachmentStrategyExecutor"]
