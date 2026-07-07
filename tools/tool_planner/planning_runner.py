from __future__ import annotations

from typing import Any

from tools.deterministic_handlers import DeterministicHandlerRouter, HandlerInput
from tools.tool_capability_registry import ToolCapabilityRegistry

from .candidate_router import ToolCandidateRouter
from .fallback_planner import FallbackToolPlanner
from .parser import ToolPlanParser
from .schema import HandlerPlan, ToolPlanResult, ToolPlanStep
from .slm_planner import SLMToolPlanner
from .validator import ToolPlanValidator


class ToolPlanningRunner:
    """
    Build a validated hybrid evidence-preparation tool plan.
    """

    def __init__(
        self,
        *,
        candidate_router: ToolCandidateRouter | None = None,
        slm_planner: SLMToolPlanner | None = None,
        parser: ToolPlanParser | None = None,
        validator: ToolPlanValidator | None = None,
        fallback_planner: FallbackToolPlanner | None = None,
        capability_registry: ToolCapabilityRegistry | None = None,
        deterministic_handler_router: DeterministicHandlerRouter | None = None,
        deterministic_handler_available: bool = True,
    ) -> None:
        self.candidate_router = candidate_router or ToolCandidateRouter()
        self.slm_planner = slm_planner or SLMToolPlanner()
        self.parser = parser or ToolPlanParser()
        self.validator = validator or ToolPlanValidator(max_steps=3)
        self.fallback_planner = fallback_planner or FallbackToolPlanner()
        self.capability_registry = capability_registry or ToolCapabilityRegistry()
        self.deterministic_handler_router = deterministic_handler_router or DeterministicHandlerRouter()
        self.deterministic_handler_available = deterministic_handler_available

    def plan(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
    ) -> ToolPlanResult:
        routing = routing or {}
        candidates = self.candidate_router.route(
            question=question,
            attachment=attachment,
            routing=routing,
            deterministic_handler_available=self.deterministic_handler_available,
        )
        raw_reply = ""
        try:
            raw_reply = self.slm_planner.plan(
                question=question,
                attachment=attachment,
                candidates=candidates,
                routing=routing,
            )
        except Exception as exc:
            raw_reply = f"[tool_planner_error] {type(exc).__name__}: {exc}"

        system_needs = self.capability_registry.infer_needs(
            question=question,
            attachment=attachment,
            routing=routing,
        )
        parsed = self.parser.parse(raw_reply) if raw_reply.strip() else self.fallback_planner.plan(
            candidates=candidates,
            routing=routing,
            deterministic_handler_requested=self.deterministic_handler_available,
        )
        parsed = self._augment_plan_with_capabilities(
            parsed,
            system_needs=system_needs,
            candidates=candidates,
            question=question,
            attachment=attachment,
        )
        validated = self.validator.validate(parsed, candidates)
        fallback_used = False
        if not validated.tool_sequence and candidates:
            parsed = self.fallback_planner.plan(
                candidates=candidates,
                routing=routing,
                deterministic_handler_requested=self.deterministic_handler_available,
            )
            parsed = self._augment_plan_with_capabilities(
                parsed,
                system_needs=system_needs,
                candidates=candidates,
                question=question,
                attachment=attachment,
            )
            validated = self.validator.validate(parsed, candidates)
            fallback_used = True
        elif parsed.planner_source == "fallback":
            fallback_used = True

        return ToolPlanResult(
            candidate_tools=candidates,
            raw_planner_reply=raw_reply,
            parsed_plan=parsed,
            validated_plan=validated,
            fallback_used=fallback_used,
        )

    def _augment_plan_with_capabilities(
        self,
        plan,
        *,
        system_needs,
        candidates,
        question: str,
        attachment: dict[str, Any] | None,
    ):
        merged_needs = []
        seen_need_keys: set[str] = set()
        for need in list(system_needs or []) + list(plan.tool_needs or []):
            key = (
                str(need.need_type),
                tuple(sorted(str(item) for item in need.required_capabilities)),
                tuple(sorted(str(item) for item in need.input_refs)),
            )
            if key in seen_need_keys:
                continue
            seen_need_keys.add(key)
            merged_needs.append(need)

        candidate_names = [candidate.tool_name for candidate in candidates]
        matched_steps = self.capability_registry.match_steps(
            needs=merged_needs,
            candidate_tool_names=candidate_names,
            available_inputs=self.capability_registry.available_inputs(
                question=question,
                attachment=attachment,
            ),
        )
        steps = []
        seen_tools: set[str] = set()
        for step in matched_steps + list(plan.tool_sequence or []):
            if step.tool_name in seen_tools:
                continue
            seen_tools.add(step.tool_name)
            steps.append(step)
        plan.tool_needs = merged_needs
        plan.tool_sequence = steps
        plan.handler_plans = self._build_handler_plans(
            existing_handler_plans=list(plan.handler_plans or []),
            candidates=candidates,
            tool_sequence=steps,
            question=question,
            attachment=attachment,
        )
        plan.requires_tools = bool(plan.requires_tools or merged_needs or steps)
        if matched_steps and "capability_match_augmented" not in plan.repair_actions:
            plan.repair_actions.append("capability_match_augmented")
        return plan

    def _build_handler_plans(
        self,
        *,
        existing_handler_plans: list[HandlerPlan],
        candidates,
        tool_sequence: list[ToolPlanStep],
        question: str,
        attachment: dict[str, Any] | None,
    ) -> list[HandlerPlan]:
        candidate_names = {candidate.tool_name for candidate in candidates}
        deterministic_selected = any(step.tool_name == "deterministic_handler" for step in tool_sequence)
        if "deterministic_handler" not in candidate_names and not deterministic_selected:
            return []

        generated = self._match_deterministic_handlers(
            question=question,
            attachment=attachment,
        )
        merged: list[HandlerPlan] = []
        seen: set[str] = set()
        for handler_plan in generated + existing_handler_plans:
            if not handler_plan.handler_name or handler_plan.handler_name in seen:
                continue
            if not self._handler_exists(handler_plan.handler_name):
                continue
            merged.append(handler_plan)
            seen.add(handler_plan.handler_name)
        return merged[:3]

    def _handler_exists(self, handler_name: str) -> bool:
        registry = getattr(self.deterministic_handler_router, "registry", None)
        getter = getattr(registry, "get", None)
        if not callable(getter):
            return True
        try:
            return getter(handler_name) is not None
        except Exception:
            return False

    def _match_deterministic_handlers(
        self,
        *,
        question: str,
        attachment: dict[str, Any] | None,
    ) -> list[HandlerPlan]:
        handler_input = HandlerInput(
            question=question,
            attachment=attachment or {},
        )
        try:
            matches = self.deterministic_handler_router.match_handlers(handler_input)
        except Exception:
            return []

        available_inputs = self._handler_available_inputs(attachment=attachment)
        plans: list[HandlerPlan] = []
        for match in matches:
            status = self._handler_plan_status(match)
            if status == "not_applicable":
                continue
            plans.append(
                HandlerPlan(
                    tool_name="deterministic_handler",
                    handler_name=match.handler_name,
                    reason=match.reason,
                    required_inputs=list(match.required_inputs or []),
                    available_inputs={
                        name: bool(available_inputs.get(name, False))
                        for name in match.required_inputs or []
                    },
                    missing_inputs=list(match.missing_inputs or []),
                    status=status,
                    next_action_hint=self._handler_next_action_hint(match),
                    confidence=float(match.confidence or 0.0),
                )
            )
            if len(plans) >= 3:
                break
        return plans

    def _handler_available_inputs(self, *, attachment: dict[str, Any] | None) -> dict[str, bool]:
        attachment = attachment or {}
        return {
            "question": True,
            "attachment": bool(attachment),
            "file_path": bool(attachment.get("file_path") or attachment.get("path")),
            "table_rows": False,
            "source_text": False,
            "search_result": False,
        }

    def _handler_plan_status(self, match) -> str:
        if match.matched and not match.missing_inputs:
            return "ready"
        if match.missing_inputs and match.confidence >= self.deterministic_handler_router.threshold:
            return "missing_input"
        return "not_applicable"

    def _handler_next_action_hint(self, match) -> str:
        if not match.missing_inputs:
            return ""
        return "Recover missing deterministic inputs: " + ", ".join(match.missing_inputs)


__all__ = ["ToolPlanningRunner"]
