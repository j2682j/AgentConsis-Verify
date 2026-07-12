from __future__ import annotations

from .schema import HandlerPlan, ToolCandidate, ToolPlan, ToolPlanStep


class ToolPlanValidator:
    """
    Validate and sanitize a tool plan without assigning scores.
    """

    def __init__(self, *, max_steps: int = 3) -> None:
        self.max_steps = max(1, max_steps)

    def validate(self, plan: ToolPlan, candidates: list[ToolCandidate]) -> ToolPlan:
        candidate_by_name = {candidate.tool_name: candidate for candidate in candidates}
        required_tools = [candidate.tool_name for candidate in candidates if candidate.required]
        errors = list(plan.validation_errors)
        repair_actions = list(plan.repair_actions)
        selected: list[ToolPlanStep] = []
        seen: set[str] = set()

        for step in plan.tool_sequence:
            tool_name = step.tool_name.strip()
            if not tool_name:
                errors.append("empty_tool_name_removed")
                continue
            if tool_name not in candidate_by_name:
                errors.append(f"unknown_tool_removed:{tool_name}")
                continue
            if tool_name in seen:
                errors.append(f"duplicate_tool_removed:{tool_name}")
                continue
            selected.append(
                ToolPlanStep(
                    tool_name=tool_name,
                    purpose=step.purpose,
                    depends_on=[
                        dependency
                        for dependency in step.depends_on
                        if dependency in candidate_by_name and dependency != tool_name
                    ],
                    expected_output=step.expected_output,
                )
            )
            seen.add(tool_name)

        for required_tool in reversed(required_tools):
            if required_tool not in seen:
                selected.insert(
                    0,
                    ToolPlanStep(
                        tool_name=required_tool,
                        purpose="required by system routing",
                        depends_on=[],
                        expected_output="required evidence",
                    ),
                )
                seen.add(required_tool)
                repair_actions.append(f"insert_required_tool:{required_tool}")

        if len(selected) > self.max_steps:
            selected = selected[: self.max_steps]
            errors.append("max_plan_steps_truncated")

        existing = {step.tool_name for step in selected}
        normalized_steps: list[ToolPlanStep] = []
        for step in selected:
            normalized_steps.append(
                ToolPlanStep(
                    tool_name=step.tool_name,
                    purpose=step.purpose,
                    depends_on=[dependency for dependency in step.depends_on if dependency in existing],
                    expected_output=step.expected_output,
                )
            )

        normalized_handler_plans: list[HandlerPlan] = []
        seen_handlers: set[str] = set()
        valid_statuses = {"ready", "missing_input", "unavailable", "not_applicable"}
        for handler_plan in plan.handler_plans:
            tool_name = str(handler_plan.tool_name or "deterministic_handler").strip()
            handler_name = str(handler_plan.handler_name or "").strip()
            required_handler_role = str(handler_plan.required_handler_role or "").strip()
            if tool_name != "deterministic_handler":
                errors.append(f"handler_plan_non_deterministic_tool_removed:{tool_name}")
                continue
            if not handler_name and not required_handler_role:
                errors.append("empty_handler_plan_removed")
                continue
            dedupe_key = handler_name or f"role:{required_handler_role}"
            if dedupe_key in seen_handlers:
                errors.append(f"duplicate_handler_plan_removed:{handler_name}")
                continue
            status = str(handler_plan.status or "not_applicable").strip()
            if status not in valid_statuses:
                status = "not_applicable"
                errors.append(f"handler_plan_status_normalized:{handler_name}")
            normalized_handler_plans.append(
                HandlerPlan(
                    tool_name="deterministic_handler",
                    handler_name=handler_name,
                    required_handler_role=required_handler_role,
                    reason=handler_plan.reason,
                    required_inputs=list(handler_plan.required_inputs or []),
                    available_inputs=dict(handler_plan.available_inputs or {}),
                    missing_inputs=list(handler_plan.missing_inputs or []),
                    status=status,
                    next_action_hint=handler_plan.next_action_hint,
                    confidence=float(handler_plan.confidence or 0.0),
                )
            )
            seen_handlers.add(dedupe_key)

        return ToolPlan(
            requires_tools=bool(normalized_steps),
            tool_needs=list(plan.tool_needs),
            tool_sequence=normalized_steps,
            handler_plans=normalized_handler_plans,
            stop_condition=plan.stop_condition,
            planner_source=plan.planner_source,
            validation_errors=errors,
            repair_actions=repair_actions,
        )


__all__ = ["ToolPlanValidator"]
