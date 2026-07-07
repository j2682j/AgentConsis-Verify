from __future__ import annotations

from typing import Any

from parsers.json_parse import try_parse_json

from .schema import HandlerPlan, ToolNeed, ToolPlan, ToolPlanStep


class ToolPlanParser:
    """
    Parse and lightly repair a tool planner JSON response.
    """

    def parse(self, raw_reply: str) -> ToolPlan:
        parsed = try_parse_json(raw_reply)
        if not isinstance(parsed, dict):
            return ToolPlan(
                requires_tools=False,
                planner_source="parse_failed",
                validation_errors=["planner_json_parse_failed"],
            )
        return self.from_dict(parsed)

    def from_dict(self, data: dict[str, Any]) -> ToolPlan:
        repair_actions: list[str] = []
        raw_sequence = data.get("tool_sequence")
        if raw_sequence is None and isinstance(data.get("tools"), list):
            raw_sequence = data.get("tools")
            repair_actions.append("tools_to_tool_sequence")

        steps: list[ToolPlanStep] = []
        handler_plans: list[HandlerPlan] = []
        needs: list[ToolNeed] = []
        raw_needs = data.get("tool_needs")
        if raw_needs is None and isinstance(data.get("needs"), list):
            raw_needs = data.get("needs")
            repair_actions.append("needs_to_tool_needs")
        if isinstance(raw_needs, list):
            for item in raw_needs:
                if not isinstance(item, dict):
                    continue
                capabilities = item.get("required_capabilities") or item.get("capabilities") or []
                input_refs = item.get("input_refs") or item.get("inputs") or []
                if isinstance(capabilities, str):
                    capabilities = [capabilities]
                if isinstance(input_refs, str):
                    input_refs = [input_refs]
                needs.append(
                    ToolNeed(
                        need_type=str(item.get("need_type") or item.get("type") or "").strip(),
                        required_capabilities=[
                            str(value).strip()
                            for value in capabilities
                            if str(value).strip()
                        ],
                        input_refs=[
                            str(value).strip()
                            for value in input_refs
                            if str(value).strip()
                        ],
                        reason=str(item.get("reason", "") or "").strip(),
                    )
                )
        if isinstance(raw_sequence, list):
            for item in raw_sequence:
                if not isinstance(item, dict):
                    continue
                tool_name = item.get("tool_name")
                if tool_name is None and item.get("tool") is not None:
                    tool_name = item.get("tool")
                    repair_actions.append("tool_to_tool_name")
                depends_on = item.get("depends_on")
                if depends_on is None and item.get("depends") is not None:
                    depends_on = item.get("depends")
                    repair_actions.append("depends_to_depends_on")
                if isinstance(depends_on, str):
                    depends = [depends_on]
                elif isinstance(depends_on, list):
                    depends = [str(value).strip() for value in depends_on if str(value).strip()]
                else:
                    depends = []
                steps.append(
                    ToolPlanStep(
                        tool_name=str(tool_name or "").strip(),
                        purpose=str(item.get("purpose", "") or "").strip(),
                        depends_on=depends,
                        expected_output=str(item.get("expected_output", "") or "").strip(),
                    )
                )

        raw_handler_plans = data.get("handler_plans")
        if raw_handler_plans is None and isinstance(data.get("handlers"), list):
            raw_handler_plans = data.get("handlers")
            repair_actions.append("handlers_to_handler_plans")
        if isinstance(raw_handler_plans, list):
            for item in raw_handler_plans:
                if not isinstance(item, dict):
                    continue
                required_inputs = item.get("required_inputs") or []
                missing_inputs = item.get("missing_inputs") or []
                if isinstance(required_inputs, str):
                    required_inputs = [required_inputs]
                if isinstance(missing_inputs, str):
                    missing_inputs = [missing_inputs]
                available_inputs = item.get("available_inputs")
                handler_plans.append(
                    HandlerPlan(
                        tool_name=str(item.get("tool_name") or "deterministic_handler").strip(),
                        handler_name=str(item.get("handler_name") or item.get("handler") or "").strip(),
                        reason=str(item.get("reason", "") or "").strip(),
                        required_inputs=[
                            str(value).strip()
                            for value in required_inputs
                            if str(value).strip()
                        ],
                        available_inputs=available_inputs if isinstance(available_inputs, dict) else {},
                        missing_inputs=[
                            str(value).strip()
                            for value in missing_inputs
                            if str(value).strip()
                        ],
                        status=str(item.get("status", "not_applicable") or "not_applicable").strip(),
                        next_action_hint=str(item.get("next_action_hint", "") or "").strip(),
                        confidence=self._float_value(item.get("confidence")),
                    )
                )

        return ToolPlan(
            requires_tools=bool(data.get("requires_tools", bool(steps or needs or handler_plans))),
            tool_needs=needs,
            tool_sequence=steps,
            handler_plans=handler_plans,
            stop_condition=str(data.get("stop_condition", "") or "").strip(),
            planner_source=str(data.get("planner_source", "slm") or "slm"),
            validation_errors=[],
            repair_actions=repair_actions,
        )

    def _float_value(self, value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["ToolPlanParser"]
