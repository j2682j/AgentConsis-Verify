from __future__ import annotations

from typing import Any

from .context_budget import ContextBudgetManager
from .context_builder import ContextBuilder, ContextPacket


STAGE1_SYSTEM_PROMPT = """You are one agent in a multi-agent reasoning network.

1. Use Verified Evidence when it directly supports the answer.
2. Unverified References are tentative retrieval context, not verified answer support.
3. If only Unverified References are available, inspect them critically and lower confidence.
4. Do not list unverified references in used_evidence_ids.
5. Do not answer from general knowledge when the task asks for a specific external fact.
6. Your reasoning will be evaluated step by step.
Return JSON only. Do not include markdown or text outside JSON.
"""


STAGE1_USER_PROMPT = """Question:
{question}

Answer_Requirement:
{answer_requirement}

Solver_Result:
{solver_result}

Attachment_Result:
{attachment_result}

Search_Result:
{search_result}


Return exactly this JSON schema:
{{
  "reasoning_steps": [
    "step 1. Identify the required value from evidence.",
    "step 2. Convert units or normalize values if needed.",
    "step 3. Perform one calculation or comparison.",
    "step 4. Apply rounding if required.",
    ......
    "step N. Format the computed value according to the question."
  ],
  "final_answer": "short final answer only",
  "confidence": 0.0,
  "used_evidence_ids": ["E1"],
  "answer_type": "number | date | person | organization | location | title | list | short_text | boolean | unknown",
  "tool_request": null
}}

Rules:
- Each step must be one short sentence and exactly one action.
- If one step needs multiple actions, split it into multiple steps.
- Keep final_answer short and separate from reasoning_steps.
- Never include "Final Answer", "Answer", "boxed", or conclusion text inside reasoning_steps.
- Split lookup, conversion, calculation, rounding, and formatting into separate steps.
- For numeric work, write one explicit equation: "Calculation: <expression> = <value> <unit>".
- Every equation input must come from the question, Evidence, or a previously computed step."""


class Stage1ContextBuilder(ContextBuilder):
    """Build Stage1 agent chat messages from question and evidence packets."""

    REQUIRED_PACKET_TYPES = {"question", "system_instruction"}
    EVIDENCE_PACKET_TYPES = {
        "answer_requirement",
        "search_result",
        "attachment_result",
        "solver_result",
    }

    def __init__(
        self,
        *args: Any,
        budget_manager: ContextBudgetManager | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.budget_manager = budget_manager or ContextBudgetManager()

    def gather(
        self,
        *,
        question: str,
        evidence_packets: list[ContextPacket] | None = None,
        system_instructions: str | None = None,
        **_: Any,
    ) -> list[ContextPacket]:
        packets = [
            ContextPacket(
                packet_type="question",
                content=self._normalize_text(question),
                priority=100,
                metadata={"source": "user"},
            )
        ]

        if system_instructions:
            packets.append(
                ContextPacket(
                    packet_type="system_instruction",
                    content=system_instructions,
                    priority=1000,
                    metadata={"source": "system"},
                )
            )

        packets.extend(evidence_packets or [])
        return packets

    def select(self, packets: list[ContextPacket], **_: Any) -> list[ContextPacket]:
        selected = [
            packet
            for packet in packets
            if packet.packet_type in self.REQUIRED_PACKET_TYPES or packet.content.strip()
        ]
        return sorted(selected, key=lambda packet: packet.priority, reverse=True)

    def structure(self, packets: list[ContextPacket], **_: Any) -> dict[str, Any]:
        structured = {
            "system": STAGE1_SYSTEM_PROMPT,
            "question": "",
            "answer_requirement": self.config.none_text,
            "search_result": self.config.none_text,
            "attachment_result": self.config.none_text,
            "solver_result": self.config.none_text,
        }
        buckets = {
            "answer_requirement": [],
            "search_result": [],
            "attachment_result": [],
            "solver_result": [],
        }

        for packet in packets:
            content = packet.content.strip()
            if packet.packet_type == "system_instruction" and content:
                structured["system"] = content
            elif packet.packet_type == "question":
                structured["question"] = content
            elif packet.packet_type in buckets and content:
                buckets[packet.packet_type].append(content)

        for key, values in buckets.items():
            if values:
                structured[key] = "\n\n".join(values)

        return structured

    def compress(self, structured: dict[str, Any], **_: Any) -> dict[str, Any]:
        compressed = dict(structured)
        compressed["search_result"] = (
            self._compress_multiline_text(
                compressed["search_result"],
                max_lines=self.config.max_context_lines,
                max_chars=self.config.max_context_chars,
            )
            or self.config.none_text
        )
        compressed["attachment_result"] = (
            self._compress_multiline_text(
                compressed["attachment_result"],
                max_lines=self.config.max_context_lines,
                max_chars=self.config.max_context_chars,
            )
            or self.config.none_text
        )
        compressed["solver_result"] = (
            self._compress_multiline_text(
                compressed["solver_result"],
                max_lines=20,
                max_chars=self.config.max_solver_chars,
            )
            or self.config.none_text
        )
        return self._apply_context_budget(compressed)

    def _apply_context_budget(self, structured: dict[str, Any]) -> dict[str, Any]:
        budget_sections = {
            key: value
            for key, value in structured.items()
            if key not in {"system", "_context_budget"}
        }
        budgeted = self.budget_manager.apply(budget_sections)
        compressed = dict(structured)
        compressed.update(budgeted.sections)
        compressed["_context_budget"] = budgeted.diagnostics.to_dict()
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_USER_PROMPT.format(
            question=compressed["question"],
            answer_requirement=compressed["answer_requirement"],
            solver_result=compressed["solver_result"],
            attachment_result=compressed["attachment_result"],
            search_result=compressed["search_result"],
        )
        return [
            {"role": "system", "content": str(compressed["system"])},
            {"role": "user", "content": user_content},
        ]


__all__ = [
    "STAGE1_SYSTEM_PROMPT",
    "STAGE1_USER_PROMPT",
    "Stage1ContextBuilder",
]
