from __future__ import annotations

import json
from typing import Any

from .context_builder import ContextBuilder, ContextPacket

"""
Stage2 Judge 需要評分「這個 Agent 的每一步 reasoning 是否被 tool evidence 支持？」
"""


STAGE2_SYSTEM_PROMPT = """You are a strict scoring judge in a multi-agent reasoning network.

Your only job is to assign numeric scores to the target reasoning steps.
Do not critique, explain, summarize, justify, or discuss.
No markdown. No prose. No extra keys."""


STAGE2_USER_PROMPT = """Question:
{question}

Target_Agent_Final_Answer:
{target_answer}

Target_Agent_Reasoning:
{target_reasoning}

Target_Tool_Evidence:
{target_tool_evidence}

Task:
Score each reasoning step using the evidence.

Scoring:
1.0 = correct, useful, and supported by the question or evidence.
0.5 = mostly correct but incomplete or only partially supported.
0.0 = unclear, redundant, or cannot be judged.
-0.5 = unsupported, skips an important check, or weakly conflicts with evidence.
-1.0 = contradicts evidence, invents evidence, misuses tool results, or supports a wrong/malformed final answer.

Rules:
- Do not reward reasoning that is plausible but unsupported.
- If evidence says A but final answer or reasoning says B, score the related step -1.0.
- If a step claims a tool result that is not in Evidence, score at most -0.5.
- If Evidence is empty, judge only from the question, final answer, and reasoning.
- If the final answer is malformed, score steps supporting it at most -0.5.
- Output JSON only.

Return exactly this JSON shape:
{{"step_scores": [{{"step": 1, "score": 0}}, {{"step": 2, "score": 0}}]}}"""


class Stage2ContextBuilder(ContextBuilder):
    """Build Stage2 judge chat messages for scoring another agent's reasoning."""

    REQUIRED_PACKET_TYPES = {
        "question",
        "system_instruction",
        "target_answer",
        "target_tool_evidence",
        "target_reasoning",
    }

    def gather(
        self,
        *,
        question: str,
        target_answer: str,
        target_reasoning: str,
        target_tool_evidence: Any | None = None,
        system_instructions: str | None = None,
        context_packets: list[ContextPacket] | None = None,
        **_: Any,
    ) -> list[ContextPacket]:
        packets = [
            ContextPacket(
                packet_type="question",
                content=self._normalize_text(question),
                priority=100,
                metadata={"source": "user"},
            ),
            ContextPacket(
                packet_type="target_answer",
                content=self._normalize_text(target_answer),
                priority=90,
                metadata={"source": "target_agent"},
            ),
            ContextPacket(
                packet_type="target_reasoning",
                content=str(target_reasoning or "").strip(),
                priority=80,
                metadata={"source": "target_agent"},
            ),
            ContextPacket(
                packet_type="target_tool_evidence",
                content=self._serialize_tool_evidence(target_tool_evidence),
                priority=70,
                metadata={"source": "target_agent"},
            ),
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

        packets.extend(context_packets or [])
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
            "system": STAGE2_SYSTEM_PROMPT,
            "question": "",
            "target_answer": self.config.none_text,
            "target_reasoning": self.config.none_text,
            "target_tool_evidence": self.config.none_text,
        }

        for packet in packets:
            content = packet.content.strip()
            if packet.packet_type == "system_instruction" and content:
                structured["system"] = content
            elif packet.packet_type == "question":
                structured["question"] = content
            elif packet.packet_type == "target_answer" and content:
                structured["target_answer"] = content
            elif packet.packet_type == "target_reasoning" and content:
                structured["target_reasoning"] = content
            elif packet.packet_type == "target_tool_evidence" and content:
                structured["target_tool_evidence"] = content

        return structured

    def compress(self, structured: dict[str, Any], **_: Any) -> dict[str, Any]:
        compressed = dict(structured)
        compressed["target_reasoning"] = (
            self._compress_multiline_text(
                compressed["target_reasoning"],
                max_lines=self.config.max_context_lines,
                max_chars=self.config.max_context_chars,
            )
            or self.config.none_text
        )
        compressed["target_tool_evidence"] = (
            self._compress_multiline_text(
                compressed["target_tool_evidence"],
                max_lines=self.config.max_context_lines,
                max_chars=self.config.max_context_chars,
            )
            or self.config.none_text
        )
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE2_USER_PROMPT.format(
            question=compressed["question"],
            target_answer=compressed["target_answer"],
            target_reasoning=compressed["target_reasoning"],
            target_tool_evidence=compressed["target_tool_evidence"],
        )
        return [
            {"role": "system", "content": str(compressed["system"])},
            {"role": "user", "content": user_content},
        ]

    def _serialize_tool_evidence(self, tool_evidence: Any | None) -> str:
        if not tool_evidence:
            return self.config.none_text
        if isinstance(tool_evidence, str):
            return tool_evidence.strip() or self.config.none_text
        try:
            return json.dumps(tool_evidence, ensure_ascii=False, indent=2)
        except TypeError:
            return str(tool_evidence)


__all__ = [
    "STAGE2_SYSTEM_PROMPT",
    "STAGE2_USER_PROMPT",
    "Stage2ContextBuilder",
]
