from __future__ import annotations

import json
from typing import Any

from .context_builder import ContextBuilder, ContextPacket

"""
Stage2 Judge 需要評分「這個 Agent 的每一步 reasoning 是否被 tool evidence 支持？」
"""


STAGE2_SYSTEM_PROMPT = """
You are a strict scoring judge in a multi-agent reasoning network.
"""


STAGE2_SCORING_RUBRIC = """

Scoring Rules:
1.0:
- Correct, useful, and directly supported by the question or Evidence.
- Never assign 1.0 to unsupported claims.

0.5:
- Mostly correct but incomplete.
- Partially supported by Evidence or the question.

0.0:
- Unclear, redundant, or cannot be judged.
- Evidence is missing and the step cannot be verified from the question alone.

-0.5:
- Unsupported.
- Skips an important check.
- Weakly conflicts with Evidence.
- Claims a tool result that is not present in Evidence.

-1.0:
- Contradicts Evidence.
- Invents Evidence.
- Misuses tool results.
- Supports a wrong or malformed final answer.
- Evidence says A, but reasoning/final answer says B.

Compact Evidence Rules:
- If Verified Candidate Answer is present, judge whether the reasoning selects the best supported candidate.
- If a Fact supports a candidate, reward reasoning that uses that candidate correctly.
- If a Fact refutes a candidate, score steps supporting that candidate -1.0.
- If reasoning invents an answer outside verified candidates, score at most -0.5 unless evidence clearly requires it.
- If all candidates are marked weak or missing, do not reward unsupported guessing."""


STAGE2_USER_PROMPT = """Question:
{question}

Target_Agent_Final_Answer:
{target_answer}

Target_Agent_Reasoning:
{target_reasoning}

Evidence:
{target_tool_evidence}

Task:
Score each reasoning step using the evidence.
{scoring_rubric}


Return exactly this JSON shape:
{{"step_scores": [{{"step": 1, "score": 0}}, {{"step": 2, "score": 0}}]}}"""


class Stage2ContextBuilder(ContextBuilder):
    """
    收集Stage1的輸出，以及Stage2需要的system instruction，構建成Stage2 Judge的輸入格式
    """

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
        """
        收集並優先排序不同來源的上下文信息（如問題、目標答案、推理過程和工具證據)

        Args:
            question: 用戶提出的問題
            target_answer: 目標Agent的最終答案
            target_reasoning: 目標Agent的推理過程
            target_tool_evidence: 目標Agent使用工具的證據（如API調用結果）
            system_instructions: 可選的系統指令，覆蓋默認的系統提示
            context_packets: 來自其他來源的額外上下文

        Returns:
            一個ContextPacket列表，按照優先級排序，供後續選擇和結構化使用
        """
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
        """
        從收集到的ContextPacket中選擇對Stage2 Judge最有用的部分，並按照優先級排序

        Args:
            packets: 從gather方法收集到的ContextPacket列表

        Returns:
            一個ContextPacket列表，包含對Stage2 Judge最有用的信息
        """
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
            scoring_rubric=STAGE2_SCORING_RUBRIC,
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
    "STAGE2_SCORING_RUBRIC",
    "STAGE2_SYSTEM_PROMPT",
    "STAGE2_USER_PROMPT",
    "Stage2ContextBuilder",
]
