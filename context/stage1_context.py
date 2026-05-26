from __future__ import annotations

from typing import Any

from .context_builder import ContextBuilder, ContextPacket


STAGE1_SYSTEM_PROMPT = """You are one agent in a multi-agent reasoning network.

Solve the question independently using the provided evidence.
Use Solver_Result, Attachment_Result, and Search_Result only when relevant.
If evidence is missing or uncertain, say only what can be supported.
Write reasoning as explicit numbered steps so peer agents can judge each step separately.
Return only the required format."""


STAGE1_USER_PROMPT = """Question:
{question}

Solver_Result:
{solver_result}

Attachment_Result:
{attachment_result}

Search_Result:
{search_result}

Instructions:
- Solve the question independently.
- Prefer Solver_Result when it directly answers the question.
- Prefer Attachment_Result for file-based questions.
- Use Search_Result for external factual questions.
- If evidence conflicts, explain the conflict briefly in REASONING and choose the best-supported answer.
- REASONING must be written as explicit numbered steps, one step per line.
- Use exactly this step style:
  step 1. ...
  step 2. ...
  step 3. ...
  step N. ...
- FINAL_ANSWER must be short and contain only the answer requested by the question.
- Do not add extra explanation after FINAL_ANSWER.

Return exactly this format:
REASONING =
step 1. first reasoning step
step 2. second reasoning step
step 3. third reasoning step
step N. final reasoning step
FINAL_ANSWER = final answer only"""


class Stage1ContextBuilder(ContextBuilder):
    """Build Stage1 agent chat messages from question and evidence packets."""

    REQUIRED_PACKET_TYPES = {"question", "system_instruction"}
    EVIDENCE_PACKET_TYPES = {"search_result", "attachment_result", "solver_result"}

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
            "search_result": self.config.none_text,
            "attachment_result": self.config.none_text,
            "solver_result": self.config.none_text,
        }
        buckets = {
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
        return compressed

    def render(self, compressed: dict[str, Any], **_: Any) -> list[dict[str, str]]:
        user_content = STAGE1_USER_PROMPT.format(
            question=compressed["question"],
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
