from __future__ import annotations

from ..config import AgentEvidencePacket


class AgentEvidenceRenderer:
    """
    Render compact verified evidence packets for Stage1 and Stage2 prompts.

    Args:
        - None.

    Returns:
        - AgentEvidenceRenderer: Prompt renderer for AgentEvidencePacket.
    """

    def render(self, packet: AgentEvidencePacket) -> str:
        """
        Convert an AgentEvidencePacket into compact prompt text.

        Args:
            - packet: Compact evidence packet.

        Returns:
            - str: Prompt-ready compact evidence text.
        """
        lines = [
            "Original Question:",
            packet.question,
            "",
            f"Answer Type: {packet.answer_type or 'unknown'}",
            "",
            "Verified Candidate Answer:",
        ]

        if packet.candidates:
            for candidate in packet.candidates:
                risk = ", ".join(candidate.risk_flags) or "-"
                lines.append(
                    f"[{candidate.candidate_id}] answer={candidate.answer}; "
                    f"type={candidate.answer_type}; "
                    f"support={candidate.support_count}; "
                    f"refute={candidate.refute_count}; "
                    f"neutral={candidate.neutral_count}; "
                    f"confidence={candidate.confidence}; "
                    f"risk={risk}"
                )
        else:
            lines.append("None")

        lines.extend(["", "Fact:"])
        if packet.facts:
            for fact in packet.facts:
                constraints = ", ".join(fact.constraint_matches) or "-"
                relation_text = {
                    "support": "supports",
                    "refute": "refutes",
                    "neutral": "is neutral for",
                }.get(fact.relation, fact.relation)
                lines.extend(
                    [
                        f"[{fact.fact_id}] {relation_text} {fact.candidate_id}; confidence={fact.confidence}",
                        f"Claim: {fact.claim}",
                        f"Constraint Match: {constraints}",
                    ]
                )
        else:
            lines.append("None")

        lines.extend(["", "Missing Info:"])
        lines.append(", ".join(packet.missing_info) if packet.missing_info else "None")

        lines.extend(["", "Risk Flags:"])
        lines.append(", ".join(packet.risk_flags) if packet.risk_flags else "None")

        return "\n".join(lines).strip()


__all__ = ["AgentEvidenceRenderer"]

