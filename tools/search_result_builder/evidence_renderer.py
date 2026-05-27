from __future__ import annotations

from .config import EvidenceOutput


class EvidenceRenderer:
    """
    Render EvidenceOutput into compact text for Stage1 agents and Stage2 judges.

    Args:
        - None.

    Returns:
        - EvidenceRenderer: Formatter for evidence-oriented search output.
    """

    def render(
        self,
        output: EvidenceOutput,
        *,
        max_evidence_items: int = 8,
        max_candidates: int = 3,
    ) -> str:
        """
        Convert an EvidenceOutput object into a stable prompt context block.

        Args:
            - output: Evidence-oriented search result bundle.
            - max_evidence_items: Maximum evidence chunks to render.
            - max_candidates: Maximum candidate answers to render.

        Returns:
            - str: Prompt-ready search evidence context.
        """
        lines = [
            "Search evidence bundle:",
            f"Question focus: {output.question}",
            "",
            "Queries:",
        ]

        if output.queries:
            for plan in output.queries:
                lines.append(f"[{plan.query_id}] {plan.query}")
        else:
            lines.append("None")

        lines.extend(["", "Evidence:"])
        if output.evidence_items:
            for item in output.evidence_items[:max_evidence_items]:
                lines.extend(
                    [
                        f"[{item.evidence_id}]",
                        f"Source: {item.title or item.source_id}",
                        f"URL: {item.url}",
                        f"Query: {item.query_id}",
                        f"Relevance: {round(item.relevance_score, 3)}",
                        f"Text: {item.text}",
                    ]
                )
        else:
            lines.append("None")

        lines.extend(["", "Candidate answers:"])
        if output.candidates:
            for index, candidate in enumerate(output.candidates[:max_candidates], start=1):
                evidence_ids = ", ".join(candidate.evidence_ids) or "-"
                lines.append(
                    f"[C{index}] answer={candidate.answer}; "
                    f"type={candidate.answer_type}; "
                    f"support={candidate.support_count}; "
                    f"verification={candidate.verification_score}; "
                    f"evidence={evidence_ids}"
                )
        else:
            lines.append("None")

        if output.blocked_sources:
            lines.extend(["", "Filtered sources:"])
            for source in output.blocked_sources[:5]:
                lines.append(f"- {source.url} ({source.block_reason})")

        return "\n".join(lines).strip()


__all__ = ["EvidenceRenderer"]
