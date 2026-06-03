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
    ) -> str:
        """
        Convert an EvidenceOutput object into a stable prompt context block.

        Args:
            - output: Evidence-oriented search result bundle.
            - max_evidence_items: Maximum evidence chunks to render.
        Returns:
            - str: Prompt-ready search evidence context.
        """
        lines = [
            "Original Question:",
            output.question,
            "",
            "Query:",
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
                        f"Query: {item.query_id}",
                        f"Text: {item.text}",
                    ]
                )
        else:
            lines.append("None")

        return "\n".join(lines).strip()


__all__ = ["EvidenceRenderer"]
