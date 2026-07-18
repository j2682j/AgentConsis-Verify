from __future__ import annotations

from enum import Enum


class EvidenceSupportLevel(str, Enum):
    """Canonical evidence strength used by evaluation and winner selection."""

    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    BRIDGE_EVIDENCE = "bridge_evidence"
    DIRECT_EVIDENCE = "direct_evidence"
    VERIFIED_DERIVED = "verified_derived"
    TRUSTED_TOOL_FINAL = "trusted_tool_final"


_LEVEL_ORDER = (
    EvidenceSupportLevel.CONTRADICTED,
    EvidenceSupportLevel.UNSUPPORTED,
    EvidenceSupportLevel.BRIDGE_EVIDENCE,
    EvidenceSupportLevel.DIRECT_EVIDENCE,
    EvidenceSupportLevel.VERIFIED_DERIVED,
    EvidenceSupportLevel.TRUSTED_TOOL_FINAL,
)

_STATUS_LEVELS = {
    "contradicted": EvidenceSupportLevel.CONTRADICTED,
    "invalid": EvidenceSupportLevel.UNSUPPORTED,
    "tool_failed_model_only": EvidenceSupportLevel.UNSUPPORTED,
    "no_support": EvidenceSupportLevel.UNSUPPORTED,
    "tool_intermediate_supported": EvidenceSupportLevel.BRIDGE_EVIDENCE,
    "search_evidence_supported": EvidenceSupportLevel.DIRECT_EVIDENCE,
    "attachment_evidence_supported": EvidenceSupportLevel.DIRECT_EVIDENCE,
    "derived_evidence_supported": EvidenceSupportLevel.VERIFIED_DERIVED,
    "tool_final_supported": EvidenceSupportLevel.TRUSTED_TOOL_FINAL,
}


def support_level_for_status(status: str) -> EvidenceSupportLevel:
    """Map a detailed checker status to the shared ordinal support level."""

    return _STATUS_LEVELS.get(
        str(status or "").strip().lower(),
        EvidenceSupportLevel.UNSUPPORTED,
    )


def support_level_rank(level: EvidenceSupportLevel | str) -> int:
    """Return the ordinal rank without introducing a weighted support score."""

    try:
        normalized = EvidenceSupportLevel(level)
    except ValueError:
        normalized = EvidenceSupportLevel.UNSUPPORTED
    return _LEVEL_ORDER.index(normalized)


def compare_support_levels(
    left: EvidenceSupportLevel | str,
    right: EvidenceSupportLevel | str,
) -> int:
    """Compare two support levels and return -1, 0, or 1."""

    left_rank = support_level_rank(left)
    right_rank = support_level_rank(right)
    return (left_rank > right_rank) - (left_rank < right_rank)


__all__ = [
    "EvidenceSupportLevel",
    "compare_support_levels",
    "support_level_for_status",
    "support_level_rank",
]
