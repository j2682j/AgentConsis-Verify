from score.evidence_support_level import (
    EvidenceSupportLevel,
    compare_support_levels,
    support_level_for_status,
)


def test_status_mapping_uses_single_support_contract() -> None:
    assert support_level_for_status("tool_final_supported") is EvidenceSupportLevel.TRUSTED_TOOL_FINAL
    assert support_level_for_status("derived_evidence_supported") is EvidenceSupportLevel.VERIFIED_DERIVED
    assert support_level_for_status("search_evidence_supported") is EvidenceSupportLevel.DIRECT_EVIDENCE
    assert support_level_for_status("tool_intermediate_supported") is EvidenceSupportLevel.BRIDGE_EVIDENCE
    assert support_level_for_status("unknown_status") is EvidenceSupportLevel.UNSUPPORTED


def test_support_level_comparison_is_ordinal_not_weighted() -> None:
    assert compare_support_levels("direct_evidence", "unsupported") == 1
    assert compare_support_levels("contradicted", "contradicted") == 0
    assert compare_support_levels("bridge_evidence", "verified_derived") == -1
