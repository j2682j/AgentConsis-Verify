from .answer_role_compatibility import (
    AnswerRoleCompatibilityGate,
    AnswerRoleCompatibilityResult,
)
from .answer_candidate import AnswerCandidate, EvidenceAnswerExtractor
from .candidate_span_quality_gate import (
    CandidateSpanQualityGate,
    CandidateSpanQualityResult,
)
from .candidate_span_grounder import (
    CandidateSpanExpander,
    CandidateSpanGrounder,
    CandidateSpanGroundingResult,
    GroundedCandidateSpan,
)
from .evidence_contract import EvidenceSelectionContract
from .evidence_role_contract import (
    BridgeEvidenceContract,
    DirectEvidenceContract,
    EvidenceRoleContractBuilder,
    EvidenceRoleContracts,
    RejectedEvidenceSpan,
)
from .evidence_converter import EvidenceConversionDiagnostics, EvidenceConverter
from .evidence_utility_gate import EvidenceUtilityGate, EvidenceUtilityResult
from .passage_evidence_unit_builder import (
    PassageEvidenceUnit,
    PassageEvidenceUnitBuilder,
    PassageEvidenceUnitResult,
)
from tools.evidence.fact_extraction import (
    AnswerValueCanonicalizer,
    DirectEvidencePromoter,
    DirectEvidencePromotionResult,
    GroundedAnswerValue,
    PromotionDiagnostic,
)
from .role_aware_span_finalizer import (
    FinalizedSpan,
    RoleAwareSpanFinalizationResult,
    RoleAwareSpanFinalizer,
)
from .span_builder import EvidenceSpan, SpanBuilder
from .span_recovery import RecoveredSpans, SpanRecovery
from .span_role_classifier import (
    ANSWER_SUPPORT,
    BRIDGE,
    NOISE,
    CandidateSpan,
    SpanRoleBatchResult,
    SpanRoleClassifier,
    SpanRoleResult,
)

__all__ = [
    "AnswerCandidate",
    "AnswerRoleCompatibilityGate",
    "AnswerRoleCompatibilityResult",
    "CandidateSpanQualityGate",
    "CandidateSpanQualityResult",
    "CandidateSpanExpander",
    "CandidateSpanGrounder",
    "CandidateSpanGroundingResult",
    "EvidenceSelectionContract",
    "BridgeEvidenceContract",
    "DirectEvidenceContract",
    "EvidenceRoleContractBuilder",
    "EvidenceRoleContracts",
    "RejectedEvidenceSpan",
    "EvidenceConversionDiagnostics",
    "EvidenceConverter",
    "EvidenceAnswerExtractor",
    "EvidenceUtilityGate",
    "EvidenceUtilityResult",
    "PassageEvidenceUnit",
    "PassageEvidenceUnitBuilder",
    "PassageEvidenceUnitResult",
    "AnswerValueCanonicalizer",
    "DirectEvidencePromoter",
    "DirectEvidencePromotionResult",
    "GroundedAnswerValue",
    "PromotionDiagnostic",
    "EvidenceSpan",
    "FinalizedSpan",
    "ANSWER_SUPPORT",
    "BRIDGE",
    "NOISE",
    "CandidateSpan",
    "GroundedCandidateSpan",
    "RecoveredSpans",
    "RoleAwareSpanFinalizationResult",
    "RoleAwareSpanFinalizer",
    "SpanRoleBatchResult",
    "SpanRoleClassifier",
    "SpanRoleResult",
    "SpanBuilder",
    "SpanRecovery",
]
