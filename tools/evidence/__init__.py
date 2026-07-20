__all__ = [
    "EvidenceBuilder",
    "EvidenceReadiness",
    "EvidenceReadinessEvaluator",
    "EvidenceReadinessStatus",
]


def __getattr__(name: str):
    if name == "EvidenceBuilder":
        from .builder import EvidenceBuilder

        return EvidenceBuilder
    if name in {
        "EvidenceReadiness",
        "EvidenceReadinessEvaluator",
        "EvidenceReadinessStatus",
    }:
        from .evidence_readiness import (
            EvidenceReadiness,
            EvidenceReadinessEvaluator,
            EvidenceReadinessStatus,
        )

        return {
            "EvidenceReadiness": EvidenceReadiness,
            "EvidenceReadinessEvaluator": EvidenceReadinessEvaluator,
            "EvidenceReadinessStatus": EvidenceReadinessStatus,
        }[name]
    raise AttributeError(name)
