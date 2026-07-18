__all__ = ["EvidenceBuilder"]


def __getattr__(name: str):
    if name == "EvidenceBuilder":
        from .builder import EvidenceBuilder

        return EvidenceBuilder
    raise AttributeError(name)
