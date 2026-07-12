from __future__ import annotations

from dataclasses import asdict, dataclass, field

from utils.network_utils import normalize_text

from .rag_labeler import CONTINUE_TAG, FINISH_TAG, TERMINATE_TAG, RAGLabelResult


@dataclass(frozen=True)
class LabelContractResult:
    """
    Normalize EfficientRAG labeler output into retrieval-control contracts.

    Args:
        - raw_tag: Labeler's original sequence tag.
        - normalized_tag: Supported tag after normalization.
        - useful_tokens: Raw useful token/span outputs.
        - useful_spans: Restored usable spans.
        - label_status: valid_continue / invalid_continue / valid_terminate / terminal_but_unresolved.
        - valid_for_next_hop: Whether useful_spans may drive the next query.
        - valid_for_evidence: Whether this chunk may become final evidence.
        - invalid_reasons: Contract violations.

    Returns:
        - LabelContractResult: Explicit contract used by retrieval control.
    """

    raw_tag: str
    normalized_tag: str
    useful_tokens: list[str] = field(default_factory=list)
    useful_spans: list[str] = field(default_factory=list)
    label_status: str = "terminal_but_unresolved"
    valid_for_next_hop: bool = False
    valid_for_evidence: bool = False
    invalid_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LabelContractValidator:
    """
    Enforce retrieval semantics over raw EfficientRAG labels.

    Args:
        - None.

    Returns:
        - LabelContractValidator: Stateless label contract validator.
    """

    SUPPORTED_TAGS = {CONTINUE_TAG, TERMINATE_TAG, FINISH_TAG}

    def validate(self, result: RAGLabelResult) -> LabelContractResult:
        """
        Validate one RAGLabelResult.

        Args:
            - result: Raw EfficientRAG labeler output.

        Returns:
            - LabelContractResult: Contract-aware label status.
        """
        raw_tag = normalize_text(str(result.metadata.get("sequence_tag", "") or ""))
        normalized_tag = raw_tag if raw_tag in self.SUPPORTED_TAGS else TERMINATE_TAG
        useful_tokens = self._clean_items(result.kept_tokens)
        useful_spans = self._clean_items(
            result.metadata.get("useful_spans") or result.kept_tokens
        )
        has_span = bool(useful_spans)
        invalid_reasons: list[str] = []

        if normalized_tag == CONTINUE_TAG:
            if has_span:
                return LabelContractResult(
                    raw_tag=raw_tag,
                    normalized_tag=normalized_tag,
                    useful_tokens=useful_tokens,
                    useful_spans=useful_spans,
                    label_status="valid_continue",
                    valid_for_next_hop=True,
                    valid_for_evidence=True,
                )
            invalid_reasons.append("continue_without_useful_span")
            return LabelContractResult(
                raw_tag=raw_tag,
                normalized_tag=normalized_tag,
                useful_tokens=useful_tokens,
                useful_spans=useful_spans,
                label_status="invalid_continue",
                valid_for_next_hop=False,
                valid_for_evidence=False,
                invalid_reasons=invalid_reasons,
            )

        if normalized_tag in {TERMINATE_TAG, FINISH_TAG}:
            if has_span:
                return LabelContractResult(
                    raw_tag=raw_tag,
                    normalized_tag=normalized_tag,
                    useful_tokens=useful_tokens,
                    useful_spans=useful_spans,
                    label_status="valid_terminate",
                    valid_for_next_hop=False,
                    valid_for_evidence=True,
                )
            invalid_reasons.append("terminal_without_answer_span")
            return LabelContractResult(
                raw_tag=raw_tag,
                normalized_tag=normalized_tag,
                useful_tokens=useful_tokens,
                useful_spans=useful_spans,
                label_status="terminal_but_unresolved",
                valid_for_next_hop=False,
                valid_for_evidence=False,
                invalid_reasons=invalid_reasons,
            )

        invalid_reasons.append("unsupported_sequence_tag")
        return LabelContractResult(
            raw_tag=raw_tag,
            normalized_tag=normalized_tag,
            useful_tokens=useful_tokens,
            useful_spans=useful_spans,
            label_status="terminal_but_unresolved",
            invalid_reasons=invalid_reasons,
        )

    def _clean_items(self, items: object) -> list[str]:
        if not isinstance(items, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = normalize_text(str(item or ""))
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result


__all__ = ["LabelContractResult", "LabelContractValidator"]
