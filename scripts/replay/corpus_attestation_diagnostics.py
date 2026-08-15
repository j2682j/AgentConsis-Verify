"""Why does corpus attestation report zero for an answer the corpus states?

Task 034's gold is `Rockhopper penguin`. Attestation counted `penguins` 12,
`penguin` 11 and `Rockhopper penguins` 0, and reserved the specific candidate --
three supporting runs -- in favour of the generic one with a single run. Zero is
the number that needs explaining, because the same task's retrieval text does
contain the species name.

There are two different faults that both produce zero, and they call for
opposite repairs:

  WINDOW_BLIND       the exact form is in the corpus but past the 200,000
                     character cut `_corpus_mention_counts` applies
  MORPHOLOGY_BLIND   the exact form is nowhere, but an inflection of it is,
                     so the candidate is attested and the counter cannot see it

Guessing between them risks loosening a gate that measures 8 helps to 2 hurts
across five runs. This module only measures; it changes no gate.

Occurrence counts alone cannot settle it either, since the same page arrives in
several retrieval rounds -- hence the document counts, which separate one page
indexed repeatedly from several sources agreeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from scripts.replay.retrieval_document_reader import (
    RetrievalDocument,
    documents,
    load_task,
    mention_report,
)

#: `_corpus_mention_counts(max_corpus_chars=200_000)`.
PRODUCTION_WINDOW_CHARS = 200_000


from score.surface_form_morphology import (  # noqa: E402
    MIN_INFLECTED_WORD_CHARS,
    singular_variants as morphological_variants,
)


@dataclass
class CandidateAttestation:
    candidate: str
    production_mentions: int
    full_corpus_exact_mentions: int = 0
    attestation_window_exact_mentions: int = 0
    canonical_morph_mentions: int = 0
    exact_match_document_count: int = 0
    canonical_match_document_count: int = 0
    matched_surface_forms: list[dict[str, Any]] = field(default_factory=list)
    classification: str = "UNCLASSIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "production_mentions": self.production_mentions,
            "full_corpus_exact_mentions": self.full_corpus_exact_mentions,
            "attestation_window_exact_mentions": self.attestation_window_exact_mentions,
            "canonical_morph_mentions": self.canonical_morph_mentions,
            "exact_match_document_count": self.exact_match_document_count,
            "canonical_match_document_count": self.canonical_match_document_count,
            "matched_surface_forms": self.matched_surface_forms,
            "classification": self.classification,
        }


def classify(row: CandidateAttestation) -> str:
    full = row.full_corpus_exact_mentions
    window = row.attestation_window_exact_mentions
    morph = row.canonical_morph_mentions

    if window > 0 and row.production_mentions == 0:
        return "COUNTING_IMPLEMENTATION_BUG"
    if full > 0 and window == 0 and morph > 0:
        return "WINDOW_AND_MORPHOLOGY_BLIND"
    if full > 0 and window == 0:
        return "WINDOW_BLIND"
    if full == 0 and morph > 0:
        return "MORPHOLOGY_BLIND"
    if full == 0 and morph == 0:
        return "TRULY_UNATTESTED"
    return "ATTESTED"


def _windowed(docs: list[RetrievalDocument]) -> tuple[list[RetrievalDocument], int, bool]:
    """The documents production would have seen inside its character cap."""

    kept: list[RetrievalDocument] = []
    total = 0
    truncated = False
    for document in docs:
        if total >= PRODUCTION_WINDOW_CHARS:
            truncated = True
            break
        kept.append(document)
        total += len(document.text)
    if total > PRODUCTION_WINDOW_CHARS:
        truncated = True
    return kept, total, truncated


def diagnose(run: str, task_number: str) -> dict[str, Any]:
    task = load_task(run, task_number)
    docs = documents(task)
    window_docs, scanned_chars, truncated = _windowed(docs)

    meta = (task.get("network_summary") or {}).get("metadata") or {}
    trace = ((meta.get("winner_selection") or {}).get("selection_trace") or {})
    production: dict[str, int] = {}
    for gate in trace.get("gate_trace") or []:
        if gate.get("gate_name") != "corpus_attestation":
            continue
        for decision in gate.get("decisions") or []:
            details = decision.get("details") or {}
            if "corpus_mentions" in details:
                production[str(decision.get("candidate_key"))] = int(
                    details["corpus_mentions"] or 0
                )

    rows: list[CandidateAttestation] = []
    for candidate in trace.get("candidates") or []:
        answer = str(candidate.get("answer") or "")
        key = str(candidate.get("candidate_key") or "")
        row = CandidateAttestation(
            candidate=answer,
            production_mentions=production.get(key, 0),
        )
        full = mention_report(docs, answer)
        window = mention_report(window_docs, answer)
        row.full_corpus_exact_mentions = full.occurrences
        row.exact_match_document_count = full.document_count
        row.attestation_window_exact_mentions = window.occurrences
        row.matched_surface_forms = [hit.to_dict() for hit in full.hits[:8]]

        morph_documents: set[str] = set()
        for variant in morphological_variants(answer):
            report = mention_report(docs, variant)
            row.canonical_morph_mentions += report.occurrences
            morph_documents.update(hit.url or hit.document_id for hit in report.hits)
            row.matched_surface_forms.extend(hit.to_dict() for hit in report.hits[:4])
        row.canonical_match_document_count = len(morph_documents)
        row.classification = classify(row)
        rows.append(row)

    return {
        "source_run": run,
        "task": task_number,
        "gold": str(task.get("expected") or ""),
        "documents_scanned": len(docs),
        "documents_in_window": len(window_docs),
        "scanned_chars": scanned_chars,
        "corpus_truncated": truncated,
        "candidates": [row.to_dict() for row in rows],
    }


__all__ = [
    "PRODUCTION_WINDOW_CHARS",
    "CandidateAttestation",
    "classify",
    "diagnose",
    "morphological_variants",
]
