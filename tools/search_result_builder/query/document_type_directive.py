from __future__ import annotations

"""Detect explicit document-type directives stated by the question.

A question that names a specific document kind ("the official script",
"press release", "patent") is not satisfied by topically-related pages such
as fan transcripts or news coverage; the answer lives inside one specific
kind of document. Detection here only ADDS a targeted query variant and a
fetch-content preference — it never removes or rewrites existing queries, so
questions without a directive are completely unaffected.

Deliberately excluded: bare "PDF"/"the PDF" mentions, which in GAIA usually
refer to an attached file rather than a web document to search for.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentTypeDirective:
    """One detected document-type requirement from the question text."""

    directive: str
    type_terms: str
    required_content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "directive": self.directive,
            "type_terms": self.type_terms,
            "required_content": self.required_content,
        }


_DIRECTIVE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\bofficial\s+script\b|\bshooting\s+script\b|\bscreenplay\b|\bscene\s+heading\b",
            re.IGNORECASE,
        ),
        "official script PDF",
        "pdf_text",
    ),
    (
        re.compile(r"\bofficial\s+transcript\b", re.IGNORECASE),
        "official transcript",
        "html_text",
    ),
    (
        re.compile(r"\bpress\s+release\b", re.IGNORECASE),
        "press release",
        "html_text",
    ),
    (
        re.compile(r"\bpatent\b", re.IGNORECASE),
        "patent document",
        "pdf_text",
    ),
    (
        re.compile(r"\bannual\s+report\b", re.IGNORECASE),
        "annual report PDF",
        "pdf_text",
    ),
)


def detect_document_type_directive(question: str) -> DocumentTypeDirective | None:
    """Return the first matched document-type directive, if any."""

    text = str(question or "")
    if not text.strip():
        return None
    for pattern, type_terms, required_content in _DIRECTIVE_RULES:
        match = pattern.search(text)
        if match:
            return DocumentTypeDirective(
                directive=match.group(0).lower(),
                type_terms=type_terms,
                required_content=required_content,
            )
    return None


__all__ = ["DocumentTypeDirective", "detect_document_type_directive"]
