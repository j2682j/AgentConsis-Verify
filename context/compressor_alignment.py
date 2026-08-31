"""Where compressed text sits in the original, in the compressor's own coordinates.

`_compress_multiline_text` strips each line, drops the blank ones, keeps the
first `max_lines`, and cuts the result to `max_chars` with a trailing `" ..."`.
It never rewrites, reorders or generates. Everything it emits is a prefix of the
line-normalised original.

An earlier probe compared whole lines and concluded otherwise: a 68,000-character
block came back `transformed_or_unalignable` at 0.50 line coverage, and the
finding was that long evidence gets rewritten before the budget ever sees it. It
does not. The one line the character cut landed inside no longer equalled its
original, and whole-line equality counted that as unalignable -- measuring the
probe, not the compressor.

So alignment is computed where the compressor works. The original is normalised
the same way, each normalised character keeps a pointer back to the character it
came from, and the compressed payload is matched as a prefix of that. What
survives and what does not is then a matter of offsets rather than similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ELLIPSIS = " ..."

SHAPES = (
    "exact_after_line_normalization",
    "prefix_after_line_normalization",
    "unaligned",
)


@dataclass
class Normalised:
    """The line-normalised text, with every character traceable to the original."""

    text: str
    #: `origin[i]` is the index in the original of `text[i]`.
    origin: list[int] = field(default_factory=list)

    def original_offset(self, index: int) -> int | None:
        if not self.origin:
            return None
        if index >= len(self.origin):
            return self.origin[-1] + 1
        return self.origin[index]


def normalise(raw: str) -> Normalised:
    """Reproduce the compressor's line handling, keeping the offset trail.

    Stripping happens per line and blank lines vanish, so the mapping cannot be
    a simple shift -- indentation removed from line 40 moves everything after it.
    Each surviving character carries its own origin instead.
    """

    text = str(raw or "")
    stripped_prefix = len(text) - len(text.lstrip())
    body = text.strip()

    out: list[str] = []
    origin: list[int] = []
    cursor = stripped_prefix
    first = True
    for line in body.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        lead = len(content) - len(content.lstrip())
        stripped = content.strip()
        if stripped:
            if not first:
                out.append("\n")
                origin.append(cursor)
            first = False
            start = cursor + lead
            for offset, character in enumerate(stripped):
                out.append(character)
                origin.append(start + offset)
        cursor += len(line)
    return Normalised(text="".join(out), origin=origin)


def align(raw: str, compressed: str, *, max_lines: int, max_chars: int) -> dict:
    """Locate the compressed payload inside the normalised original."""

    normalised = normalise(raw)
    # Equality is tested before the marker is stripped, because text can end in
    # `...` on its own. Removing it first turned an untouched block into a
    # prefix and reported a truncation that never happened.
    truncated = compressed != normalised.text and compressed.endswith(ELLIPSIS)
    payload = compressed[: -len(ELLIPSIS)] if truncated else compressed

    # The line cut comes first, so the reachable region is bounded by it even
    # when the character cut never fires.
    lines = normalised.text.split("\n")
    line_limited = "\n".join(lines[:max_lines]).strip()

    if payload == normalised.text:
        shape = "exact_after_line_normalization"
    elif normalised.text.startswith(payload.rstrip()):
        shape = "prefix_after_line_normalization"
    elif line_limited.startswith(payload.rstrip()):
        shape = "prefix_after_line_normalization"
    else:
        shape = "unaligned"

    kept_chars = len(payload.rstrip()) if shape != "unaligned" else 0
    return {
        "shape": shape,
        "truncation_marker_added": truncated,
        "normalised_chars": len(normalised.text),
        "line_limited_chars": len(line_limited),
        "kept_normalised_chars": kept_chars,
        "kept_original_offset_end": normalised.original_offset(kept_chars),
        "raw_chars": len(str(raw or "")),
        "max_lines": max_lines,
        "max_chars": max_chars,
    }


def child_survival(raw: str, alignment: dict, spans: list[tuple[int, int]]) -> list[dict]:
    """Whether each raw span falls inside, across, or beyond the cut.

    Spans are offsets into the raw text, so the cut -- which lives in normalised
    coordinates -- is translated back before they are compared. A span that
    straddles the boundary is `partial`, and saying so matters: a passage half
    delivered is neither kept nor lost.
    """

    boundary = alignment.get("kept_original_offset_end")
    if alignment.get("shape") == "unaligned" or boundary is None:
        return [
            {"start": start, "end": end, "survival": "unsupported",
             "retained_chars": 0, "retained_ratio": 0.0}
            for start, end in spans
        ]
    out = []
    for start, end in spans:
        retained = max(0, min(end, boundary) - start)
        survival = (
            "kept" if end <= boundary
            else "dropped" if start >= boundary
            else "partial"
        )
        out.append({
            "start": start, "end": end, "survival": survival,
            "retained_chars": retained,
            "retained_ratio": round(retained / max(end - start, 1), 4),
        })
    return out


__all__ = [
    "ELLIPSIS",
    "SHAPES",
    "Normalised",
    "align",
    "child_survival",
    "normalise",
]
