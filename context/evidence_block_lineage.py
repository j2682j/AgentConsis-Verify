"""Which evidence blocks survived the budget, block by block.

Asking whether an answer survived truncation by searching the trimmed text for
it does not work, and cannot be made to work. It needs a rule per answer -- the
digit `3` is meaningless on its own, `research` matches any sentence about
research, `Claus` sits inside `Clausen` -- and a table of such rules is a table
about one benchmark, not a property of the system.

So the question changes. Instead of looking for the answer in the output, each
evidence block is followed from the prepared text into the rendered text and its
fate recorded: kept whole, trimmed, or gone. Whatever the block carried goes
with it. Nothing here reads a gold answer, and the same code works on any task,
level or dataset.

Lineage is derived by comparing the two texts rather than by instrumenting the
compactor. That keeps it unable to change what the compactor does -- the paths
it takes are load-bearing and were tuned against measured regressions -- and it
means the same function reconstructs lineage for runs recorded months ago, which
an instrumented compactor never could.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

#: `[E1]`, `[R12]` at the start of a line. The compactor splits on these, so the
#: lineage has to agree with it about where a block begins.
BLOCK_MARKER = re.compile(r"^\[(?P<kind>[ER])(?P<index>\d+)\]", re.MULTILINE)

DISPOSITIONS = ("kept", "truncated", "dropped")


def digest(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16] if text else ""


@dataclass(frozen=True)
class EvidenceBlock:
    """One `[E#]` or `[R#]` block, addressed by its own marker."""

    block_id: str
    kind: str
    index: int
    text: str
    start: int
    end: int

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def text_hash(self) -> str:
        return digest(self.text)


@dataclass
class BlockLineage:
    """What became of one block between prepared and rendered text."""

    block_id: str
    kind: str
    index: int
    original_chars: int
    original_text_hash: str
    rendered_chars: int
    rendered_text_hash: str
    disposition: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LineageResult:
    blocks: list[BlockLineage] = field(default_factory=list)

    def to_dict(self) -> dict:
        counts = {name: 0 for name in DISPOSITIONS}
        for block in self.blocks:
            counts[block.disposition] += 1
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "block_count": len(self.blocks),
            "kept_count": counts["kept"],
            "truncated_count": counts["truncated"],
            "dropped_count": counts["dropped"],
            # The one number a delivery question turns on: whether anything the
            # budget removed might have carried the answer.
            "lost_block_ids": [
                b.block_id for b in self.blocks if b.disposition != "kept"
            ],
        }


def parse_blocks(text: str) -> list[EvidenceBlock]:
    """Split a search section into its marked blocks, ignoring the preamble."""

    body = str(text or "")
    matches = list(BLOCK_MARKER.finditer(body))
    blocks: list[EvidenceBlock] = []
    for position, match in enumerate(matches):
        start = match.start()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        kind, index = match.group("kind"), int(match.group("index"))
        blocks.append(
            EvidenceBlock(
                block_id=f"{kind}{index}",
                kind=kind,
                index=index,
                text=body[start:end].strip(),
                start=start,
                end=end,
            )
        )
    return blocks


def trace(prepared: str, rendered: str) -> LineageResult:
    """Follow every prepared block into the rendered text.

    A block is `kept` when its text survives unchanged, `truncated` when its
    marker is still there but the body is shorter, and `dropped` when the marker
    is gone. The marker is what identifies it: the compactor keeps `[R3]` as
    `[R3]`, so a block that no longer appears under its own marker was removed
    rather than renumbered.
    """

    rendered_blocks = {b.block_id: b for b in parse_blocks(rendered)}
    lineage: list[BlockLineage] = []
    for block in parse_blocks(prepared):
        survivor = rendered_blocks.get(block.block_id)
        if survivor is None:
            disposition, rendered_chars, rendered_hash = "dropped", 0, ""
        elif survivor.text_hash == block.text_hash:
            disposition = "kept"
            rendered_chars, rendered_hash = survivor.chars, survivor.text_hash
        else:
            disposition = "truncated"
            rendered_chars, rendered_hash = survivor.chars, survivor.text_hash
        lineage.append(
            BlockLineage(
                block_id=block.block_id,
                kind=block.kind,
                index=block.index,
                original_chars=block.chars,
                original_text_hash=block.text_hash,
                rendered_chars=rendered_chars,
                rendered_text_hash=rendered_hash,
                disposition=disposition,
            )
        )
    return LineageResult(blocks=lineage)


__all__ = [
    "BLOCK_MARKER",
    "DISPOSITIONS",
    "BlockLineage",
    "EvidenceBlock",
    "LineageResult",
    "digest",
    "parse_blocks",
    "trace",
]
