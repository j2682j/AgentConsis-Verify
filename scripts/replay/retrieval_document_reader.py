"""One reading of a recorded run's retrieval documents, shared by every replay.

Both the attestation diagnostics and the evidence-delivery funnel need to ask
what the corpus actually said, and they must ask it the same way. Two readers
would disagree about what counts as a document -- the same page arrives in
several rounds, so a naive count of occurrences conflates one page indexed
repeatedly with several independent sources saying the same thing.

`documents()` therefore reports both, and `mention_report()` returns document
counts alongside occurrence counts. That distinction is the point: the singular
`rockhopper penguin` occurs 133 times in task 034's record, which says nothing
until you know whether that is one page counted 133 times or many pages
agreeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import json
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    round_index: int
    text: str
    title: str
    url: str
    record_id: str
    duplicate: bool

    @property
    def identity(self) -> str:
        """What makes two entries the same page rather than two sources."""

        return self.url or self.record_id or self.document_id


@dataclass
class MentionHit:
    surface: str
    document_id: str
    record_id: str
    url: str
    round_index: int
    character_span: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "document_id": self.document_id,
            "source_id": self.record_id,
            "url": self.url,
            "round_index": self.round_index,
            "character_span": list(self.character_span),
        }


@dataclass
class MentionReport:
    occurrences: int = 0
    document_count: int = 0
    hits: list[MentionHit] = field(default_factory=list)

    def to_dict(self, *, max_hits: int = 12) -> dict[str, Any]:
        return {
            "occurrences": self.occurrences,
            "document_count": self.document_count,
            "hits": [hit.to_dict() for hit in self.hits[:max_hits]],
        }


def task_path(run: str, task_number: str) -> str:
    matches = glob.glob(f"c:/SCP/outputs/{run}/tasks/{task_number}_*.json")
    if not matches:
        raise FileNotFoundError(f"{run}/{task_number}")
    return matches[0]


def load_task(run: str, task_number: str) -> dict[str, Any]:
    return json.loads(open(task_path(run, task_number), encoding="utf-8").read())


def documents(task: dict[str, Any], *, include_duplicates: bool = True) -> list[RetrievalDocument]:
    """Every document the recorded rounds carry, in round order."""

    rounds = (task.get("search_summary") or {}).get("retrieval_rounds") or []
    out: list[RetrievalDocument] = []
    for entry in rounds:
        index = int(entry.get("round_index") or 0)
        for row in entry.get("documents") or []:
            if not isinstance(row, dict):
                continue
            duplicate = bool(row.get("duplicate"))
            if duplicate and not include_duplicates:
                continue
            out.append(
                RetrievalDocument(
                    document_id=str(row.get("document_id") or ""),
                    round_index=index,
                    text=str(row.get("text") or ""),
                    title=str(row.get("title") or ""),
                    url=str(row.get("url") or row.get("source_url") or ""),
                    record_id=str(row.get("record_id") or ""),
                    duplicate=duplicate,
                )
            )
    return out


def mention_report(
    docs: Iterable[RetrievalDocument],
    surface: str,
    *,
    whole_word: bool = True,
) -> MentionReport:
    """Where a surface form appears, by occurrence and by distinct page.

    Word boundaries by default, because `rockhopper penguin` is a prefix of
    `rockhopper penguins` and counting one inside the other is the mistake the
    whole diagnostic exists to avoid.
    """

    needle = re.escape(str(surface or "").strip())
    if not needle:
        return MentionReport()
    pattern = re.compile(
        rf"(?i)(?<![a-z0-9]){needle}(?![a-z0-9])" if whole_word else f"(?i){needle}"
    )
    report = MentionReport()
    seen_pages: set[str] = set()
    for document in docs:
        found = list(pattern.finditer(document.text))
        if not found:
            continue
        report.occurrences += len(found)
        seen_pages.add(document.identity)
        for match in found:
            report.hits.append(
                MentionHit(
                    surface=document.text[match.start() : match.end()],
                    document_id=document.document_id,
                    record_id=document.record_id,
                    url=document.url,
                    round_index=document.round_index,
                    character_span=(match.start(), match.end()),
                )
            )
    report.document_count = len(seen_pages)
    return report


__all__ = [
    "MentionHit",
    "MentionReport",
    "RetrievalDocument",
    "documents",
    "load_task",
    "mention_report",
    "task_path",
]
