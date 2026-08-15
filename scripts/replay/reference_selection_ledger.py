"""Which rule kept the answer out of the eight references?

The funnel replay showed tasks 004 and 013 losing their gold between retrieval
and the reference list: the string is in the fetched documents and not in what
Stage 1 received. That says the stage, not the reason. Reference selection has
several ways to drop a passage and they call for different repairs -- a gold
passage ranked ninth is a ranking problem, one displaced by relaxed candidates
that filled all eight slots is a quota problem, and one cut by the 900-character
per-item trim is neither.

`_web_retrieval_unverified_references` merges two sources into one list:

    converter.last_relaxed_references   ranked by question-term coverage
    BestEffortReferenceSelector.select  ranked by retrieval score

relaxed entries are offered first, best-effort fills what is left, and both are
subject to a duplicate-text check, a 7,200-character total and a hard cap of
eight.

The merge loop is simulated here rather than called, because the production one
records nothing about what it rejected. `verify_against_production` then asserts
the simulation produced the same references, so the ledger describes the real
selection and not a plausible imitation of it.

Gold is matched only after every ordering decision is made. It labels the ledger;
it never participates in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from utils.network_utils import normalize_text

#: The order matters: the first condition a candidate fails is what blocked it.
BLOCKERS = (
    "selected",
    "empty_text",
    "duplicate_content",
    "char_budget",
    "outer_reference_cap",
    "not_a_candidate",
)


@dataclass
class LedgerEntry:
    branch: str
    position_in_branch: int
    document_id: str
    url: str
    domain: str
    record_type: str
    retrieval_score: float
    text_length: int
    truncated: bool
    gold_in_source: bool
    gold_survived_truncation: bool
    selected: bool = False
    selected_position: int | None = None
    first_blocker: str = "not_a_candidate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "position_in_branch": self.position_in_branch,
            "document_id": self.document_id,
            "url": self.url,
            "domain": self.domain,
            "record_type": self.record_type,
            "retrieval_score": self.retrieval_score,
            "text_length": self.text_length,
            "truncated": self.truncated,
            "gold_in_source": self.gold_in_source,
            "gold_survived_truncation": self.gold_survived_truncation,
            "selected": self.selected,
            "selected_position": self.selected_position,
            "first_blocker": self.first_blocker,
        }


def carries(haystack: str, gold: str) -> bool:
    needle = re.sub(r"\s+", " ", str(gold or "")).strip().casefold()
    if not needle:
        return False
    text = re.sub(r"\s+", " ", str(haystack or "")).casefold()
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def _domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", str(url or ""))
    return match.group(1).casefold() if match else ""


def build_ledger(
    *,
    relaxed: list[dict[str, Any]],
    best_effort: list[dict[str, Any]],
    max_reference_chars: int,
    max_references: int,
    gold: str,
) -> tuple[list[LedgerEntry], list[dict[str, Any]]]:
    """Replay the merge, recording why each candidate did or did not survive.

    Mirrors `_web_retrieval_unverified_references` exactly, including the order
    of its checks -- truncate, then duplicate, then total budget -- because the
    first check a candidate fails is the answer this ledger exists to give.
    """

    total_budget = max_reference_chars * max_references
    used_chars = 0
    seen_texts: set[str] = set()
    entries: list[LedgerEntry] = []
    produced: list[dict[str, Any]] = []

    for branch, payloads in (("relaxed", relaxed), ("best_effort", best_effort)):
        for position, payload in enumerate(payloads, start=1):
            raw_text = str(payload.get("text") or "")
            text = normalize_text(raw_text)
            truncated = len(text) > max_reference_chars
            if truncated:
                text = text[:max_reference_chars].rstrip() + "..."
            entry = LedgerEntry(
                branch=branch,
                position_in_branch=position,
                document_id=str(
                    payload.get("document_id") or payload.get("reference_id") or ""
                ),
                url=str(payload.get("url") or payload.get("source_url") or ""),
                domain=_domain(payload.get("url") or payload.get("source_url") or ""),
                record_type=str(payload.get("record_type") or ""),
                retrieval_score=float(payload.get("retrieval_score") or 0.0),
                text_length=len(text),
                truncated=truncated,
                gold_in_source=carries(raw_text, gold),
                gold_survived_truncation=carries(text, gold),
            )

            if len(produced) >= max_references:
                entry.first_blocker = "outer_reference_cap"
            elif not text:
                entry.first_blocker = "empty_text"
            elif text.casefold()[:400] in seen_texts:
                entry.first_blocker = "duplicate_content"
            elif used_chars + len(text) > total_budget:
                entry.first_blocker = "char_budget"
            else:
                seen_texts.add(text.casefold()[:400])
                used_chars += len(text)
                entry.selected = True
                entry.selected_position = len(produced) + 1
                entry.first_blocker = "selected"
                emitted = dict(payload)
                emitted["text"] = text
                emitted["reference_id"] = f"R{len(produced) + 1}"
                produced.append(emitted)
            entries.append(entry)
    return entries, produced


def verify_against_production(
    simulated: list[dict[str, Any]], produced: list[dict[str, Any]]
) -> bool:
    """The simulation is only evidence if it matches what production emitted."""

    if len(simulated) != len(produced):
        return False
    return all(
        str(a.get("text") or "") == str(b.get("text") or "")
        and str(a.get("reference_id") or "") == str(b.get("reference_id") or "")
        for a, b in zip(simulated, produced)
    )


__all__ = ["BLOCKERS", "LedgerEntry", "build_ledger", "carries", "verify_against_production"]
