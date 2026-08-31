from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class ContextBudget:
    max_total_chars: int = 6000
    # The search allowance is `items * chars`, not a count of items kept: the
    # section is fitted to that character budget. At 5 it was 2250, which
    # delivered 3.5 references of which 63% were fragments. At 8 it is 3600,
    # delivering about 4.8 with 83% complete, and the whole prompt still lands
    # near 5400 against the 6000 total. Raising `chars` instead would also
    # deepen each `[E#]` block, which is a separate question.
    max_search_evidence_items: int = 8
    max_search_evidence_chars: int = 450
    max_attachment_chars: int = 1200
    max_tool_result_chars: int = 800
    max_deterministic_chars: int = 800
    max_available_tools_chars: int = 2200
    max_policy_chars: int = 900


def _digest(text: str) -> str:
    """Short content hash, so two runs can be compared without storing both."""

    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16] if text else ""


@dataclass
class ContextBudgetDiagnostics:
    """What the budget did, per prompt section.

    The search evidence is measured on both sides of the budget rather than
    only after it. Reading the prepared evidence and reading what the agent was
    finally shown are different questions, and the run record used to answer
    neither: `tool_context` holds a runtime tool trace under tool use and
    formatted evidence otherwise, so the same field means two things and an
    analysis that reads it gets a plausible wrong answer. It did -- 4 characters
    of `"None"` was mistaken for an empty context when the prompt was 6004
    characters and had dropped nothing.
    """

    original_chars: int = 0
    final_chars: int = 0
    truncation_applied: bool = False
    dropped_evidence_count: int = 0
    truncated_sections: list[str] = field(default_factory=list)
    section_chars: dict[str, int] = field(default_factory=dict)
    #: The prepared search evidence as it arrived, before any budgeting.
    prepared_search_context_chars: int = 0
    prepared_search_context_hash: str = ""
    #: The same section as it appears in the prompt the agent is sent.
    rendered_search_context_chars: int = 0
    rendered_search_context_hash: str = ""
    search_result_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextBudgetResult:
    sections: dict[str, str]
    diagnostics: ContextBudgetDiagnostics


class ContextBudgetManager:
    """
    Apply a deterministic character budget to Stage1 prompt sections.
    """

    _ELLIPSIS = " ..."
    _REFERENCE_SECTION_RE = re.compile(r"(?m)^(?:Unverified References:|\[[BR]\d+\]\s*$)")

    SECTION_LIMITS = {
        "solver_result": "max_deterministic_chars",
        "attachment_result": "max_attachment_chars",
        "attachment_metadata": "max_attachment_chars",
        "tool_trace": "max_tool_result_chars",
        "tool_gap": "max_tool_result_chars",
        "available_tools": "max_available_tools_chars",
        "tool_turn_policy": "max_policy_chars",
    }

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    def apply(self, sections: dict[str, Any]) -> ContextBudgetResult:
        original = {key: self._text(value) for key, value in sections.items()}
        compact = dict(original)
        truncated: list[str] = []
        dropped_evidence_count = 0

        if "search_result" in compact:
            compact_search, dropped = self._compact_search_evidence(compact["search_result"])
            if compact_search != compact["search_result"]:
                truncated.append("search_result")
            compact["search_result"] = compact_search
            dropped_evidence_count += dropped

        for key, limit_attr in self.SECTION_LIMITS.items():
            if key not in compact:
                continue
            limit = int(getattr(self.budget, limit_attr))
            trimmed = self._truncate(compact[key], limit)
            if trimmed != compact[key]:
                truncated.append(key)
            compact[key] = trimmed

        compact, total_truncated = self._fit_total_budget(compact)
        truncated.extend(total_truncated)

        original_chars = sum(len(value) for value in original.values())
        final_chars = sum(len(value) for value in compact.values())
        prepared_search = original.get("search_result", "")
        rendered_search = compact.get("search_result", "")
        diagnostics = ContextBudgetDiagnostics(
            original_chars=original_chars,
            final_chars=final_chars,
            truncation_applied=bool(truncated or dropped_evidence_count or final_chars < original_chars),
            dropped_evidence_count=dropped_evidence_count,
            truncated_sections=sorted(set(truncated)),
            section_chars={key: len(value) for key, value in compact.items()},
            prepared_search_context_chars=len(prepared_search),
            prepared_search_context_hash=_digest(prepared_search),
            rendered_search_context_chars=len(rendered_search),
            rendered_search_context_hash=_digest(rendered_search),
            search_result_truncated=prepared_search != rendered_search,
        )
        return ContextBudgetResult(sections=compact, diagnostics=diagnostics)

    def _search_evidence_budget(self) -> int:
        return self.budget.max_search_evidence_items * self.budget.max_search_evidence_chars

    def _compact_search_evidence(self, text: str) -> tuple[str, int]:
        """Cut the search block to the allowance.

        The plain cut on the `[R#]` reference shape is deliberate. Making it
        block-aware -- so every reference kept its head instead of the first
        few consuming the allowance -- was tried for level1_final_12 and cost
        four tasks: it turns the same allowance into 8 references of about 150
        characters where the plain cut leaves 4 of about 430, and the 4B Agents
        do worse with the shallow spread. The effect was confined to exactly
        the tasks that carry references (21% -> 4% correct) while tasks without
        them held (48% -> 50%), and run-level accuracy rose (27.5% -> 28.8%)
        even as the task score fell, so it was the selection that broke, not
        the Agents. See tests/test_context_budget_reference_blocks.py.
        """

        cleaned = text.strip()
        if not cleaned or cleaned == "None":
            return cleaned, 0

        blocks = self._evidence_blocks(cleaned)
        if not blocks:
            # Strict evidence forms on 3 of 28 retrieval tasks, so this is the
            # path almost every task takes and the one that produced the
            # mid-word cut. References still have their own `[R#]` boundaries
            # even with no `[E#]` block above them, so honour those instead of
            # slicing the section at a character offset.
            if len(cleaned) <= self._search_evidence_budget():
                return cleaned, 0
            head, references = self._split_reference_section(cleaned)
            if references:
                room = self._search_evidence_budget() - len(head) - 1
                if room > 0:
                    kept, _dropped = self._fit_reference_blocks(references, room)
                    if kept:
                        # Reference drops stay out of `dropped`, which counts
                        # evidence blocks; see
                        # tests/test_context_budget_reference_blocks.py.
                        return f"{head}\n{kept}".strip(), 0
            return self._truncate(cleaned, self._search_evidence_budget()), 0

        # Grounded evidence and unverified references can arrive together, and
        # `_evidence_blocks` lets its last block run to the end of the text --
        # which swallows the whole reference section and then trims it to one
        # item's allowance. Measured on level1_final_14 task 046 that left one
        # reference and 848 characters where references alone give three and
        # 2,254. Splitting keeps each side on its own rule: evidence takes its
        # per-item trim, references take a plain cut of whatever the evidence
        # did not use. The total allowance is unchanged, and text without any
        # `[E#]` block never reaches here.
        head, references = self._split_reference_section(cleaned)
        if references:
            kept_head, dropped = self._compact_search_evidence(head)
            room = self._search_evidence_budget() - len(kept_head) - 1
            if room <= 0:
                return kept_head, dropped
            kept_references, dropped_references = self._fit_reference_blocks(references, room)
            return (
                f"{kept_head}\n{kept_references}".strip(),
                dropped + dropped_references,
            )

        prefix = blocks[0] if blocks and not blocks[0].startswith("[E") else ""
        evidence_blocks = blocks[1:] if prefix else blocks
        kept_blocks: list[str] = []
        if prefix:
            kept_blocks.append(prefix)
        for block in evidence_blocks[: self.budget.max_search_evidence_items]:
            kept_blocks.append(self._truncate_evidence_block(block))
        dropped = max(0, len(evidence_blocks) - min(len(evidence_blocks), self.budget.max_search_evidence_items))
        return "\n".join(kept_blocks).strip(), dropped

    def _fit_reference_blocks(self, references: str, room: int) -> tuple[str, int]:
        """Keep whole `[R#]` references; drop the ones that do not fit.

        The section used to be cut as one string at a character offset, which
        landed mid-word on 89% of level1_final_16's retrieval tasks and
        mid-sentence on all of them. Every task ended on a fragment averaging
        294 characters -- 13% of the allowance -- and 14% ended on a reference
        header with no content at all, so the Agents' last piece of evidence
        read `Hiccup would have had to carry 8 ...`.

        This is not the block-aware truncation reverted after level1_final_12.
        That gave every reference its head, turning 4 references of ~430
        characters into 8 of ~150, and the shallower spread cost four tasks.
        Per-reference depth is unchanged here; only the point where the section
        ends moves, from an arbitrary offset to the last complete reference.

        No task's gold sits solely in the discarded fragment, measured across
        the 21 comparable retrieval tasks, so nothing recoverable is lost.
        """

        blocks = self._reference_blocks(references)
        if not blocks:
            return self._truncate(references, room), 0

        kept: list[str] = []
        used = 0
        dropped = 0
        carries_reference = False
        for block in blocks:
            cost = len(block) + (1 if kept else 0)
            is_reference = bool(re.match(r"^\[[BR]\d+\]", block))
            if used + cost <= room:
                kept.append(block)
                used += cost
                carries_reference = carries_reference or is_reference
                continue
            # A first reference longer than the whole allowance is truncated
            # rather than dropped: dropping it leaves only the section header,
            # which tells the reader nothing.
            if is_reference and not carries_reference:
                kept.append(self._truncate(block, max(0, room - used - 1)))
                used = room
                carries_reference = True
                continue
            dropped += 1
        return "\n".join(kept).strip(), dropped

    def _reference_blocks(self, text: str) -> list[str]:
        """The reference section split on its `[R#]`/`[B#]` markers."""

        matches = list(re.finditer(r"(?m)^\[[BR]\d+\]", text))
        if not matches:
            return []
        blocks: list[str] = []
        prefix = text[: matches[0].start()].strip()
        if prefix:
            blocks.append(prefix)
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start() : end].strip()
            if block:
                blocks.append(block)
        return blocks

    def _split_reference_section(self, text: str) -> tuple[str, str]:
        """Everything before the unverified-reference section, and that section."""

        match = self._REFERENCE_SECTION_RE.search(text)
        if match is None:
            return text, ""
        return text[: match.start()].rstrip(), text[match.start() :].strip()

    def _evidence_blocks(self, text: str) -> list[str]:
        matches = list(re.finditer(r"(?m)^\[E\d+\]\s*$", text))
        if not matches:
            return []
        prefix = text[: matches[0].start()].strip()
        blocks: list[str] = [prefix] if prefix and prefix.casefold() != "evidence:" else ["Evidence:"]
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            blocks.append(text[start:end].strip())
        return blocks

    def _truncate_evidence_block(self, block: str) -> str:
        lines = [line.rstrip() for line in block.splitlines()]
        result: list[str] = []
        evidence_seen = False
        evidence_chars = 0
        for line in lines:
            if line.startswith("Evidence:"):
                evidence_seen = True
                prefix = "Evidence: "
                body = line[len("Evidence:") :].strip()
                body = self._truncate(body, self.budget.max_search_evidence_chars)
                evidence_chars = len(body)
                result.append(prefix + body)
                continue
            if evidence_seen:
                remaining = self.budget.max_search_evidence_chars - evidence_chars
                if remaining <= 0:
                    continue
                trimmed = self._truncate(line, remaining)
                evidence_chars += len(trimmed)
                if trimmed:
                    result.append(trimmed)
            else:
                result.append(line)
        return "\n".join(result).strip()

    def _fit_total_budget(self, sections: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        compact = dict(sections)
        truncated: list[str] = []
        total = sum(len(value) for value in compact.values())
        if total <= self.budget.max_total_chars:
            return compact, truncated

        shrink_order = [
            "tool_trace",
            "available_tools",
            "attachment_result",
            "search_result",
            "solver_result",
            "tool_gap",
            "tool_turn_policy",
        ]
        for key in shrink_order:
            if total <= self.budget.max_total_chars:
                break
            value = compact.get(key, "")
            if not value or value == "None":
                continue
            excess = total - self.budget.max_total_chars
            target = max(160, len(value) - excess)
            trimmed = self._truncate(value, target)
            if trimmed != value:
                compact[key] = trimmed
                truncated.append(key)
                total = sum(len(item) for item in compact.values())
        return compact, truncated

    def _truncate(self, text: str, max_chars: int) -> str:
        value = self._text(text).strip()
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + self._ELLIPSIS

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value)


__all__ = [
    "ContextBudget",
    "ContextBudgetDiagnostics",
    "ContextBudgetManager",
    "ContextBudgetResult",
]
