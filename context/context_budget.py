from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class ContextBudget:
    max_total_chars: int = 6000
    max_search_evidence_items: int = 5
    max_search_evidence_chars: int = 450
    max_attachment_chars: int = 1200
    max_tool_result_chars: int = 800
    max_deterministic_chars: int = 800
    max_available_tools_chars: int = 2200
    max_policy_chars: int = 900


@dataclass
class ContextBudgetDiagnostics:
    original_chars: int = 0
    final_chars: int = 0
    truncation_applied: bool = False
    dropped_evidence_count: int = 0
    truncated_sections: list[str] = field(default_factory=list)
    section_chars: dict[str, int] = field(default_factory=dict)

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
        diagnostics = ContextBudgetDiagnostics(
            original_chars=original_chars,
            final_chars=final_chars,
            truncation_applied=bool(truncated or dropped_evidence_count or final_chars < original_chars),
            dropped_evidence_count=dropped_evidence_count,
            truncated_sections=sorted(set(truncated)),
            section_chars={key: len(value) for key, value in compact.items()},
        )
        return ContextBudgetResult(sections=compact, diagnostics=diagnostics)

    def _compact_search_evidence(self, text: str) -> tuple[str, int]:
        cleaned = text.strip()
        if not cleaned or cleaned == "None":
            return cleaned, 0

        blocks = self._evidence_blocks(cleaned)
        if not blocks:
            return self._truncate(cleaned, self.budget.max_search_evidence_items * self.budget.max_search_evidence_chars), 0

        prefix = blocks[0] if blocks and not blocks[0].startswith("[E") else ""
        evidence_blocks = blocks[1:] if prefix else blocks
        kept_blocks: list[str] = []
        if prefix:
            kept_blocks.append(prefix)
        for block in evidence_blocks[: self.budget.max_search_evidence_items]:
            kept_blocks.append(self._truncate_evidence_block(block))
        dropped = max(0, len(evidence_blocks) - min(len(evidence_blocks), self.budget.max_search_evidence_items))
        return "\n".join(kept_blocks).strip(), dropped

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
        return value[:max_chars].rstrip() + " ..."

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value)


__all__ = [
    "ContextBudget",
    "ContextBudgetDiagnostics",
    "ContextBudgetManager",
    "ContextBudgetResult",
]
