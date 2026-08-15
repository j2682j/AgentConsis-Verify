"""Rebuild the evidence funnel from recorded retrieval, using production code.

The funnel between retrieval and the Agent's prompt is where task 045's answer
died -- Whisper heard every page number and a header ate the budget before the
transcript reached Stage 1. Tasks 004 and 013 look like the same shape: the gold
string is in the prepared evidence and not in what the Agent saw. Finding which
stage drops it needs the stages re-run, not re-approximated.

So every step here calls the production object. `EvidenceConverter`,
`BestEffortReferenceSelector`, the renderer and `ContextBudgetManager` are the
ones the benchmark runs; a reimplementation would answer a question about the
reimplementation.

What must never be an input is the funnel's own output. The recorded
`raw_result` carries `evidence_items`, `unverified_references` and `summary`
alongside the retrieval it was built from, and feeding those back in would
reproduce them perfectly while testing nothing -- the same circularity that made
a requirement-gate repair measure as a no-op for five runs. `retrieval_input`
strips them, and `OUTPUT_FIELDS` names them so a test can assert their absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from context.context_budget import ContextBudgetManager
from core.evidence_runner import EvidenceRunner

#: Produced by the funnel. Present in the recording, forbidden as input.
OUTPUT_FIELDS = (
    "evidence_items",
    "verified_evidence_items",
    "unverified_references",
    "answer_candidates",
    "summary",
)


def digest(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]


def retrieval_input(raw_result: dict[str, Any]) -> dict[str, Any]:
    """The recorded trace with everything the funnel produced removed."""

    return {
        key: value
        for key, value in (raw_result or {}).items()
        if key not in OUTPUT_FIELDS
    }


@dataclass
class FunnelStage:
    name: str
    fidelity: str = "unsupported"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunnelReplay:
    task_id: str
    gold: str
    stages: list[FunnelStage] = field(default_factory=list)
    retrieved_document_ids: list[str] = field(default_factory=list)
    strict_evidence_ids: list[str] = field(default_factory=list)
    strict_evidence_text_hashes: list[str] = field(default_factory=list)
    relaxed_reference_ids: list[str] = field(default_factory=list)
    relaxed_reference_text_hashes: list[str] = field(default_factory=list)
    rendered_search_context: str = ""
    budgeted_stage1_context: str = ""

    @property
    def rendered_search_context_hash(self) -> str:
        return digest(self.rendered_search_context)

    @property
    def budgeted_stage1_context_hash(self) -> str:
        return digest(self.budgeted_stage1_context)


def _runner(question: str) -> EvidenceRunner:
    """An EvidenceRunner with only what the funnel methods touch.

    Built without `__init__` on purpose: the real constructor wires tool
    managers and model clients, none of which the funnel uses, and half of them
    would reach the network.
    """

    from tools.evidence.fact_extraction import TaskFactStore
    from tools.search_result_builder.evidence import (
        BestEffortReferenceSelector,
        EvidenceConverter,
        SpanBuilder,
    )

    runner = EvidenceRunner.__new__(EvidenceRunner)
    runner.question = question
    # Constructed the way `EvidenceRunner.__init__` does, so the converter gets
    # its span builder rather than whatever its own default happens to be.
    runner.span_builder = SpanBuilder()
    runner.evidence_converter = EvidenceConverter(span_builder=runner.span_builder)
    runner.best_effort_reference_selector = BestEffortReferenceSelector()
    runner.fact_store = TaskFactStore()
    return runner


def rebuild(task: dict[str, Any], *, task_id: str) -> FunnelReplay:
    """Run the recorded retrieval back through the production funnel."""

    gold = str(task.get("expected") or "")
    replay = FunnelReplay(task_id=task_id, gold=gold)
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    raw = next(
        (
            item.get("raw_result")
            for item in (meta.get("tool_usage") or [])
            if isinstance(item, dict)
            and item.get("tool_name") == "search"
            and isinstance(item.get("raw_result"), dict)
        ),
        None,
    )
    if not raw:
        replay.stages.append(FunnelStage("documents", "unsupported"))
        return replay

    output_dict = retrieval_input(raw)
    rounds = (output_dict.get("retrieval") or {}).get("rounds") or []
    replay.retrieved_document_ids = [
        str(document.get("document_id") or "")
        for entry in rounds
        for document in (entry.get("documents") or [])
        if isinstance(document, dict)
    ]
    replay.stages.append(
        FunnelStage(
            "documents",
            "exact" if replay.retrieved_document_ids else "unsupported",
            {"document_count": len(replay.retrieved_document_ids)},
        )
    )

    runner = _runner(str(output_dict.get("question") or task.get("question") or ""))
    diagnostics = output_dict.get("diagnostics") or {}
    contract = runner._evidence_selection_contract(output_dict)
    replay.stages.append(
        FunnelStage(
            "contract",
            "exact" if diagnostics.get("query_plan") else "approximate",
            {"rebuilt_from_query_plan": bool(diagnostics.get("query_plan"))},
        )
    )

    evidence_items = runner._web_retrieval_evidence_items(output_dict, contract=contract)
    replay.strict_evidence_ids = [
        str(item.get("evidence_id") or "") for item in evidence_items
    ]
    replay.strict_evidence_text_hashes = [
        digest(str(item.get("evidence_text") or item.get("text") or ""))
        for item in evidence_items
    ]
    replay.stages.append(
        FunnelStage("conversion", "exact", {"evidence_count": len(evidence_items)})
    )

    references = runner._web_retrieval_unverified_references(
        output_dict, evidence_items=evidence_items
    )
    replay.relaxed_reference_ids = [
        str(item.get("reference_id") or "") for item in references
    ]
    replay.relaxed_reference_text_hashes = [
        digest(str(item.get("text") or "")) for item in references
    ]
    replay.stages.append(
        FunnelStage("reference", "exact", {"reference_count": len(references)})
    )

    replay.rendered_search_context = runner._render_web_retrieval_evidence(
        evidence_items,
        unverified_references=references,
        answer_candidates=[],
        contract=contract,
    )
    replay.stages.append(
        FunnelStage(
            "render", "exact", {"chars": len(replay.rendered_search_context)}
        )
    )

    budgeted = ContextBudgetManager().apply(
        {"search_result": replay.rendered_search_context}
    )
    replay.budgeted_stage1_context = budgeted.sections.get("search_result", "")
    replay.stages.append(
        FunnelStage(
            "context",
            "exact",
            {
                "chars": len(replay.budgeted_stage1_context),
                "truncated": budgeted.diagnostics.truncation_applied,
                "dropped_evidence_count": budgeted.diagnostics.dropped_evidence_count,
            },
        )
    )
    return replay


__all__ = [
    "OUTPUT_FIELDS",
    "FunnelReplay",
    "FunnelStage",
    "digest",
    "rebuild",
    "retrieval_input",
]
