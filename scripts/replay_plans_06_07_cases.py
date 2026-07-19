from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply, VerifierScoreByReasoning
from core.network import Network
from score.answer_requirement_contract import TaskAnswerRequirementContract
from tools.attachment_reader_tool import AttachmentReaderTool
from tools.deterministic_handlers import DeterministicHandlerRouter
from tools.evidence.fact_extraction import (
    AttachmentFactExtractor,
    DirectEvidencePromoter,
    EvidenceFact,
    SemanticExtractionResult,
    TaskFactStore,
)
from utils.network_utils import normalize_for_exact


CASE_NUMBERS = (4, 8, 10, 17, 19, 15)


class _NoModelSemanticExtractor:
    """Keep replay offline; plans 06/07 are evaluated without new model calls."""

    def extract_batch(self, **_: Any) -> SemanticExtractionResult:
        return SemanticExtractionResult(diagnostics={"replay_no_model": True})


class _NoModelCrossContextExtractor:
    """Skip cross-context model calls while preserving the extractor contract."""

    def extract_windows(self, **_: Any) -> SemanticExtractionResult:
        return SemanticExtractionResult(diagnostics={"replay_no_model": True})


def _dataclass_from_dict(cls: type, payload: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _stage1_results(payload: dict[str, Any]) -> list[AgentReasoningSummary]:
    output: list[AgentReasoningSummary] = []
    for item in payload["network_summary"]["stage1_results"]:
        runs = [_dataclass_from_dict(EachAgentReply, run) for run in item.get("runs", [])]
        summary_payload = dict(item)
        summary_payload["runs"] = runs
        output.append(_dataclass_from_dict(AgentReasoningSummary, summary_payload))
    return output


def _verifiers(payload: dict[str, Any]) -> list[VerifierScoreByReasoning]:
    return [
        _dataclass_from_dict(VerifierScoreByReasoning, item)
        for item in payload["network_summary"].get("verifier_results", [])
    ]


def _rebuild_search_facts(payload: dict[str, Any], store: TaskFactStore) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    promoter = DirectEvidencePromoter()
    for usage in payload["network_summary"]["metadata"].get("tool_usage", []):
        if usage.get("tool_name") != "search":
            continue
        for evidence_item in (usage.get("raw_result") or {}).get("evidence_items", []):
            for contract in evidence_item.get("direct_contracts", []):
                result = promoter.promote(
                    model_role="ANSWER_SUPPORT",
                    candidate_span=str(contract.get("answer_span") or ""),
                    context=str(contract.get("context") or evidence_item.get("text") or ""),
                    question=payload["question"],
                    answer_requirement=payload["question"],
                    answer_target=str(contract.get("answer_requirement") or ""),
                    source_id=str(contract.get("url") or evidence_item.get("url") or ""),
                    source_title=str(contract.get("source_title") or evidence_item.get("title") or ""),
                    document_id=str(contract.get("document_id") or ""),
                    goal_id=str(contract.get("goal_id") or ""),
                    semantic_facts=[],
                )
                store.extend(result.promoted_facts)
                diagnostics.extend(item.to_dict() for item in result.diagnostics)
    return diagnostics


def _rebuild_attachment_facts(payload: dict[str, Any], store: TaskFactStore) -> dict[str, Any]:
    attachment = dict(payload.get("attachment") or {})
    path = str(attachment.get("file_path") or "")
    if not path or not Path(path).is_file():
        return {"status": "not_available"}
    extension = Path(path).suffix.casefold()
    if extension == ".docx":
        parsed = AttachmentReaderTool().run(
            {"question": payload["question"], "file_path": path}
        )["parsed_payload"]
        extractor = AttachmentFactExtractor(
            semantic_extractor=_NoModelSemanticExtractor(),
            cross_context_extractor=_NoModelCrossContextExtractor(),
        )
        result = extractor.extract(
            question=payload["question"],
            answer_requirement=payload["question"],
            parsed_payload=parsed,
        )
        store.extend(result.facts)
        for contract in result.diagnostics.get("completeness_contracts", []):
            from tools.evidence.fact_extraction import CompletenessContract

            store.add_completeness_contract(CompletenessContract.from_dict(contract))
        return dict(result.diagnostics)
    if extension in {".xlsx", ".xlsm"}:
        handler = DeterministicHandlerRouter().run(
            question=payload["question"],
            attachment=attachment,
            handler_name="color_grid_hamiltonian",
        )
        if handler.ok:
            store.add(
                EvidenceFact(
                    fact_id="replay-color-grid-hamiltonian",
                    subject="owned color-grid graph",
                    relation="has_hamiltonian_cycle",
                    object=handler.answer,
                    qualifiers={
                        "answer_binding": "direct",
                        "operation": "hamiltonian_cycle",
                    },
                    role="ANSWER_SUPPORT",
                    evidence_spans=[handler.evidence_text],
                    context=handler.evidence_text,
                    source_id=path,
                    source_type="handler",
                    grounding_status="grounded",
                    extraction_method="trusted_handler",
                )
            )
        return handler.to_dict()
    return {"status": "not_rebuilt_for_extension", "extension": extension}


def _retain_applicable_legacy_handler_facts(payload: dict[str, Any], store: TaskFactStore) -> None:
    for item in (payload["network_summary"]["metadata"].get("fact_store") or {}).get("facts", []):
        fact = EvidenceFact.from_dict(item)
        if fact.extraction_method != "deterministic_adapter":
            continue
        relation = fact.relation.casefold()
        question = payload["question"].casefold()
        if "logic_equivalence" in relation and "translate" in question:
            continue
        store.add(fact)


def _attach_saved_versa(paths: list[Any], verifiers: list[VerifierScoreByReasoning]) -> None:
    index = {
        (item.target_agent_id, int(item.metadata.get("target_run_index") or 0)): item
        for item in verifiers
    }
    for path in paths:
        verifier = index.get((path.identity.agent_id, path.identity.run_index))
        if verifier is None:
            continue
        process = dict(verifier.metadata.get("process_verification") or {})
        path.versa_available = bool(process)
        path.versa_status = "available" if process else "unavailable"
        path.critical_step_floor = float(process.get("critical_step_floor") or 0.0)
        path.critical_step_geometric_mean = float(
            process.get("critical_step_geometric_mean") or 0.0
        )
        path.average_verifier_probability = float(
            process.get("average_probability") or verifier.verifier_score or 0.0
        )
        path.versa_step_scores = list(verifier.step_scores or [])


def replay(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage1 = _stage1_results(payload)
    verifiers = _verifiers(payload)
    agents = [AgentConfig(agent_id=item.agent_id, model_name=item.model_name) for item in stage1]
    network = Network(
        payload["question"],
        agents,
        attachment=payload.get("attachment"),
        enable_evidence_prepare=False,
    )
    contract = TaskAnswerRequirementContract.build(question=payload["question"])
    candidates = network.answer_candidate_clusterer.cluster(
        stage1,
        answer_requirement=contract.requirement_text,
        answer_role=contract.answer_role,
    )
    store = TaskFactStore()
    search_diagnostics = _rebuild_search_facts(payload, store)
    attachment_diagnostics = _rebuild_attachment_facts(payload, store)
    _retain_applicable_legacy_handler_facts(payload, store)
    evidence = {
        "answer_requirement": contract.requirement_text,
        "answer_role": contract.answer_role,
        "answer_target": contract.answer_target,
        "task_answer_requirement_contract": contract.to_dict(),
        "routing": payload["network_summary"]["metadata"].get("routing", {}),
        "tool_usage": [],
        "_fact_store": store,
        "fact_store": store.to_dict(),
    }
    bundle = network.candidate_path_evaluator.evaluate_candidates(
        candidates=candidates,
        stage1_results=stage1,
        evidence=evidence,
        enable_versa=False,
        evidence_revision=store.revision,
    )
    _attach_saved_versa(bundle.path_evaluations, verifiers)
    selection = network.final_winner_selector.select(
        stage1_results=stage1,
        candidates=candidates,
        path_evaluations=bundle.path_evaluations,
        verifier_results=verifiers,
        evidence=evidence,
    )
    predicted = selection.winner.compressed_answer if selection.winner else ""
    expected = str(payload.get("expected") or "")
    return {
        "case": int(path.name.split("_", 1)[0]),
        "task_id": payload["task_id"],
        "expected": expected,
        "original_predicted": payload.get("predicted", ""),
        "replayed_predicted": predicted,
        "normalized_match": normalize_for_exact(predicted) == normalize_for_exact(expected),
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "contract": contract.to_dict(),
        "candidate_states": [
            {
                "answer": item.answer,
                "support_status": item.support_status,
                "selection_state": item.selection_state,
                "soft_deferred_by": item.soft_deferred_by,
            }
            for item in selection.evaluations
        ],
        "fact_count": len(store.all()),
        "verifiable_facts": [
            item.to_dict() for item in store.verifiable_answer_facts()
        ],
        "search_promotion_diagnostics": search_diagnostics,
        "attachment_diagnostics": attachment_diagnostics,
    }


def main() -> None:
    task_dir = ROOT / "outputs" / "level1_full_system_final" / "tasks"
    results = []
    for case_number in CASE_NUMBERS:
        task_path = next(task_dir.glob(f"{case_number:03d}_*.json"))
        result = replay(task_path)
        results.append(result)
        print(
            f"Q{case_number}: original={result['original_predicted']!r} "
            f"replay={result['replayed_predicted']!r} expected={result['expected']!r} "
            f"status={result['selection_status']}"
        )
    output = ROOT / "outputs" / "plans_06_07_six_case_replay.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
