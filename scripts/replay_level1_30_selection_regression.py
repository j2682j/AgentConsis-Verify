from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.gaia.answer_matcher import exact_match
from core.config import AgentConfig, AgentReasoningSummary, EachAgentReply, VerifierScoreByReasoning
from core.network import Network
from score.answer_requirement_contract import TaskAnswerRequirementContract
from tools.evidence.fact_extraction import TaskFactStore


def _load_dataclass(cls: type, payload: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _stage1(payload: dict[str, Any]) -> list[AgentReasoningSummary]:
    results = []
    for item in payload["network_summary"]["stage1_results"]:
        data = dict(item)
        data["runs"] = [
            _load_dataclass(EachAgentReply, run) for run in item.get("runs", [])
        ]
        results.append(_load_dataclass(AgentReasoningSummary, data))
    return results


def _verifiers(payload: dict[str, Any]) -> list[VerifierScoreByReasoning]:
    return [
        _load_dataclass(VerifierScoreByReasoning, item)
        for item in payload["network_summary"].get("verifier_results", [])
    ]


def _attach_saved_versa(paths, verifiers) -> None:
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
    stage1 = _stage1(payload)
    verifiers = _verifiers(payload)
    metadata = payload["network_summary"].get("metadata", {})
    contract = TaskAnswerRequirementContract.build(question=payload["question"])
    store = TaskFactStore.from_dict(metadata.get("fact_store"))
    evidence = {
        "answer_requirement": contract.requirement_text,
        "answer_role": contract.answer_role,
        "answer_target": contract.answer_target,
        "required_relation": contract.required_relation,
        "required_relation_goal_id": contract.required_relation_goal_id,
        "task_answer_requirement_contract": contract.to_dict(),
        "routing": dict(metadata.get("routing") or {}),
        "tool_usage": list(metadata.get("tool_usage") or []),
        "_fact_store": store,
        "fact_store": store.to_dict(),
    }
    network = Network(
        payload["question"],
        [AgentConfig(agent_id=item.agent_id, model_name=item.model_name) for item in stage1],
        enable_evidence_prepare=False,
        enable_candidate_verification_search=False,
    )
    candidates = network.answer_candidate_clusterer.cluster(
        stage1,
        answer_requirement=contract.requirement_text,
        answer_role=contract.answer_role,
    )
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
    predicted = (
        selection.winner.compressed_answer
        if selection.winner is not None
        else selection.resolved_answer
    )
    expected = str(payload.get("expected") or "")
    return {
        "case": int(path.name.split("_", 1)[0]),
        "original_exact": bool(payload.get("exact_match")),
        "original_predicted": str(payload.get("predicted") or ""),
        "replayed_predicted": predicted,
        "expected": expected,
        "replayed_exact": bool(exact_match(predicted, expected)),
        "selection_status": selection.status,
        "selection_reason": selection.reason,
    }


def main() -> None:
    task_dir = ROOT / "outputs" / "level1_30_system_final" / "tasks"
    results = [replay(path) for path in sorted(task_dir.glob("*.json"))]
    regressions = [
        item for item in results if item["original_exact"] and not item["replayed_exact"]
    ]
    improvements = [
        item for item in results if not item["original_exact"] and item["replayed_exact"]
    ]
    report = {
        "task_count": len(results),
        "original_exact_count": sum(item["original_exact"] for item in results),
        "replayed_exact_count": sum(item["replayed_exact"] for item in results),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "regressions": regressions,
        "improvements": improvements,
        "results": results,
    }
    output = ROOT / "outputs" / "level1_30_selection_regression.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"tasks={len(results)} original_exact={report['original_exact_count']} "
        f"replayed_exact={report['replayed_exact_count']} "
        f"regressions={len(regressions)} improvements={len(improvements)}"
    )
    for item in regressions:
        print(
            f"REGRESSION Q{item['case']}: original={item['original_predicted']!r} "
            f"replay={item['replayed_predicted']!r} expected={item['expected']!r}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
