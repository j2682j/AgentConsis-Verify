from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parsers.reasoning_parser import extract_reasoning_steps
from score.versa_prm_scorer import (
    DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    DEFAULT_VERSA_PRM_MODEL_ID,
    VersaPRMScorer,
)


DEFAULT_TASK_PREFIXES = "002,005,016,019,020,022,023,026,029"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score saved Stage1 agent reasoning steps with VersaPRM without rerunning GAIA."
        )
    )
    parser.add_argument(
        "--tasks-dir",
        default=r"C:\SCP\outputs\level1_30_wo_stage2_v2\tasks",
        help="Directory containing GAIA task JSON files.",
    )
    parser.add_argument(
        "--task-prefixes",
        default=DEFAULT_TASK_PREFIXES,
        help="Comma-separated task number prefixes, e.g. 002,005,016.",
    )
    parser.add_argument(
        "--output-json",
        default=r"C:\SCP\outputs\wrong_consensus_versa_scores.json",
    )
    parser.add_argument(
        "--output-md",
        default=r"C:\SCP\outputs\wrong_consensus_versa_scores.md",
    )
    parser.add_argument("--model-id", default=DEFAULT_VERSA_PRM_MODEL_ID)
    parser.add_argument("--base-model-id", default=DEFAULT_VERSA_PRM_BASE_MODEL_ID)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--unload",
        action="store_true",
        help="Unload VersaPRM after all saved reasoning has been scored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_dir = Path(args.tasks_dir)
    task_prefixes = [
        item.strip() for item in str(args.task_prefixes or "").split(",") if item.strip()
    ]
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    scorer = VersaPRMScorer(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    records: list[dict[str, Any]] = []
    for prefix in task_prefixes:
        task_path = find_task_path(tasks_dir, prefix)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        records.append(score_task(task=task, task_path=task_path, scorer=scorer))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "source_tasks_dir": str(tasks_dir),
                "task_prefixes": task_prefixes,
                "model_id": args.model_id,
                "base_model_id": args.base_model_id,
                "tasks": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(records), encoding="utf-8")

    unload_status = {}
    if args.unload:
        unload_status = scorer.unload()

    print(f"[OK] wrote {output_json}")
    print(f"[OK] wrote {output_md}")
    if unload_status:
        print(f"[OK] unload={unload_status}")
    return 0


def find_task_path(tasks_dir: Path, prefix: str) -> Path:
    matches = sorted(tasks_dir.glob(f"{prefix}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No task JSON found for prefix {prefix!r} in {tasks_dir}")
    return matches[0]


def score_task(
    *,
    task: dict[str, Any],
    task_path: Path,
    scorer: VersaPRMScorer,
) -> dict[str, Any]:
    network_summary = task.get("network_summary", {}) or {}
    agents = []
    for stage_result in network_summary.get("stage1_results", []) or []:
        agent_id = stage_result.get("agent_id", "")
        model_name = stage_result.get("model_name", "")
        runs = []
        for run in stage_result.get("runs", []) or []:
            steps = reasoning_steps_for_run(run)
            score_result = scorer.score_steps(
                question=str(task.get("question", "")),
                reasoning_steps=steps,
            )
            runs.append(
                {
                    "run_index": run.get("run_index"),
                    "final_answer": final_answer_for_run(stage_result, run),
                    "step_count": len(steps),
                    "avg_reward_probability": round(
                        score_result.avg_reward_probability, 6
                    ),
                    "reward_count": score_result.metadata.get("reward_count", 0),
                    "reward_count_matches_step_count": score_result.metadata.get(
                        "reward_count_matches_step_count", False
                    ),
                    "step_scores": [
                        {
                            "step_index": item.step_index,
                            "step_text": item.step_text,
                            "reward_probability": item.reward_probability,
                        }
                        for item in score_result.step_scores
                    ],
                    "metadata": dict(score_result.metadata),
                }
            )
        agents.append(
            {
                "agent_id": agent_id,
                "model_name": model_name,
                "runs": runs,
            }
        )
    return {
        "task_prefix": task_path.name.split("_", 1)[0],
        "task_id": task.get("task_id", ""),
        "question": task.get("question", ""),
        "predicted": task.get("predicted", ""),
        "expected": task.get("expected", ""),
        "exact_match": task.get("exact_match", False),
        "winner_agent_id": task.get("winner_agent_id", ""),
        "json_path": str(task_path),
        "agents": agents,
    }


def reasoning_steps_for_run(run: dict[str, Any]) -> list[tuple[int, str]]:
    reasoning = str(run.get("reasoning") or "").strip()
    raw_reply = str(run.get("raw_reply") or "").strip()
    steps = extract_reasoning_steps(reasoning)
    if steps:
        return steps
    steps = extract_reasoning_steps(raw_reply)
    if steps:
        return steps
    text = " ".join((reasoning or raw_reply).split())
    return [(1, text)] if text else []


def final_answer_for_run(
    stage_result: dict[str, Any],
    run: dict[str, Any],
) -> str:
    return str(
        stage_result.get("final_answer")
        or stage_result.get("answer")
        or stage_result.get("compressed_answer")
        or run.get("final_answer")
        or run.get("answer")
        or run.get("compressed_answer")
        or run.get("parsed_answer")
        or ""
    )


def render_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Wrong Consensus VersaPRM Scores",
        "",
        "每個 Agent 的 reasoning steps 都送入 VersaPRM，表中的 reward probability 保留原始 0~1 機率。",
        "",
    ]
    for task in records:
        lines.extend(
            [
                f"## Q{task['task_prefix']}. {task['task_id']}",
                "",
                f"- Predicted: `{task['predicted']}`",
                f"- Expected: `{task['expected']}`",
                f"- Winner: `{task['winner_agent_id']}`",
                f"- JSON: `{task['json_path']}`",
                "",
                "**Question**",
                "",
                str(task["question"]).strip(),
                "",
            ]
        )
        for agent in task["agents"]:
            lines.extend(
                [
                    f"### Agent: {agent['agent_id']} / {agent['model_name']}",
                    "",
                ]
            )
            for run in agent["runs"]:
                lines.extend(
                    [
                        f"- Run: `{run['run_index']}`",
                        f"- Final answer: `{run['final_answer']}`",
                        f"- Avg reward probability: `{run['avg_reward_probability']}`",
                        f"- Step count / reward count: `{run['step_count']}` / `{run['reward_count']}`",
                        f"- Reward count matches step count: `{run['reward_count_matches_step_count']}`",
                        "",
                        "| Step | Reward probability | Step text |",
                        "|---:|---:|---|",
                    ]
                )
                for step in run["step_scores"]:
                    step_text = str(step["step_text"]).replace("|", "\\|")
                    lines.append(
                        f"| {step['step_index']} | {step['reward_probability']} | {step_text} |"
                    )
                lines.append("")
        lines.extend(["---", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
