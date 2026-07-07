from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.gaia.dataset import GAIADataset
from benchmark.gaia.gaia_runner import build_attachment
from core.evidence_runner import EvidenceRunner


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GAIA evidence preparation only, without Stage1 or Stage2.",
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "gaia"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument(
        "--log-name",
        default="gaia_level1_30_evidence_only_after_tool_capability",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-evidence-driven-search", action="store_true")
    parser.add_argument("--no-deterministic-handler-router", action="store_true")
    parser.add_argument("--no-tool-planner", action="store_true")
    parser.add_argument("--max-parallel-next-hop-queries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.log_name
    task_dir = output_root / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    items = GAIADataset(
        split=args.split,
        level=args.level,
        local_data_dir=args.data_dir,
    ).load()[: max(0, args.max_samples)]

    records: list[dict[str, Any]] = []
    for index, sample in enumerate(items, 1):
        record = run_one_sample(index=index, total=len(items), sample=sample, args=args)
        safe_task_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in str(record.get("task_id", ""))
        ) or f"task_{index:03d}"
        (task_dir / f"{index:03d}_{safe_task_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)

    summary_path = output_root / f"{args.log_name}.md"
    summary_path.write_text(render_summary(records, args=args), encoding="utf-8")
    print(f"\n[OK] Summary written: {summary_path}", flush=True)
    print(f"[OK] Task JSON dir: {task_dir}", flush=True)


def run_one_sample(
    *,
    index: int,
    total: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.time()
    task_id = str(sample.get("task_id", ""))
    print(f"\n===== Evidence Prepare {index}/{total} task_id={task_id} =====", flush=True)

    record: dict[str, Any] = {
        "index": index,
        "task_id": task_id,
        "level": sample.get("level", args.level),
        "question": sample.get("question", ""),
        "expected": sample.get("final_answer", ""),
        "elapsed_seconds": 0.0,
        "error": "",
        "tool_names": [],
        "tool_errors": [],
        "search_queries": [],
        "evidence_count": 0,
        "retrieval_stop_reason": "",
        "final_intent_state": "",
        "tool_plan": {},
        "search_result": "",
        "attachment_result": "",
        "solver_result": "",
        "tool_usage": [],
    }

    try:
        result = EvidenceRunner(
            question=str(sample.get("question", "")),
            attachment=build_attachment(sample),
            enable_evidence_driven_search=not args.no_evidence_driven_search,
            enable_deterministic_handler_router=not args.no_deterministic_handler_router,
            enable_tool_planner=not args.no_tool_planner,
            max_parallel_next_hop_queries=args.max_parallel_next_hop_queries,
        ).run()

        search_raw = first_tool_raw_result(result, "search")
        retrieval = search_raw.get("retrieval") or {}
        rounds = retrieval.get("rounds") or []
        states = [
            ((round_info.get("coverage") or {}).get("intent_state") or {}).get("state", "")
            for round_info in rounds
        ]

        tool_plan = {}
        planner_raw = first_tool_raw_result(result, "tool_planner")
        if planner_raw:
            tool_plan = planner_raw.get("validated_plan") or {}

        record.update(
            {
                "elapsed_seconds": time.time() - start,
                "tool_names": [
                    item.get("tool_name")
                    for item in result.get("tool_usage") or []
                ],
                "tool_errors": [
                    item.get("error")
                    for item in result.get("tool_usage") or []
                    if item.get("error")
                ],
                "search_queries": search_raw.get("queries") or [],
                "evidence_count": len(search_raw.get("evidence_items") or []),
                "retrieval_stop_reason": retrieval.get("stop_reason", ""),
                "final_intent_state": states[-1] if states else "",
                "tool_plan": tool_plan,
                "search_result": result.get("search_result", ""),
                "attachment_result": result.get("attachment_result", ""),
                "solver_result": result.get("solver_result", ""),
                "tool_usage": result.get("tool_usage") or [],
            }
        )
        print(
            f"[OK] tools={record['tool_names']} evidence={record['evidence_count']} "
            f"stop={record['retrieval_stop_reason'] or '-'} "
            f"state={record['final_intent_state'] or '-'} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
    except Exception as exc:
        record["elapsed_seconds"] = time.time() - start
        record["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)

    return record


def first_tool_raw_result(result: dict[str, Any], tool_name: str) -> dict[str, Any]:
    for item in result.get("tool_usage") or []:
        if item.get("tool_name") != tool_name:
            continue
        raw = item.get("raw_result")
        return raw if isinstance(raw, dict) else {}
    return {}


def render_summary(records: list[dict[str, Any]], *, args: argparse.Namespace) -> str:
    stop_counts = Counter(
        record["retrieval_stop_reason"] or ("error" if record["error"] else "no_search")
        for record in records
    )
    tool_counts = Counter(
        tool
        for record in records
        for tool in record.get("tool_names", [])
        if tool
    )
    state_counts = Counter(
        record["final_intent_state"] or ("error" if record["error"] else "no_state")
        for record in records
    )
    total = len(records)
    average_evidence = sum(record["evidence_count"] for record in records) / max(1, total)
    average_time = sum(record["elapsed_seconds"] for record in records) / max(1, total)

    lines = [
        f"# {args.log_name}",
        "",
        f"- Split: {args.split}",
        f"- Level: {args.level}",
        f"- Total: {total}",
        f"- Errors: {sum(1 for record in records if record['error'])}",
        f"- Search triggered: {sum(1 for record in records if record['search_queries'])}",
        f"- Average evidence count: {average_evidence:.2f}",
        f"- Average evidence prepare time: {average_time:.2f}s",
        "",
        "## Tool Counts",
    ]
    lines.extend(f"- {key}: {value}" for key, value in tool_counts.most_common())
    if not tool_counts:
        lines.append("- none: 0")

    lines.append("")
    lines.append("## Stop Reasons")
    lines.extend(f"- {key}: {value}" for key, value in stop_counts.most_common())

    lines.append("")
    lines.append("## Intent States")
    lines.extend(f"- {key}: {value}" for key, value in state_counts.most_common())

    lines.extend(
        [
            "",
            "## Tasks",
            "",
            "| # | task_id | tools | evidence | stop | state | time |",
            "|---:|---|---|---:|---|---|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['index']} | {record['task_id']} | "
            f"{', '.join(record.get('tool_names') or [])} | "
            f"{record['evidence_count']} | {record['retrieval_stop_reason'] or '-'} | "
            f"{record['final_intent_state'] or '-'} | {record['elapsed_seconds']:.1f}s |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
