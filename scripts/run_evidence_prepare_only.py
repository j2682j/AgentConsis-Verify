"""Run Evidence Prepare over a GAIA split without Stage 1 or Stage 2.

Retrieval decides what the Agents ever get to read, and on recent runs it is
where the losses concentrate: level1_final_13 finished 28 retrieval tasks with
one grounded evidence item, and 27 of them recorded
`evidence_conversion_empty:goal_incomplete_*`. Measuring that through a full
run costs five hours, of which Stage 1 is 54% and Stage 2 3%, and buries the
signal under Stage 1 sampling.

This runs only the Evidence Prepare stage, so a 53-task sweep takes roughly two
hours and the result is a property of retrieval alone. The per-task JSON keeps
the same `search_summary` shape the GAIA runner writes, so the analysis scripts
that read `outputs/level1_*` work unchanged.

    .\\venv312\\Scripts\\python.exe scripts/run_evidence_prepare_only.py \\
      --level 1 --max-samples 53 --bypass-search-labeler --log-name evidence_only_01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.gaia.dataset import GAIADataset
from benchmark.gaia.gaia_runner import build_attachment, extract_search_summary
from core.evidence_runner import EvidenceRunner
from tools.attachment_workspace import AttachmentWorkspace
from tools.tool_manager import ToolManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "gaia"))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated GAIA task ids (prefixes allowed). Overrides --max-samples.",
    )
    parser.add_argument("--bypass-search-labeler", action="store_true")
    parser.add_argument(
        "--enable-evidence-driven-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compact-search-evidence", action="store_true")
    parser.add_argument("--max-parallel-next-hop-queries", type=int, default=2)
    parser.add_argument("--log-name", default="evidence_only")
    return parser.parse_args(argv)


def select_items(dataset: GAIADataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    items = dataset.load()
    if args.task_ids:
        wanted = [value.strip() for value in args.task_ids.split(",") if value.strip()]
        items = [
            item
            for item in items
            if any(str(item.get("task_id", "")).startswith(prefix) for prefix in wanted)
        ]
        missing = [
            prefix
            for prefix in wanted
            if not any(str(item.get("task_id", "")).startswith(prefix) for item in items)
        ]
        if missing:
            raise SystemExit(f"--task-ids matched nothing for: {', '.join(missing)}")
        return items
    if args.max_samples is not None:
        return items[: max(0, args.max_samples)]
    return items


def prepare_one(sample: dict[str, Any], args: argparse.Namespace, tool_manager: ToolManager) -> dict[str, Any]:
    attachment = build_attachment(sample)
    started = time.time()
    bundle: dict[str, Any] = {}
    error = ""
    try:
        bundle = EvidenceRunner(
            question=str(sample.get("question") or ""),
            attachment=attachment,
            tool_manager=tool_manager,
            compact_search_evidence=args.compact_search_evidence,
            enable_evidence_driven_search=args.enable_evidence_driven_search,
            bypass_search_labeler=args.bypass_search_labeler,
            max_parallel_next_hop_queries=args.max_parallel_next_hop_queries,
            attachment_workspace=AttachmentWorkspace(attachment),
        ).run()
    except Exception as exc:  # a bad task must not end the sweep
        error = f"{type(exc).__name__}: {exc}"

    # extract_search_summary reads the shape the GAIA runner hands it, so the
    # output stays readable by every script that already parses those folders.
    network_summary = {"metadata": {"tool_usage": bundle.get("tool_usage") or []}}
    return {
        "task_id": sample.get("task_id", ""),
        "level": sample.get("level", ""),
        "question": sample.get("question", ""),
        "attachment": attachment,
        "expected": str(sample.get("final_answer", "") or ""),
        "error": error,
        "execution_time": time.time() - started,
        "search_summary": extract_search_summary(network_summary),
        "evidence_prepare": {
            # Recorded because `tool_usage` is stripped to four keys below, which
            # drops the `raw_result` the statuses used to be readable from. A
            # replay that cannot tell `not_run` from `complete + empty`, or
            # `partial_failure` from `failed`, cannot attribute anything.
            "pipeline_status": str(bundle.get("pipeline_status") or "not_run"),
            "evidence_status": str(bundle.get("evidence_status") or "not_applicable"),
            "search_result_chars": len(str(bundle.get("search_result") or "")),
            "attachment_result_chars": len(str(bundle.get("attachment_result") or "")),
            "solver_result_chars": len(str(bundle.get("solver_result") or "")),
            "answer_requirement": str(bundle.get("answer_requirement") or ""),
            "routing": bundle.get("routing") or {},
            "evidence_readiness": bundle.get("evidence_readiness") or {},
            "tool_usage": [
                {
                    key: value
                    for key, value in (usage or {}).items()
                    if key in {"tool_name", "ok", "status", "error"}
                }
                for usage in bundle.get("tool_usage") or []
            ],
        },
    }


def write_report(results: list[dict[str, Any]], output_dir: Path, log_name: str) -> Path:
    counts = {
        "tasks": len(results),
        "with_retrieval": sum(1 for r in results if (r["search_summary"] or {}).get("retrieval_rounds")),
        "errors": sum(1 for r in results if r["error"]),
        "evidence_count": sum((r["search_summary"] or {}).get("evidence_count") or 0 for r in results),
        "strict_evidence_count": sum(
            (r["search_summary"] or {}).get("strict_evidence_count") or 0 for r in results
        ),
        "unverified_reference_count": sum(
            (r["search_summary"] or {}).get("unverified_reference_count") or 0 for r in results
        ),
        "seconds": sum(r["execution_time"] for r in results),
    }
    lines = [
        f"# Evidence Prepare Only — {log_name}",
        "",
        f"- Tasks: {counts['tasks']}",
        f"- With retrieval: {counts['with_retrieval']}",
        f"- Errors: {counts['errors']}",
        f"- Grounded evidence items: {counts['evidence_count']}",
        f"- Strict evidence items: {counts['strict_evidence_count']}",
        f"- Unverified references: {counts['unverified_reference_count']}",
        f"- Total seconds: {counts['seconds']:.0f}",
        "",
        "| # | task | evidence | strict | refs | failure stage |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, result in enumerate(results, start=1):
        summary = result["search_summary"] or {}
        lines.append(
            "| {i} | {t} | {e} | {s} | {r} | {f} |".format(
                i=index,
                t=str(result["task_id"])[:8],
                e=summary.get("evidence_count") or 0,
                s=summary.get("strict_evidence_count") or 0,
                r=summary.get("unverified_reference_count") or 0,
                f=result["error"] or summary.get("pipeline_failure_stage") or "-",
            )
        )
    path = output_dir / f"{log_name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = GAIADataset(split=args.split, level=args.level, local_data_dir=Path(args.data_dir).resolve())
    items = select_items(dataset, args)
    output_dir = ROOT / "outputs" / args.log_name
    (output_dir / "tasks").mkdir(parents=True, exist_ok=True)
    tool_manager = ToolManager()

    print(f"[INFO] evidence-prepare only: {len(items)} tasks, level={args.level}")
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(items, start=1):
        print(f"\n===== {index}/{len(items)} task_id={sample.get('task_id')} =====", flush=True)
        result = prepare_one(sample, args, tool_manager)
        results.append(result)
        path = output_dir / "tasks" / f"{index:03d}_{result['task_id']}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        summary = result["search_summary"] or {}
        print(
            "evidence={e} strict={s} refs={r} {sec:.0f}s {f}".format(
                e=summary.get("evidence_count") or 0,
                s=summary.get("strict_evidence_count") or 0,
                r=summary.get("unverified_reference_count") or 0,
                sec=result["execution_time"],
                f=result["error"] or summary.get("pipeline_failure_stage") or "",
            ),
            flush=True,
        )

    report = write_report(results, output_dir, args.log_name)
    print(f"\n[OK] {report}")


if __name__ == "__main__":
    main()
