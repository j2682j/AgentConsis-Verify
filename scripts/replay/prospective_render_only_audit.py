"""What today's renderer does to a fixed evidence snapshot, measured exactly.

This is not a replay of final_22 or final_23 and cannot become one. Those runs
recorded only the post-budget length of each section, and the budget divides one
total allowance across sections whose pre-budget contents were never saved, so a
reconstruction cannot be checked -- 285 of 286 attempted rows disagreed with the
recorded lengths. Historical block-level replay is marked unsupported and stays
that way.

What can be measured exactly is the present. Take the evidence each task
produced, hand it to the current prompt builder for each real agent, and record
the raw, compressed and rendered search sections with their content hashes in
the same execution. Nothing is reconstructed, so nothing needs fidelity grading.

Three layers, because there are two reductions. A line compressor trims by lines
and characters, then the budget divides the total allowance, and a two-layer
record cannot say which one removed a block. Both are traced separately.

No agent is called, nothing is searched, no tool runs, no verifier scores and no
winner is chosen. The audit answers one question -- which evidence blocks reach
the prompt -- and deliberately cannot answer whether the ones removed mattered.
That needs the blind role annotation, and until it exists a dropped block is not
evidence of a delivery defect: dropping `MENTION_ONLY` text is the budget
working.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"c:/SCP")

from context.evidence_block_lineage import digest

RUN = "c:/SCP/outputs/level1_final_23/tasks"
OUT = "c:/SCP/outputs/prospective_render_audit"

#: Fixed before the audit runs, so the report cannot be sliced to taste. Every
#: layer is reported separately and the whole population is reported too.
STRATA = {
    "persistent_wrong_seven": ("002", "007", "018", "019", "044", "047", "053"),
    "document_to_context_loss": ("004", "020", "033"),
    "final_selection": ("029", "034", "040"),
    "f22_f23_churn": ("001", "005", "026"),
}

RUNS_PER_AGENT = 3


def load_tasks() -> list[dict]:
    out = []
    for path in sorted(glob.glob(f"{RUN}/*.json")):
        record = json.load(open(path, encoding="utf-8"))
        meta = (record.get("network_summary") or {}).get("metadata") or {}
        raw = {}
        for item in meta.get("tool_usage") or []:
            if item.get("tool_name") == "search" and isinstance(item.get("raw_result"), dict):
                raw = item["raw_result"]
                break
        if not raw:
            continue
        out.append({
            "task_id": os.path.basename(path)[:3],
            "question": str(record.get("question") or ""),
            "record": record,
            "search_raw": raw,
        })
    return out


def evidence_snapshot(entry: dict) -> dict:
    """Every section the budget divides its allowance across, not just search.

    Feeding the search evidence alone would measure a prompt nobody builds: the
    total allowance is shared, so what search keeps depends on what the other
    sections take.
    """

    meta = (entry["record"].get("network_summary") or {}).get("metadata") or {}
    sections = {"search_result": str(entry["search_raw"].get("summary") or "")}
    for item in meta.get("tool_usage") or []:
        name = str(item.get("tool_name") or "")
        text = str(item.get("output_text") or "")
        if not text:
            continue
        if name == "attachment_reader":
            sections["attachment_result"] = text
        elif name in ("deterministic_solver", "solver"):
            sections["solver_result"] = text
    for agent in (entry["record"].get("network_summary") or {}).get("stage1_results") or []:
        for run in agent.get("runs") or []:
            requirement = ((run.get("repair_metadata") or {}).get("answer_requirement"))
            if requirement:
                sections.setdefault("answer_requirement", str(requirement))
    return sections


def build_packets(snapshot: dict, question: str):
    from context.stage1_context import ContextPacket

    packets = [ContextPacket(packet_type="question", content=question)]
    for packet_type, priority in (
        ("answer_requirement", 95), ("solver_result", 90),
        ("attachment_result", 80), ("search_result", 70),
    ):
        content = snapshot.get(packet_type)
        if content:
            packets.append(
                ContextPacket(packet_type=packet_type, content=content, priority=priority)
            )
    return packets


def agent_configs() -> list[str]:
    """The three agents as the benchmark configures them."""

    return ["nemotron", "qwen", "gemma"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    from context.stage1_context import Stage1ContextBuilder

    tasks = load_tasks()
    rows = []
    for entry in tasks:
        snapshot = evidence_snapshot(entry)
        before = digest(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        for agent in agent_configs():
            for run_index in range(1, RUNS_PER_AGENT + 1):
                builder = Stage1ContextBuilder()
                compressed = builder.compress(
                    builder.structure(build_packets(snapshot, entry["question"]))
                )
                budget = compressed.get("_context_budget") or {}
                messages = builder.render(compressed)
                rows.append({
                    "task_id": entry["task_id"],
                    "agent_id": agent,
                    "run_index": run_index,
                    "evidence_snapshot_hash": before,
                    "raw_search_chars": budget.get("raw_search_context_chars"),
                    "raw_search_hash": budget.get("raw_search_context_hash"),
                    "compressed_search_chars": budget.get("prepared_search_context_chars"),
                    "compressed_search_hash": budget.get("prepared_search_context_hash"),
                    "rendered_search_chars": budget.get("rendered_search_context_chars"),
                    "rendered_search_hash": budget.get("rendered_search_context_hash"),
                    "raw_to_compressed": budget.get("compressed_from_raw") or {},
                    "compressed_to_rendered": budget.get("rendered_from_compressed") or {},
                    "section_chars": budget.get("section_chars") or {},
                    "total_prompt_chars": sum(len(m["content"]) for m in messages),
                })
        after = digest(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        if before != after:
            raise SystemExit(
                f"renderer 就地修改了 evidence snapshot（task {entry['task_id']}）"
            )

    with open(f"{OUT}/render_audit.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"prospective_render_only_audit")
    print(f"   {len(tasks)} task × {len(agent_configs())} agent × {RUNS_PER_AGENT} run"
          f" = {len(rows)} 筆 first-turn render")
    print(f"   evidence snapshot 執行前後 hash 一致，renderer 未就地修改輸入\n")

    def retention(rows_: list[dict], a: str, b: str) -> float:
        pairs = [(r[a], r[b]) for r in rows_ if r[a] and r[b]]
        return sum(y / x for x, y in pairs) / len(pairs) if pairs else 0.0

    print(f"=== 字元保留率")
    print(f"   raw → compressed        {retention(rows,'raw_search_chars','compressed_search_chars'):.3f}")
    print(f"   compressed → rendered   {retention(rows,'compressed_search_chars','rendered_search_chars'):.3f}")

    print(f"\n=== block 去向（compressed → rendered）")
    totals = Counter()
    for row in rows:
        stage = row["compressed_to_rendered"]
        for name in ("kept_count", "truncated_count", "dropped_count"):
            totals[name] += int(stage.get(name) or 0)
    total_blocks = sum(totals.values())
    for name in ("kept_count", "truncated_count", "dropped_count"):
        share = totals[name] / total_blocks if total_blocks else 0
        print(f"   {name:<16} {totals[name]:>6}  {share:.3f}")

    print(f"\n=== 三個 agent 是否得到相同 block 集合")
    per_task: dict[str, set] = defaultdict(set)
    for row in rows:
        per_task[row["task_id"]].add(
            (row["agent_id"], tuple(row["compressed_to_rendered"].get("lost_block_ids") or []))
        )
    agreement = Counter()
    for task, entries in per_task.items():
        lost_sets = {lost for _, lost in entries}
        any_drop = any(lost for lost in lost_sets)
        if not any_drop:
            agreement["no_agent_drops"] += 1
        elif len(lost_sets) == 1:
            agreement["all_agents_drop"] += 1
        else:
            agreement["some_agents_drop"] += 1
    for name in ("no_agent_drops", "all_agents_drop", "some_agents_drop"):
        print(f"   {name:<20} {agreement[name]:>3}/{len(per_task)}")

    print(f"\n=== 同 agent 三個 run 的 rendered hash 是否一致")
    inconsistent = []
    grouped: dict[tuple, set] = defaultdict(set)
    for row in rows:
        grouped[(row["task_id"], row["agent_id"])].add(row["rendered_search_hash"])
    for key, hashes in grouped.items():
        if len(hashes) > 1:
            inconsistent.append(key)
    print(f"   一致 {len(grouped) - len(inconsistent)}/{len(grouped)}"
          f"　不一致 {inconsistent[:5] if inconsistent else '無'}")

    print(f"\n=== 分層（事前固定）")
    for name, ids in STRATA.items():
        subset = [r for r in rows if r["task_id"] in ids]
        if not subset:
            continue
        dropped = sum(
            int((r["compressed_to_rendered"].get("dropped_count") or 0)) for r in subset
        )
        tasks_with_drop = len({
            r["task_id"] for r in subset
            if (r["compressed_to_rendered"].get("dropped_count") or 0)
        })
        print(f"   {name:<26} {len(set(ids))} task"
              f"　有 block 被丟的 {tasks_with_drop}"
              f"　累計丟棄 {dropped}")
    everything = len({r["task_id"] for r in rows})
    with_drop = len({
        r["task_id"] for r in rows
        if (r["compressed_to_rendered"].get("dropped_count") or 0)
    })
    print(f"   {'全部母體':<26} {everything} task　有 block 被丟的 {with_drop}")

    print(f"\n這輪只回答『目前哪些 block 會被丟』。被丟的是否重要，"
          f"要等 blind Evidence Block Role 標註；")
    print(f"若丟的多為 MENTION_ONLY／IRRELEVANT，代表 budget 正常運作，"
          f"不構成 delivery 缺陷。")


if __name__ == "__main__":
    sys.exit(main())
