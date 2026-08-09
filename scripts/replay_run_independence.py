"""How much do Stage 1 runs share evidence, and can its effect be measured here?

``Stage1SearchAccessState`` is one object per task. When a run's search
succeeds, ``complete()`` appends its output as a ContextPacket and every later
run of every agent starts with that text in context. So runs are not
independent samples, and ``supporting_run_count`` -- the only selector signal
above chance -- cannot be called self-consistency.

Quantifying *how much* that changes answers is a different question, and this
script exists partly to show it cannot be answered from a finished run.

Execution is agent-outer: ``unload_previous_slm_on_switch`` defaults to True and
``_run_sequential_with_model_switch_unload`` loops agents outside runs, so one
agent completes all its runs before the next begins. Splitting a task at its
first successful search therefore puts mostly one model on the before side and
different models on the after side, and the resulting difference measures model
identity, not sharing. Both orderings have been tried (+0.031 run-outer, -0.186
agent-outer) and neither means anything.

The comparison that would control for model identity -- split one agent's own
runs at its own first search -- is reported below and is empty in practice.

So this prints exposure and feasibility, not an effect size:

* how many runs began with propagated evidence available
* how many clean within-agent splits exist (4 across three runs, i.e. none)

To actually measure the effect, record per run: global execution index, the
supplemental-evidence ids present at run start, and the answer distribution
before and after each evidence revision. Then compare runs that did and did not
see a given packet, holding the agent fixed.

Usage:
    python scripts/replay_run_independence.py outputs/level1_final_13 \
        outputs/level1_final_15 outputs/level1_final_16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.gaia.answer_matcher import exact_match


def searched(run: dict) -> bool:
    """True when this run's search returned usable evidence into the shared state."""

    return any(
        str(result.get("tool_name")) == "search"
        and bool(result.get("evidence_valid"))
        and str(result.get("output_text") or "").strip()
        for result in run.get("tool_results") or []
    )


def agent_blocks(task: dict) -> list[tuple[str, list[dict]]]:
    """Runs grouped by agent, in execution order: agents outer, run_index inner."""

    blocks = []
    for agent in (task.get("network_summary") or {}).get("stage1_results") or []:
        runs = sorted(agent.get("runs") or [], key=lambda run: int(run.get("run_index") or 0))
        blocks.append((str(agent.get("agent_id") or ""), runs))
    return blocks


def iter_tasks(output_dirs: list[str]):
    for output_dir in output_dirs:
        for path in sorted(glob.glob(os.path.join(output_dir, "tasks", "*.json"))):
            with open(path, encoding="utf-8") as handle:
                yield os.path.basename(output_dir), os.path.basename(path)[:3], json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", help="GAIA report directories")
    args = parser.parse_args()

    exposed = unexposed = 0
    exposed_hit = unexposed_hit = 0
    first_search_at: Counter = Counter()
    searching_agents: Counter = Counter()
    agent_totals: Counter = Counter()
    clean_splits = 0

    for _run_name, _task_number, task in iter_tasks(args.output_dirs):
        gold = str(task.get("expected") or "")
        blocks = agent_blocks(task)
        shared_yet = False

        for agent_id, runs in blocks:
            agent_totals[agent_id] += 1
            own = [index for index, run in enumerate(runs) if searched(run)]
            if own:
                searching_agents[agent_id] += 1
                first_search_at[(agent_id, own[0] + 1)] += 1
                # A clean split needs a run before this agent's own first search
                # and a run after it; the searching run is neither.
                if own[0] > 0 and own[0] < len(runs) - 1:
                    clean_splits += 1

            for index, run in enumerate(runs):
                correct = exact_match(str(run.get("final_answer") or ""), gold)
                if shared_yet and not (own and index == own[0]):
                    exposed += 1
                    exposed_hit += correct
                elif not shared_yet:
                    unexposed += 1
                    unexposed_hit += correct
                if own and index == own[0]:
                    shared_yet = True

    print("exposure to propagated evidence, by run")
    print(f"  ran before any search succeeded : {unexposed:5d}   exact {unexposed_hit}")
    print(f"  ran with shared evidence present: {exposed:5d}   exact {exposed_hit}")
    print(
        "\n  These two groups are NOT comparable: execution is agent-outer, so the"
        "\n  unexposed group is mostly the first agent and the exposed group is"
        "\n  mostly the others. Any difference here is model identity."
    )

    print("\nwhere each agent first searches (run index within its own block)")
    for agent_id in sorted(agent_totals):
        counts = ", ".join(
            f"run{index}: {count}"
            for (aid, index), count in sorted(first_search_at.items())
            if aid == agent_id
        )
        print(
            f"  {agent_id:<10} searched in {searching_agents[agent_id]:>3}/{agent_totals[agent_id]}"
            f"   {counts or '-'}"
        )

    print(f"\nclean within-agent splits available: {clean_splits}")
    print(
        "  A within-agent split is the only comparison that holds the model fixed."
        f"\n  With {clean_splits} available, the effect of sharing is NOT measurable from"
        "\n  these runs. Add per-run instrumentation before attempting a number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
