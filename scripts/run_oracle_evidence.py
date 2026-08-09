"""Hand Stage 1 a passage that provably contains the answer, and see if it helps.

26 wrong tasks never produce the answer in any Stage 1 run, and two explanations
fit equally well: the pipeline never delivered the evidence, or the models cannot
use it. Every pipeline direction attempted here has assumed the first without
testing it.

The first version of this script tested it badly, and the failures are worth
keeping in view because they are easy to repeat:

* it left tools on, so a correct answer could have come from a fresh search
  rather than from the passage, and it saved no tool trace to tell
* it selected passages by substring, so a gold of `CUB` matched inside `Cuba`
  and a gold of `2` matched `Table 2`
* it accepted a lexical hit as evidence: task 007's passage matched the prose
  "the castle appears deserted" while the question asks for a script's first
  scene heading
* it never saved fetched passages, so two tasks could not be reproduced

Two conclusions died with those defects -- "half delivery, half model" and
"007 and 050 are the 4B ceiling". Both were about tasks whose passage did not
contain the answer.

So this now runs in two steps. `--build-manifest` proposes passages and writes
them to disk with the sentence that supports the answer; a human confirms each
entry by setting `confirmed: true`. The run step reads only confirmed entries,
with tools off, and records enough per run to attribute the result.

    python scripts/run_oracle_evidence.py --from-run outputs/level1_final_16 --build-manifest
    # inspect outputs/oracle/oracle_manifest.json, set confirmed: true where the
    # passage really does state the answer
    python scripts/run_oracle_evidence.py --manifest outputs/oracle/oracle_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from unicodedata import normalize as unicode_normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark.gaia.answer_matcher import exact_match
from benchmark.gaia.gaia_runner import DEFAULT_AGENT_SPECS
from core.config import AgentConfig
from core.sampling_seed import describe as describe_seed
from core.slm_agent import SLM_Agent
from core.stage1_runner import Stage1Runner
from tools.evidence.fact_extraction.fact_store import TaskFactStore

MAX_PASSAGE_CHARS = 3000
DEFAULT_MANIFEST = ROOT / "outputs" / "oracle" / "oracle_manifest.json"


# ----------------------------------------------------------------- matching


def normalize(text: Any) -> str:
    collapsed = re.sub(r"[^a-z0-9 ]+", " ", unicode_normalize("NFKC", str(text or "")).casefold())
    return " ".join(collapsed.split())


def token_match(text: str, gold: str) -> bool:
    """The gold as a whole token, not as a substring inside another word.

    Substring matching put `050` into the sample on a passage that never held
    its answer: `CUB` inside `Cuba`. Numeric golds are worse — `2` matches any
    page with a table. This is necessary but not sufficient, which is why the
    manifest still asks a human to look.
    """

    needle = normalize(gold)
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, normalize(text)) is not None


def supporting_sentence(text: str, gold: str) -> str:
    """The sentence a reviewer needs to read to accept or reject the passage."""

    position = text.casefold().find(gold.strip().casefold())
    if position < 0:
        return ""
    start = text.rfind(".", 0, position) + 1
    end = text.find(".", position)
    end = len(text) if end < 0 else end + 1
    return " ".join(text[start:end].split())[:400]


def window(text: str, gold: str) -> str:
    if len(text) <= MAX_PASSAGE_CHARS:
        return text
    position = text.casefold().find(gold.strip().casefold())
    if position < 0:
        return text[:MAX_PASSAGE_CHARS]
    start = max(0, position - MAX_PASSAGE_CHARS // 2)
    return text[start : start + MAX_PASSAGE_CHARS]


# ----------------------------------------------------------------- manifest


def recorded_passage(task: dict, gold: str) -> tuple[str, str]:
    for round_ in task.get("search_summary", {}).get("retrieval_rounds") or []:
        for document in round_.get("documents") or []:
            text = str(document.get("text") or "")
            if token_match(text, gold):
                url = str(document.get("canonical_url") or document.get("url") or "")
                return window(text, gold), url
    return "", ""


def fetched_passage(task: dict, gold: str, limit: int) -> tuple[str, str]:
    from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
        fetch_page_content_result,
    )

    candidates = (task.get("search_summary", {}).get("source_filter") or {}).get(
        "fetch_selection_candidates"
    ) or []
    ranked = sorted(candidates, key=lambda item: -int(item.get("query_hit_count") or 0))[:limit]
    for candidate in ranked:
        url = str(candidate.get("url") or "")
        if not url:
            continue
        try:
            result = fetch_page_content_result(url, max_tokens=12000)
        except Exception:
            continue
        text = str(getattr(result, "content", "") or "") if result else ""
        if text and token_match(text, gold):
            return window(text, gold), url
    return "", ""


def build_manifest(args: argparse.Namespace) -> int:
    run_dir = Path(args.from_run).resolve()
    entries = []
    for path in sorted(run_dir.glob("tasks/*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("exact_match"):
            continue
        gold = str(task.get("expected") or "")
        runs = [
            str(run.get("final_answer") or "")
            for agent in (task.get("network_summary") or {}).get("stage1_results") or []
            for run in agent.get("runs") or []
        ]
        if any(exact_match(answer, gold) for answer in runs):
            continue  # Stage 1 already produces it; that is a selection problem.

        passage, url = recorded_passage(task, gold)
        source = "recorded"
        if not passage and args.fetch_missing:
            passage, url = fetched_passage(task, gold, args.max_fetch_per_task)
            source = "fetched"
            print(f"  [{path.name[:3]}] fetch {gold[:22]!r}: {'found' if passage else 'none'}", flush=True)
        if not passage:
            continue
        entries.append(
            {
                "task": path.name[:3],
                "task_id": str(task.get("task_id") or ""),
                "question": " ".join(str(task.get("question") or "").split()),
                "gold": gold,
                "source": source,
                "url": url,
                "supporting_sentence": supporting_sentence(passage, gold),
                "passage": passage,
                # Deliberately false. An unconfirmed entry is a guess by string
                # match, and three of those produced conclusions that had to be
                # withdrawn.
                "confirmed": False,
                "review_note": "",
            }
        )

    destination = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] {len(entries)} candidate passages -> {destination}")
    print("Set `confirmed: true` only where the sentence really states the answer.")
    for entry in entries:
        print(f"\n  {entry['task']}  gold={entry['gold'][:34]!r}  ({entry['source']})")
        print(f"      {entry['supporting_sentence'][:150]!r}")
    return 0


# --------------------------------------------------------------------- run


def oracle_bundle(passage: str, prior_context: str = "") -> dict[str, Any]:
    """Passage first, then the recorded context.

    Appending it last put it at the front of the queue for the budget's blind
    cut: the search block is trimmed near 2100 characters, the recorded context
    alone is ~2254, and the passage vanished every time. The `in_context` arm
    then scored 0/9 on four tasks while measuring nothing but truncation.

    Ordering it first keeps the answer in the prompt, which turns the arm into a
    different and answerable question: with the answer present, does the rest of
    the production context degrade the model's ability to use it?
    """

    store = TaskFactStore()
    search_result = f"{passage}\n\n{prior_context}".strip() if prior_context else passage
    return {
        "search_result": search_result,
        "attachment_result": "",
        "attachment_profile": {},
        "solver_result": "",
        "answer_requirement": "",
        "answer_role": "",
        "answer_target": "",
        "routing": {
            "evidence_prepare_enabled": False,
            "use_search": True,
            "use_attachment": False,
            "use_deterministic_solver": False,
            "use_python_solver": False,
            "provided_search_result": True,
        },
        "tool_usage": [],
        "fact_store": store.to_dict(),
        "_fact_store": store,
    }


def production_context(task_id: str, run_dir: Path) -> str:
    """What Stage 1 actually read in the recorded run, rebuilt from its summary."""

    from replay_evidence_funnel import stage1_context

    for path in sorted(run_dir.glob("tasks/*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if str(task.get("task_id") or "") != task_id:
            continue
        summary = task.get("search_summary") or {}
        if not summary.get("retrieval_rounds"):
            return ""
        try:
            return stage1_context(summary)
        except Exception:
            return ""
    return ""


def run_stage1(question: str, evidence: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    cache: dict[str, SLM_Agent] = {}

    def get_agent(config: AgentConfig) -> SLM_Agent:
        agent = cache.get(config.agent_id)
        if agent is None:
            agent = SLM_Agent(model_name=config.model_name, temperature=config.temperature)
            cache[config.agent_id] = agent
        return agent

    runner = Stage1Runner(
        question=question,
        agents=[
            AgentConfig(agent_id=agent_id, model_name=model_name, temperature=0.5)
            for agent_id, model_name in DEFAULT_AGENT_SPECS
        ],
        get_agent=get_agent,
        record_token_usage=lambda **_: None,
        stage1_runs_per_agent=args.runs_per_agent,
        max_workers=1,
        # Off, so a correct answer cannot have come from a fresh search. This is
        # the whole point of the arm and was the first version's central defect.
        enable_tool_use=False,
        tool_manager=None,
    )
    return runner.run(evidence)


def run_detail(summaries: list[Any], gold: str) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for run in getattr(summary, "runs", None) or []:
            answer = str(getattr(run, "final_answer", "") or "")
            rows.append(
                {
                    "agent": str(getattr(summary, "agent_id", "") or ""),
                    "run_index": int(getattr(run, "run_index", 0) or 0),
                    "final_answer": answer,
                    "correct": bool(exact_match(answer, gold)),
                    "reasoning": str(getattr(run, "reasoning", "") or "")[:600],
                    "raw_reply": str(getattr(run, "raw_reply", "") or "")[:600],
                    "tool_calls": list(getattr(run, "tool_calls", None) or []),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-run", default="", help="Finished run, required to build a manifest.")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--max-fetch-per-task", type=int, default=4)
    parser.add_argument("--runs-per-agent", type=int, default=3)
    parser.add_argument(
        "--arms",
        default="oracle",
        help="Comma-separated: `oracle` (passage alone), `in_context` (recorded context + passage).",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Passes per task; the middle is noisy.")
    parser.add_argument("--log-name", default="oracle_evidence")
    args = parser.parse_args(argv)

    if args.build_manifest:
        if not args.from_run:
            raise SystemExit("--build-manifest needs --from-run")
        return build_manifest(args)

    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}; run --build-manifest first")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    confirmed = [entry for entry in entries if entry.get("confirmed")]
    skipped = len(entries) - len(confirmed)
    if not confirmed:
        raise SystemExit(
            f"{manifest_path} has no confirmed entries. Read each passage and set "
            "`confirmed: true` where it genuinely states the answer."
        )

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    run_dir = Path(args.from_run).resolve() if args.from_run else None
    output_dir = ROOT / "outputs" / args.log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] oracle: {len(confirmed)} confirmed ({skipped} unconfirmed, skipped), "
        f"arms={arms}, repeats={args.repeats}, tools=off, seed={describe_seed()}",
        flush=True,
    )

    results = []
    for index, entry in enumerate(confirmed, start=1):
        gold = str(entry.get("gold") or "")
        question = str(entry.get("question") or "")
        print(f"\n===== {index}/{len(confirmed)} {entry['task']} gold={gold[:34]!r}", flush=True)
        for arm in arms:
            prior = ""
            if arm == "in_context":
                if run_dir is None:
                    print("      in_context needs --from-run; skipped", flush=True)
                    continue
                prior = production_context(str(entry.get("task_id") or ""), run_dir)
            for repeat in range(1, args.repeats + 1):
                started = time.time()
                try:
                    summaries = run_stage1(
                        question, oracle_bundle(str(entry.get("passage") or ""), prior), args
                    )
                except Exception as exc:
                    print(f"      {arm} r{repeat}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                detail = run_detail(summaries, gold)
                correct = sum(1 for row in detail if row["correct"])
                results.append(
                    {
                        "task": entry["task"],
                        "gold": gold,
                        "arm": arm,
                        "repeat": repeat,
                        "correct_runs": correct,
                        "total_runs": len(detail),
                        "runs": detail,
                        "seconds": time.time() - started,
                    }
                )
                print(
                    f"      {arm:<10} r{repeat}: {correct}/{len(detail)}  "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'task':<6} {'arm':<11} correct across repeats")
    for entry in confirmed:
        for arm in arms:
            scores = [
                f"{row['correct_runs']}/{row['total_runs']}"
                for row in results
                if row["task"] == entry["task"] and row["arm"] == arm
            ]
            if scores:
                print(f"{entry['task']:<6} {arm:<11} {'  '.join(scores)}")
    print(f"\n[OK] {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
