"""Paired A/B over the Stage 1 search budget, on the tasks where it binds.

Stage 1's own search tool finds the gold answer in 14 tasks against the prepared
pipeline's 4 and converts at 64%, and every genuinely denied search call in a
recorded run is `refinement_budget_exhausted` -- 79 per run, concentrated on 23
of 53 tasks. This asks whether raising that budget delivers more answers.

It cannot be asked with two ordinary benchmark runs. Retrieval returns a
different corpus every time (median Jaccard 0.33 across final_13/15/16), so two
runs differ by churn as much as by budget. Everything here exists to remove that:

* Evidence Prepare runs **once per task**; both arms start from the same bundle.
* Network responses are paired. The first arm to issue a query executes it, and
  any later arm asking the same question gets the byte-identical payload, so a
  shared query cannot differ between arms. Queries only one arm issues are the
  treatment effect and stay live.
* Arms share nothing mutable. Stage 1 tool use writes into the bundle's
  `_fact_store`, so each arm gets its own via `TaskFactStore.from_dict`, plus its
  own `Stage1SearchAccessState` and `ToolCache` (fresh with each Stage1Runner).
* Arm order alternates by task, so neither arm is always the one that pays for a
  fresh network call.

The pairing is deliberately *not* done by sharing one ToolCache. That cache
rewrites a repeat hit into `status="already_available"`, so the second arm would
see a different status than the first for the same query. Pairing sits under the
cache, at the tool manager, and each arm keeps a production-shaped cache of its
own.

Stage 2 and winner selection are not run: the decision metric is whether gold
reaches a Stage 1 tool result, and the selector is known to be capped anyway.
Per-run and per-agent correctness are reported as a proxy for the standing
constraint that a correct task must not become wrong.

    .\\venv312\\Scripts\\python.exe scripts/run_search_budget_ablation.py \\
      --from-run outputs/level1_final_16 --arms 2,5 --log-name budget_ab_01
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any
from unicodedata import normalize as unicode_normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.gaia.answer_matcher import exact_match
from benchmark.gaia.dataset import GAIADataset
from benchmark.gaia.gaia_runner import DEFAULT_AGENT_SPECS, build_attachment
from context.context_builder import ContextPacket
from core.config import AgentConfig
from core.evidence_runner import EvidenceRunner
from core.sampling_seed import describe as describe_seed
from core.slm_agent import SLM_Agent
from core.stage1_runner import Stage1Runner
from core.stage1_search_gate import Stage1SearchAccessState
from tools.attachment_workspace import AttachmentWorkspace
from tools.evidence.fact_extraction.fact_store import TaskFactStore
from tools.tool_manager import ToolManager

GOLD_MIN_CHARS = 3
GOLD_MAX_CHARS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "gaia"))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument(
        "--from-run",
        required=True,
        help="Finished run directory used to derive the treatment and control sets.",
    )
    parser.add_argument("--arms", default="2,5", help="Comma-separated refinement budgets.")
    parser.add_argument("--task-ids", default=None, help="Override the derived task set.")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--runs-per-agent", type=int, default=3)
    parser.add_argument("--max-tool-turns", type=int, default=5)
    parser.add_argument("--supplemental-max-items", type=int, default=3)
    parser.add_argument("--bypass-search-labeler", action="store_true", default=True)
    # Defaults below mirror what level1_final_13/15/16 recorded, so Evidence
    # Prepare here produces the same shape those measurements came from.
    parser.add_argument(
        "--enable-evidence-driven-search", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compact-search-evidence", action="store_true", default=False)
    parser.add_argument("--max-parallel-next-hop-queries", type=int, default=2)
    parser.add_argument("--log-name", default="budget_ablation")
    return parser.parse_args(argv)


def parse_arms(spec: str) -> list[tuple[int, str, str]]:
    """`2,5,5:replace` -> budgets with an admission policy each.

    A bare number keeps production's first-come admission, so the existing
    `--arms 2,5` form still means what it did.
    """

    arms: list[tuple[int, str, str]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        budget_text, _, admission = raw.partition(":")
        admission = (admission.strip() or "fifo").casefold()
        if admission not in {"fifo", "replace"}:
            raise SystemExit(f"unknown admission policy {admission!r}; use fifo or replace")
        budget = int(budget_text)
        arms.append((budget, admission, f"{budget}:{admission}"))
    if len({label for _b, _a, label in arms}) != len(arms):
        raise SystemExit("--arms has duplicate budget:admission pairs")
    return arms


# ---------------------------------------------------------------- task sets


def derive_task_sets(run_dir: Path) -> tuple[list[str], list[str]]:
    """Treatment = tasks that hit the budget wall. Control = tasks that did not.

    Controls are additionally attachment-free and currently correct, so a
    regression among them is unambiguous: those tasks have nothing to gain from
    more search and everything to lose from worse evidence.
    """

    treatment: list[str] = []
    control: list[str] = []
    for path in sorted(run_dir.glob("tasks/*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        metadata = (task.get("network_summary") or {}).get("metadata") or {}
        gate = metadata.get("stage1_search_gate") or {}
        blocked = int((gate.get("blocked_reasons") or {}).get("refinement_budget_exhausted") or 0)
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        if blocked > 0:
            treatment.append(task_id)
        elif not str(task.get("attachment") or "").strip() and task.get("exact_match"):
            control.append(task_id)
    return treatment, control


# ------------------------------------------------------------ paired tools


class PairedToolManager:
    """Give every arm the identical payload for an identical tool call.

    The first arm to make a call executes it for real; later arms replay the
    stored result verbatim. Only calls no earlier arm made reach the network,
    which is exactly the difference a budget change is supposed to produce.
    """

    def __init__(self, inner: ToolManager) -> None:
        self._inner = inner
        self._responses: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self.executed = 0
        self.replayed = 0

    def reset(self) -> None:
        """Pairing is per task; a query means nothing across different tasks."""

        with self._lock:
            self._responses.clear()

    @staticmethod
    def _key(tool_name: str, tool_args: dict[str, Any]) -> str:
        def canonical(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k).strip(): canonical(v) for k, v in sorted(value.items())}
            if isinstance(value, (list, tuple)):
                return [canonical(item) for item in value]
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip().casefold()
            return value

        return json.dumps(
            {"tool": str(tool_name or "").strip().casefold(), "args": canonical(tool_args or {})},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        key = self._key(tool_name, tool_args)
        with self._lock:
            stored = self._responses.get(key)
        if stored is not None:
            with self._lock:
                self.replayed += 1
            return copy.deepcopy(stored)

        result = dict(self._inner.execute_tool(tool_name, tool_args, **kwargs))
        with self._lock:
            # Failures are not stored: a transient error must not be pinned onto
            # the other arm as though it were a property of the query.
            if not bool(result.get("retryable")):
                self._responses[key] = copy.deepcopy(result)
            self.executed += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ----------------------------------------------------------------- scoring


def normalize(text: Any) -> str:
    collapsed = re.sub(r"[^a-z0-9 ]+", " ", unicode_normalize("NFKC", str(text or "")).casefold())
    return " ".join(collapsed.split())


def gold_comparable(gold: str) -> bool:
    return GOLD_MIN_CHARS <= len(gold.strip()) <= GOLD_MAX_CHARS


def contains_gold(text: Any, gold: str) -> bool:
    needle = normalize(gold)
    return bool(needle) and needle in normalize(text)


def search_outputs(summaries: list[Any]) -> list[str]:
    """Every successful Stage 1 search payload, in execution order."""

    outputs: list[str] = []
    for summary in summaries:
        for run in getattr(summary, "runs", None) or []:
            for result in getattr(run, "tool_results", None) or []:
                if str(result.get("tool_name")) != "search":
                    continue
                if not result.get("evidence_valid"):
                    continue
                text = str(result.get("output_text") or "").strip()
                if text:
                    outputs.append(text)
    return outputs


class ReplacementPolicy:
    """Keep the best `cap` payloads instead of the first `cap` that arrive.

    Production admits a packet only while the set is under the cap and never
    reconsiders, so a payload that arrives late is locked out however good it is
    -- measured on `4b6bb5f7`, where raising the budget pushed the gold-bearing
    result from slot 1 to slot 4 -- and a repeated query served from ToolCache
    still consumes a slot with a copy of a page already held.

    This holds the cap fixed and changes only admission, so any difference is
    the policy rather than a bigger context. Two things change:

    * duplicates are refused, by query and by payload text -- not a hypothesis,
      admitting the same page twice is never useful;
    * a full set is reconsidered, and the weakest member is evicted if the new
      payload covers its own declared gap better.

    Ranking is on the `missing_information` the agent gave when it asked for the
    search: its own statement of what it lacked, not a property of the text.
    Every text-property signal tried for reference ordering failed, so this one
    is different in kind -- which is a reason to test it, not to expect it to
    work. `contribution` records which of the two changes moved each slot.
    """

    def __init__(self, state: Any, cap: int) -> None:
        self._state = state
        self._cap = max(1, int(cap))
        self._gaps: dict[str, str] = {}
        self._candidates: list[dict[str, Any]] = []
        self._seen_text: set[str] = set()
        self._kept_keys: set[str] = set()
        self.refused_duplicates = 0
        self.evictions = 0

    @staticmethod
    def coverage(text: str, gap: str) -> float:
        """Share of the declared gap's tokens that appear in the payload."""

        wanted = set(normalize(gap).split())
        if not wanted:
            return 0.0
        return len(wanted & set(normalize(text).split())) / len(wanted)

    def install(self) -> "ReplacementPolicy":
        state = self._state
        original_authorize = state.authorize
        original_complete = state.complete

        def authorize(*, query: str, missing_information: str, agent_id: str = "") -> Any:
            key = state.normalize_query(query)
            if key and str(missing_information or "").strip():
                self._gaps.setdefault(key, str(missing_information).strip())
            return original_authorize(
                query=query, missing_information=missing_information, agent_id=agent_id
            )

        def complete(*, query: str, result: dict[str, Any]) -> None:
            # Let production do its own bookkeeping first -- reserved keys,
            # execution counts, the cached-result store -- then override only
            # which packets survive.
            original_complete(query=query, result=result)
            self._admit(query, result)

        state.authorize = authorize
        state.complete = complete
        return self

    def _admit(self, query: str, result: dict[str, Any]) -> None:
        stored = dict(result or {})
        text = str(stored.get("output_text") or "").strip()
        key = self._state.normalize_query(query)
        fingerprint = normalize(text)
        if stored.get("evidence_valid") and text:
            if any(item["key"] == key for item in self._candidates) or fingerprint in self._seen_text:
                self.refused_duplicates += 1
            else:
                self._seen_text.add(fingerprint)
                self._candidates.append(
                    {
                        "key": key,
                        "text": text,
                        "score": self.coverage(text, self._gaps.get(key, "")),
                    }
                )
        # Rebuild on every call, not only on admission: production's own
        # `complete()` has already appended by now, so a refused duplicate would
        # otherwise stay in the packet list it just landed in.
        self._rebuild()

    def _rebuild(self) -> None:
        ranked = sorted(
            self._candidates,
            key=lambda item: (-item["score"], self._candidates.index(item)),
        )
        keep = ranked[: self._cap]
        kept_keys = {item["key"] for item in keep}
        if self._kept_keys and self._kept_keys - kept_keys:
            self.evictions += 1
        self._kept_keys = kept_keys
        with self._state._lock:
            self._state._supplemental_packets = [
                ContextPacket(
                    packet_type="search_result",
                    content=item["text"],
                    priority=75,
                    metadata={"source": "stage1_tool_use", "query": item["key"]},
                )
                for item in keep
            ]

    def snapshot(self) -> dict[str, int]:
        return {
            "refused_duplicates": self.refused_duplicates,
            "evictions": self.evictions,
            "candidates_seen": len(self._candidates),
        }


def missing_information_by_query(summaries: list[Any]) -> dict[str, str]:
    """The gap each search claimed to fill, keyed by normalized query.

    `complete()` does not store it on the packet, but the gate refuses a search
    without it, so every admitted packet has one recorded on its tool call. It
    is the one ranking signal the reference-ordering work never had: an agent's
    own statement of what it was missing, rather than a property of the text.
    """

    found: dict[str, str] = {}
    for summary in summaries:
        for run in getattr(summary, "runs", None) or []:
            for call in getattr(run, "tool_calls", None) or []:
                if str((call or {}).get("tool_name")) != "search":
                    continue
                args = (call or {}).get("tool_args") or {}
                query = normalize(args.get("input") or args.get("query"))
                gap = str((call or {}).get("missing_information") or "").strip()
                if query and gap and query not in found:
                    found[query] = gap
    return found


def admitted_packets(state: Any, summaries: list[Any], gold: str) -> list[dict[str, Any]]:
    """The packets later runs actually inherit, read from the gate itself.

    Taking the first N valid tool results instead is only a proxy: a repeated
    query served from ToolCache still reaches `complete()` with its payload
    intact and occupies a slot, so the admitted set can hold the same text
    twice. Reading the real list is what distinguishes "the cap is full" from
    "the cap is full of duplicates".
    """

    gaps = missing_information_by_query(summaries)
    packets = []
    for slot, packet in enumerate(state.supplemental_packets() if state else []):
        content = str(getattr(packet, "content", "") or "")
        query = normalize((getattr(packet, "metadata", None) or {}).get("query"))
        packets.append(
            {
                "slot": slot,
                "query": query,
                "missing_information": gaps.get(query, ""),
                "chars": len(content),
                "contains_gold": contains_gold(content, gold),
            }
        )
    return packets


def search_queries(summaries: list[Any]) -> list[str]:
    """Every search query the agents issued, normalized, in order.

    The query lives under `tool_args.input`; `query` is accepted only as a
    fallback. Reading the wrong key silently yields zero queries rather than an
    error, which is the difference between "the budget bought new searches" and
    "it re-ran the same ones" going unmeasured.
    """

    queries: list[str] = []
    for summary in summaries:
        for run in getattr(summary, "runs", None) or []:
            for call in getattr(run, "tool_calls", None) or []:
                if str((call or {}).get("tool_name")) != "search":
                    continue
                args = (call or {}).get("tool_args") or {}
                query = normalize(args.get("input") or args.get("query"))
                if query:
                    queries.append(query)
    return queries


def score_arm(summaries: list[Any], gold: str, supplemental_max: int) -> dict[str, Any]:
    outputs = search_outputs(summaries)
    queries = search_queries(summaries)
    run_answers = [
        str(getattr(run, "final_answer", "") or "")
        for summary in summaries
        for run in getattr(summary, "runs", None) or []
    ]
    agent_answers = [str(getattr(summary, "compressed_answer", "") or "") for summary in summaries]
    return {
        "search_results": len(outputs),
        "gold_in_tool_result": any(contains_gold(text, gold) for text in outputs),
        # complete() stores the first N valid payloads as supplemental packets,
        # so this is what later runs actually inherit.
        "gold_in_supplemental": any(
            contains_gold(text, gold) for text in outputs[:supplemental_max]
        ),
        # Where the gold-bearing payload sits decides whether a bigger cap or a
        # ranking rule would propagate it. Storing only the booleans above made
        # that unanswerable offline once, and cost a re-run to notice.
        "gold_output_index": next(
            (index for index, text in enumerate(outputs) if contains_gold(text, gold)), -1
        ),
        "output_chars": [len(text) for text in outputs],
        "unique_queries": len(set(queries)),
        "total_queries": len(queries),
        # Kept raw so a metric defined wrongly can be recomputed without
        # re-running four hours of search.
        "queries": queries,
        "runs_exact": sum(exact_match(answer, gold) for answer in run_answers),
        "runs_total": len(run_answers),
        "agents_exact": sum(exact_match(answer, gold) for answer in agent_answers),
        "agent_answers": agent_answers,
    }


# ------------------------------------------------------------------- arms


def run_arm(
    *,
    budget: int,
    admission: str,
    question: str,
    attachment: dict[str, Any],
    base_bundle: dict[str, Any],
    base_fact_dict: dict[str, Any],
    agents: list[AgentConfig],
    agent_cache: dict[str, SLM_Agent],
    tool_manager: PairedToolManager,
    args: argparse.Namespace,
) -> tuple[list[Any], dict[str, Any], dict[str, int]]:
    """One arm, on its own copy of everything Stage 1 is able to mutate."""

    evidence = copy.deepcopy({k: v for k, v in base_bundle.items() if k != "_fact_store"})
    evidence["_fact_store"] = TaskFactStore.from_dict(base_fact_dict)
    evidence["fact_store"] = evidence["_fact_store"].to_dict()

    tokens = {"prompt": 0, "completion": 0}

    def record_token_usage(*, stage: str, prompt_tokens: int, completion_tokens: int) -> None:
        tokens["prompt"] += int(prompt_tokens or 0)
        tokens["completion"] += int(completion_tokens or 0)

    def get_agent(config: AgentConfig) -> SLM_Agent:
        agent = agent_cache.get(config.agent_id)
        if agent is None:
            agent = SLM_Agent(model_name=config.model_name, temperature=config.temperature)
            agent_cache[config.agent_id] = agent
        return agent

    runner = Stage1Runner(
        question=question,
        agents=agents,
        get_agent=get_agent,
        record_token_usage=record_token_usage,
        attachment=attachment,
        stage1_runs_per_agent=args.runs_per_agent,
        max_workers=1,
        enable_tool_use=True,
        max_tool_turns=args.max_tool_turns,
        tool_manager=tool_manager,
        prepared_search_refinement_budget=budget,
        supplemental_search_evidence_max_items=args.supplemental_max_items,
        attachment_workspace=AttachmentWorkspace(attachment),
    )

    policy = None
    if admission == "replace":
        # Build the state here so the policy can wrap it; run() only creates one
        # when the runner does not already hold it.
        runner.search_access_state = Stage1SearchAccessState.from_evidence(
            evidence,
            refinement_budget=budget,
            supplemental_evidence_max_items=args.supplemental_max_items,
            per_agent_refinement_floor=runner.prepared_search_per_agent_floor,
        )
        policy = ReplacementPolicy(
            runner.search_access_state, args.supplemental_max_items
        ).install()

    before = (tool_manager.executed, tool_manager.replayed)
    summaries = runner.run(evidence)
    backend = {
        "executed": tool_manager.executed - before[0],
        "replayed": tool_manager.replayed - before[1],
    }
    return (
        summaries,
        runner.search_gate_metadata(),
        tokens,
        runner.search_access_state,
        backend,
        policy.snapshot() if policy else {},
    )


def run_task(
    sample: dict[str, Any],
    *,
    budgets: list[int],
    reverse: bool,
    role: str,
    args: argparse.Namespace,
    tool_manager: PairedToolManager,
    agents: list[AgentConfig],
) -> dict[str, Any]:
    question = str(sample.get("question") or "")
    gold = str(sample.get("final_answer") or "")
    attachment = build_attachment(sample)

    tool_manager.reset()
    agent_cache: dict[str, SLM_Agent] = {}

    prepare_started = time.time()
    base_bundle = EvidenceRunner(
        question=question,
        attachment=attachment,
        tool_manager=tool_manager,
        compact_search_evidence=args.compact_search_evidence,
        enable_evidence_driven_search=args.enable_evidence_driven_search,
        bypass_search_labeler=args.bypass_search_labeler,
        max_parallel_next_hop_queries=args.max_parallel_next_hop_queries,
        attachment_workspace=AttachmentWorkspace(attachment),
    ).run()
    prepare_seconds = time.time() - prepare_started
    base_fact_dict = copy.deepcopy(base_bundle.get("fact_store") or {})

    order = list(reversed(budgets)) if reverse else list(budgets)
    arms: dict[str, Any] = {}
    for budget, admission, label in order:
        started = time.time()
        summaries, gate, tokens, state, backend, policy = run_arm(
            budget=budget,
            admission=admission,
            question=question,
            attachment=attachment,
            base_bundle=base_bundle,
            base_fact_dict=base_fact_dict,
            agents=agents,
            agent_cache=agent_cache,
            tool_manager=tool_manager,
            args=args,
        )
        packets = admitted_packets(state, summaries, gold)
        arms[label] = {
            "budget": budget,
            "admission": admission,
            "policy": policy,
            "seconds": time.time() - started,
            "tokens": tokens["prompt"] + tokens["completion"],
            "logical_requests": int(gate.get("request_count") or 0),
            # Counts searches the gate let through, including ones this arm
            # replayed from the paired store. Real network calls are `backend`.
            "authorized_executions": int(gate.get("physical_execution_count") or 0),
            "backend_calls": backend,
            "blocked": int(gate.get("blocked_count") or 0),
            "blocked_reasons": gate.get("blocked_reasons") or {},
            "admitted_packets": packets,
            "packets_admitted": len(packets),
            "packets_distinct_queries": len({p["query"] for p in packets if p["query"]}),
            "gold_in_admitted_packets": any(p["contains_gold"] for p in packets),
            **score_arm(summaries, gold, args.supplemental_max_items),
        }

    return {
        "task_id": str(sample.get("task_id") or ""),
        "role": role,
        "question": question,
        "expected": gold,
        "gold_comparable": gold_comparable(gold),
        "arm_order": [label for _budget, _admission, label in order],
        "prepare_seconds": prepare_seconds,
        "paired_calls": {"executed": tool_manager.executed, "replayed": tool_manager.replayed},
        "arms": arms,
    }


# ----------------------------------------------------------------- report


def write_report(
    results: list[dict[str, Any]], budgets: list[tuple[int, str, str]], path: Path, log_name: str
) -> None:
    comparable = [r for r in results if r["gold_comparable"] and not r.get("error")]
    lines = [
        f"# Stage 1 search budget ablation — {log_name}",
        "",
        f"- Tasks: {len(results)} ({sum(1 for r in results if r['role'] == 'treatment')} treatment, "
        f"{sum(1 for r in results if r['role'] == 'control')} control)",
        f"- Comparable gold: {len(comparable)}",
        f"- Seed: {describe_seed()}",
        "",
        "## Decision metric — gold reaching a Stage 1 tool result",
        "",
        "| arm | gold in tool result | gold in admitted packets | authorized | backend | queries (unique/total) | packets (slots/distinct q) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _budget, _admission, key in budgets:
        rows = [r["arms"][key] for r in comparable if key in r["arms"]]
        lines.append(
            "| {b} | {g}/{n} | {a}/{n} | {p} | {e} | {u}/{t} | {sl}/{dq} |".format(
                b=key,
                n=len(rows),
                g=sum(1 for row in rows if row["gold_in_tool_result"]),
                a=sum(1 for row in rows if row.get("gold_in_admitted_packets")),
                p=sum(row.get("authorized_executions", 0) for row in rows),
                e=sum((row.get("backend_calls") or {}).get("executed", 0) for row in rows),
                u=sum(row["unique_queries"] for row in rows),
                t=sum(row.get("total_queries", 0) for row in rows),
                sl=sum(row.get("packets_admitted", 0) for row in rows),
                dq=sum(row.get("packets_distinct_queries", 0) for row in rows),
            )
        )
    lines += [
        "",
        "Unique rising with total is new search; total rising alone is the same",
        "queries re-issued, which buys nothing. Slots above distinct queries means",
        "the cap is partly filled with the same payload admitted twice.",
        "`authorized` counts what the gate allowed, including paired replays;",
        "`backend` is real network calls this arm paid for.",
    ]

    lines += ["", "## Correctness proxy (Stage 1 only, no selector)", "",
              "| arm | runs exact | agents exact | tokens | seconds |", "| --- | ---: | ---: | ---: | ---: |"]
    for _budget, _admission, key in budgets:
        rows = [r["arms"][key] for r in comparable if key in r["arms"]]
        lines.append(
            "| {b} | {re}/{rt} | {ae} | {tk} | {sec:.0f} |".format(
                b=key,
                re=sum(row["runs_exact"] for row in rows),
                rt=sum(row["runs_total"] for row in rows),
                ae=sum(row["agents_exact"] for row in rows),
                tk=sum(row["tokens"] for row in rows),
                sec=sum(row["seconds"] for row in rows),
            )
        )

    def contrast(low: str, high: str, title: str, note: str) -> list[str]:
        pairs = [r for r in comparable if low in r["arms"] and high in r["arms"]]

        def moved(field: str) -> tuple[list, list]:
            up = [r for r in pairs if r["arms"][high][field] and not r["arms"][low][field]]
            down = [r for r in pairs if r["arms"][low][field] and not r["arms"][high][field]]
            return up, down

        fetched_up, fetched_down = moved("gold_in_tool_result")
        held_up, held_down = moved("gold_in_admitted_packets")
        runs_up = sum(
            max(0, r["arms"][high]["runs_exact"] - r["arms"][low]["runs_exact"]) for r in pairs
        )
        runs_down = sum(
            max(0, r["arms"][low]["runs_exact"] - r["arms"][high]["runs_exact"]) for r in pairs
        )
        return [
            "",
            f"## {title} — {low} vs {high}",
            "",
            f"- Gold fetched: +{len(fetched_up)} / -{len(fetched_down)} "
            f"{[r['task_id'][:8] for r in fetched_up]}",
            f"- Gold **propagated**: +{len(held_up)} / -{len(held_down)} "
            f"{[r['task_id'][:8] for r in held_up]}",
            f"- Correct runs: +{runs_up} / -{runs_down} (net {runs_up - runs_down:+d})",
            f"- {note}",
        ]

    labels = [label for _b, _a, label in budgets]
    if len(labels) >= 2:
        lines += contrast(
            labels[0],
            labels[1],
            "Budget",
            "Gold fetched not rising means the bottleneck is not the budget.",
        )
    # The pair that decides the diagnosis: same budget, same cap, admission only.
    # If propagation rises here, first-come is the defect and the fix is free.
    for low in labels:
        for high in labels:
            if low == high:
                continue
            lb, la, _ = next(a for a in budgets if a[2] == low)
            hb, ha, _ = next(a for a in budgets if a[2] == high)
            if lb == hb and la == "fifo" and ha == "replace":
                lines += contrast(
                    low,
                    high,
                    "Admission at fixed cap",
                    "Propagation rising at identical context size isolates first-come "
                    "as the defect; flat means the cap is genuinely too small.",
                )

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    budgets = parse_arms(args.arms)
    if len(budgets) < 2:
        raise SystemExit("--arms needs at least two arms, e.g. 2,5 or 2,5,5:replace")

    treatment, control = derive_task_sets(Path(args.from_run).resolve())
    roles = {task_id: "treatment" for task_id in treatment}
    roles.update({task_id: "control" for task_id in control})
    if args.task_ids:
        wanted = {value.strip() for value in args.task_ids.split(",") if value.strip()}
        roles = {k: v for k, v in roles.items() if any(k.startswith(p) for p in wanted)}

    dataset = GAIADataset(
        split=args.split, level=args.level, local_data_dir=Path(args.data_dir).resolve()
    )
    samples = [item for item in dataset.load() if str(item.get("task_id") or "") in roles]
    if args.max_tasks:
        samples = samples[: args.max_tasks]

    output_dir = ROOT / "outputs" / args.log_name
    (output_dir / "tasks").mkdir(parents=True, exist_ok=True)
    tool_manager = PairedToolManager(ToolManager())
    agents = [
        AgentConfig(agent_id=agent_id, model_name=model_name, temperature=0.5)
        for agent_id, model_name in DEFAULT_AGENT_SPECS
    ]

    print(
        f"[INFO] budget ablation: {len(samples)} tasks "
        f"({sum(1 for s in samples if roles[str(s['task_id'])] == 'treatment')} treatment), "
        f"arms={budgets}, seed={describe_seed()}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        task_id = str(sample.get("task_id") or "")
        role = roles[task_id]
        print(f"\n===== {index}/{len(samples)} {task_id[:8]} ({role}) =====", flush=True)
        try:
            result = run_task(
                sample,
                budgets=budgets,
                reverse=index % 2 == 0,
                role=role,
                args=args,
                tool_manager=tool_manager,
                agents=agents,
            )
        except Exception as exc:  # one bad task must not end the sweep
            result = {
                "task_id": task_id,
                "role": role,
                "expected": str(sample.get("final_answer") or ""),
                "gold_comparable": False,
                "error": f"{type(exc).__name__}: {exc}",
                "arms": {},
            }
        results.append(result)
        (output_dir / "tasks" / f"{index:03d}_{task_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        if result.get("error"):
            print(f"  ERROR {result['error']}", flush=True)
        else:
            for key, arm in result["arms"].items():
                policy = arm.get("policy") or {}
                churn = (
                    f" dedup={policy['refused_duplicates']} evict={policy['evictions']}"
                    if policy
                    else ""
                )
                print(
                    f"  {key:>10}: gold_tool={arm['gold_in_tool_result']} "
                    f"gold_pkt={arm['gold_in_admitted_packets']} "
                    f"auth={arm['authorized_executions']} "
                    f"net={(arm.get('backend_calls') or {}).get('executed', 0)} "
                    f"slots={arm['packets_admitted']}/{arm['packets_distinct_queries']} "
                    f"runs={arm['runs_exact']}/{arm['runs_total']} "
                    f"{arm['seconds']:.0f}s{churn}",
                    flush=True,
                )

    report = output_dir / f"{args.log_name}.md"
    write_report(results, budgets, report, args.log_name)
    print(f"\n[OK] {report}")


if __name__ == "__main__":
    main()
