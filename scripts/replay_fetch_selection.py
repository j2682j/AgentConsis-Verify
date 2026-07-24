"""Offline replay of Plan-12 fetch selection over saved GAIA runs.

Rebuilds each task's candidate sources from the recorded search trace and
compares the legacy rank-only fetch policy with the new signal-driven
selector. Reports the acceptance criteria from the plan:

- every page the legacy policy fetched is still fetched (additive safety),
- named sources reach the initial batch,
- echo / product pages stop monopolising the batch.

Saved runs only record a subset of candidate metadata, so a source that was
already deduplicated away cannot be reconstructed. Tasks whose trace lacks
candidate URLs are reported as unreplayable rather than silently skipped.

Usage:
    python scripts/replay_fetch_selection.py outputs/level1_full_system_final_2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_result_builder.config import SearchSourceCandidate
from tools.search_result_builder.source_analyze.seer.fetch_candidate_selector import (
    FetchCandidateSelector,
    TIER_DEMOTED,
    TIER_ECHO,
    TIER_NAMED_SOURCE,
)
from tools.search_result_builder.source_analyze.seer.source_filter import SourceFilter
from tools.search_result_builder.source_analyze.seer.source_selection_signals import (
    SourceSelectionSignalBuilder,
)


def candidates_from_task(data: dict) -> tuple[list[SearchSourceCandidate], dict[str, str]]:
    """Rebuild candidate sources from the recorded retrieval documents."""
    meta = (data.get("network_summary") or {}).get("metadata") or {}
    sources: list[SearchSourceCandidate] = []
    query_text_by_id: dict[str, str] = {}
    seen: set[str] = set()
    for usage in meta.get("tool_usage") or []:
        raw = usage.get("raw_result") if isinstance(usage.get("raw_result"), dict) else {}
        retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        for index, query in enumerate(raw.get("queries") or [], start=1):
            query_text_by_id.setdefault(f"Q{index}", str(query))
        for round_info in retrieval.get("rounds") or []:
            query_id = f"Q{int(round_info.get('round_index', 1) or 1)}"
            query_text_by_id.setdefault(query_id, str(round_info.get("query") or ""))
            for document in round_info.get("documents") or []:
                if not isinstance(document, dict):
                    continue
                url = str(document.get("url") or "")
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append(
                    SearchSourceCandidate(
                        source_id=f"S{len(sources) + 1}",
                        query_id=query_id,
                        title=str(document.get("title") or ""),
                        url=url,
                        snippet=str(document.get("text") or "")[:400],
                        rank=len(sources),
                    )
                )
    return sources, query_text_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--legacy-limit", type=int, default=6)
    parser.add_argument("--initial-limit", type=int, default=8)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "tasks", "*.json")))
    rows = []
    unreplayable = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        task = os.path.basename(path)[:3]
        sources, query_text_by_id = candidates_from_task(data)
        if not sources:
            if (data.get("network_summary") or {}).get("metadata", {}).get("search_used"):
                unreplayable.append(task)
            continue
        question = str(data.get("question") or "")

        legacy = [item.url for item in sources[: args.legacy_limit]]

        SourceSelectionSignalBuilder().build(
            sources, question=question, query_text_by_id=query_text_by_id
        )
        selector = FetchCandidateSelector(
            legacy_fetch_limit=args.legacy_limit,
            initial_fetch_limit=args.initial_limit,
            promoted_slots=args.initial_limit - args.legacy_limit,
            demoted_domain_markers=SourceFilter.DEMOTED_DOMAIN_MARKERS,
            product_page_markers=SourceFilter.PRODUCT_PAGE_MARKERS,
        )
        result = selector.select(sources, fetch_limit=args.initial_limit)
        initial = [item.url for item in result.initial_sources]

        rows.append(
            {
                "task": task,
                "exact": bool(data.get("exact_match")),
                "candidates": len(sources),
                "legacy": legacy,
                "initial": initial,
                "kept_all_legacy": all(url in initial for url in legacy),
                "promoted": [url for url in initial if url not in legacy],
                "named_in_initial": sum(
                    1 for item in result.initial_sources if item.named_source_match
                ),
                "named_total": sum(1 for item in sources if item.named_source_match),
                "echo_in_initial": sum(
                    1
                    for item in result.initial_sources
                    if item.fetch_priority_tier in {TIER_ECHO, TIER_DEMOTED}
                ),
                "echo_in_legacy": sum(
                    1
                    for item in sources[: args.legacy_limit]
                    if item.fetch_priority_tier in {TIER_ECHO, TIER_DEMOTED}
                ),
                "deferred": len(result.deferred_sources),
            }
        )

    if unreplayable:
        print(
            f"UNREPLAYABLE: {len(unreplayable)} search task(s) have no candidate URLs "
            f"in the saved trace: {', '.join(unreplayable)}"
        )
        print()

    broke_legacy = [r for r in rows if not r["kept_all_legacy"]]
    promoted_named = [r for r in rows if r["named_in_initial"]]
    print(f"replayed tasks: {len(rows)}")
    if broke_legacy:
        print(f"!! ADDITIVE SAFETY VIOLATED on {len(broke_legacy)} task(s):")
        for r in broke_legacy:
            print(f"   {r['task']}: lost {[u for u in r['legacy'] if u not in r['initial']]}")
    else:
        print("additive safety: every legacy fetch is preserved on all tasks")
    print(
        f"tasks with a named source promoted into the initial batch: {len(promoted_named)}"
    )
    echo_before = sum(r["echo_in_legacy"] for r in rows)
    echo_after = sum(r["echo_in_initial"] for r in rows)
    print(f"echo/product pages in fetch batch: {echo_before} (legacy) -> {echo_after} (new)")
    print()
    print("task | exact | cand | legacy | initial | named(in/tot) | echo(leg->new) | promoted")
    for r in rows:
        print(
            f" {r['task']}  | {str(r['exact']):5} | {r['candidates']:4} |"
            f" {len(r['legacy']):6} | {len(r['initial']):7} |"
            f" {r['named_in_initial']}/{r['named_total']:<11} |"
            f" {r['echo_in_legacy']}->{r['echo_in_initial']:<11} | {len(r['promoted'])}"
        )
    return 2 if broke_legacy else 0


if __name__ == "__main__":
    raise SystemExit(main())
