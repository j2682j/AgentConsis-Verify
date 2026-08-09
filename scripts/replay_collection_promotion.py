"""Offline replay of collection-link promotion over a saved GAIA run.

A structured corpus record is a stub: it carries a title and a `content_url`
but no page body (`original_content_chars = 0`). `IterativeRetrievalControl`
promotes some of them by fetching that link, bounded by
`max_collection_links_to_fetch = 3`. On level1_final_06 that left 61.7% of
everything reaching retrieval as bodyless stubs, six tasks promoted nothing at
all, and pages holding the answer -- a scikit-learn changelog, an FRE article, a
clinical-trial record -- stayed unread behind links the run had already found.

This script rebuilds the promotion candidates from a finished run and ranks them
the way the control does, to answer what the budget should be rather than guess:
at what rank does the link holding the answer sit?

Two arms:
  dry-run (default)  rank offline and report how many fetches each budget costs
  --fetch            fetch content_urls in rank order and score gold recovery,
                     which is the ground truth and the only arm that reaches the
                     network; --max-rank bounds it

Fidelity: the control ranks with `retriever.search` over the run's FAISS index;
this ranks the same record texts by bge-m3 cosine similarity, which reproduces
the ordering but not index-level tie-breaking. Stub counts and the rank-1..3
selection are reported so the reconstruction can be compared against the run.

Usage:
    python scripts/replay_collection_promotion.py [--run outputs/level1_final_06]
                                                  [--only-missing] [--fetch]
                                                  [--max-rank 20]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_RE = re.compile(r'"corpus_path"\s*:\s*"((?:[^"\\]|\\.)*)"')
ABBREV = {"saint": "st", "mount": "mt", "fort": "ft", "doctor": "dr"}


def loose(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]+", " ", str(value or "").casefold())
    return "".join(ABBREV.get(word, word) for word in text.split())


def corpus_paths(raw: str) -> list[str]:
    return sorted({json.loads(f'"{match}"') for match in CORPUS_RE.findall(raw)})


def load_tasks(run_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        task = json.loads(raw)
        task["_id"] = path.name[:3]
        task["_corpus"] = corpus_paths(raw)
        tasks.append(task)
    return tasks


def promotion_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Stubs the control would consider: typed, linked, not the parent page."""

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for corpus in task["_corpus"]:
        if not os.path.exists(corpus):
            continue
        with open(corpus, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_type = str(record.get("record_type") or "").strip()
                content_url = str(record.get("content_url") or "").strip()
                parent_url = str(record.get("parent_url") or "").strip()
                if record_type in {"", "passage"} or not content_url:
                    continue
                if content_url.casefold() == parent_url.casefold():
                    continue
                key = content_url.casefold()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(record)
    return candidates


def rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    embedder: Any,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    import numpy as np

    query = (
        embedder.prepare_query_text(question)
        if hasattr(embedder, "prepare_query_text")
        else question
    )
    texts = [
        (
            embedder.prepare_passage_text(str(record.get("text") or ""))
            if hasattr(embedder, "prepare_passage_text")
            else str(record.get("text") or "")
        )
        for record in candidates
    ]
    vectors = np.asarray(embedder.embed([query, *texts]), dtype=np.float32)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    norms = np.maximum(np.linalg.norm(doc_vectors, axis=1), 1e-12)
    scores = (doc_vectors @ query_vector) / (
        norms * max(float(np.linalg.norm(query_vector)), 1e-12)
    )
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order]


def gold_rank_by_fetch(
    ranked: list[dict[str, Any]],
    gold: str,
    max_rank: int,
) -> int:
    """Fetch links in rank order; return the 1-based rank that holds the answer."""

    from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
        fetch_page_content_result,
    )

    for position, record in enumerate(ranked[:max_rank], start=1):
        url = str(record.get("content_url") or "")
        try:
            result = fetch_page_content_result(url, max_tokens=5000)
        except Exception:
            continue
        if result is None:
            continue
        if gold and gold in loose(result.content):
            return position
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/level1_final_06")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="score only tasks whose gold answer never reached retrieval",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="fetch content_urls to score gold recovery (reaches the network)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    tasks = [
        task
        for task in load_tasks(run_dir)
        if (task["search_summary"].get("web_search_count") or 0)
    ]

    if args.only_missing:
        kept = []
        for task in tasks:
            gold = loose(task.get("expected"))
            reached = any(
                gold and gold in loose(doc.get("text"))
                for round_trace in task["search_summary"].get("retrieval_rounds") or []
                for doc in round_trace.get("documents") or []
            )
            if not reached:
                kept.append(task)
        tasks = kept

    from tools.search_result_builder.embeddings.embedder import Embedder

    embedder = Embedder(args.model, batch_size=64, text_normalize=True)

    print(f"run: {run_dir}")
    print(f"tasks scored: {len(tasks)}")
    print(f"current max_collection_links_to_fetch = 3")
    print()
    header = "%-4s %-3s %-24s %7s %9s" % ("id", "ex", "expected", "stubs", "distinct")
    print(header + ("  gold_rank" if args.fetch else ""))

    rows = []
    for task in tasks:
        candidates = promotion_candidates(task)
        ranked = rank_candidates(str(task.get("question") or ""), candidates, embedder)
        gold_rank = -1
        if args.fetch:
            gold_rank = gold_rank_by_fetch(ranked, loose(task.get("expected")), args.max_rank)
        rows.append((task, ranked, gold_rank))
        line = "%-4s %-3s %-24s %7d %9d" % (
            task["_id"],
            "T" if task["exact_match"] else ".",
            str(task["expected"])[:22],
            len(candidates),
            len({str(r.get("content_url") or "").casefold() for r in candidates}),
        )
        if args.fetch:
            line += "  %9s" % (gold_rank if gold_rank else "not found")
        print(line)

    print()
    total = sum(len(ranked) for _, ranked, _ in rows)
    print(f"promotion candidates in total : {total}")
    print(f"reachable at the current budget: {3 * len(rows)}  ({100 * 3 * len(rows) / max(total, 1):.1f}%)")
    print()
    print("budget → fetches this run would issue:")
    for budget in (3, 6, 10, 15, 20, 30):
        cost = sum(min(budget, len(ranked)) for _, ranked, _ in rows)
        line = f"  {budget:>3} → {cost:>5} fetches"
        if args.fetch:
            recovered = sum(
                1 for _, _, rank in rows if rank and rank <= budget
            )
            line += f"   gold recovered: {recovered}/{len(rows)}"
        print(line)

    if not args.fetch:
        print()
        print("dry run: no network. Re-run with --fetch to score gold recovery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
