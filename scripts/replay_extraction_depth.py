"""Measure whether the answer survives page extraction and corpus building.

A page can be found, fetched and still not help: the extractor caps the main
article text, so on long pages the answer is discarded before any ranking
happens. This script fetches the pages that hold known GAIA answers and reports
where the answer is lost.

Reported per page:
  html          size of the raw document
  extracted     text the main-content extractor produced
  kept          text that survived the section caps
  in_extracted  answer present in the extractor output
  in_kept       answer present after the caps      <- the number that matters
  chunks        chunks that would enter the corpus <- must not grow

Usage:
    python scripts/replay_extraction_depth.py
    python scripts/replay_extraction_depth.py --json baseline.json
    python scripts/replay_extraction_depth.py --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from tools.search_result_builder.corpus.chunker import DocumentChunker
from tools.search_result_builder.corpus.web_corpus_builder import WebCorpusBuilder
from tools.search_result_builder.source_analyze.seer.page_content_fetcher import (
    REQUEST_HEADERS,
    _extract_with_trafilatura,
    fetch_page_content_result,
)


# Pages that demonstrably contain a GAIA answer, with the string to look for.
# Chosen from tasks whose answer never reached the corpus in level1 run 4.
CASES = [
    {
        "task": "032",
        "url": "https://scikit-learn.org/stable/whats_new/v0.19.html",
        "answer": "BaseLabelPropagation",
        "question": "In the Scikit-Learn July 2017 changelog, what other predictor base command received a bug fix?",
    },
    {
        "task": "039",
        "url": "https://www.law.cornell.edu/rules/fre/rule_101",
        "answer": "inference",
        "question": "Under the fifth section of federal rules alphabetically, what word was deleted in the last amendment to the first rule?",
    },
    {
        "task": "029",
        "url": "https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Introductory_Chemistry_(CK-12)/01%3A_Introduction_to_Chemistry/1.E%3A_Exercises",
        "answer": "Louvrier",
        "question": "What is the surname of the equine veterinarian mentioned in 1.E Exercises?",
    },
    {
        "task": "002",
        "url": "https://en.wikipedia.org/wiki/Mercedes_Sosa",
        "answer": "Cantora",
        "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
    },
    {
        "task": "020",
        "url": "https://www.merriam-webster.com/word-of-the-day/jingoism-2022-06-27",
        "answer": "Annie Levin",
        "question": "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?",
    },
    {
        "task": "047",
        "url": "https://clinicaltrials.gov/study/NCT03411733",
        "answer": "90",
        "question": "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018?",
    },
    {
        "task": "013",
        "url": "https://journal.finfar.org/articles/dragons-are-tricksy-the-uncanny-dragons-of-childrens-literature/",
        "answer": "fluffy",
        "question": "In Emily Midkiff's June 2014 article, what word was quoted from two different authors in distaste for the nature of dragon depictions?",
    },
    {
        "task": "006",
        "url": "https://pietromurano.org/publications.html",
        "answer": "Mapping Human Oriented Information",
        "question": "What was the title of the first paper authored by Pietro Murano?",
    },
]


def measure(case: dict, chunk_budget: int) -> dict:
    url = case["url"]
    try:
        html = requests.get(url, headers=REQUEST_HEADERS, timeout=25).text
    except Exception as exc:  # network is inherently flaky; report, don't crash
        return {"task": case["task"], "error": f"{type(exc).__name__}: {exc}"}

    extracted = ""
    got = _extract_with_trafilatura(html, url)
    if got and got[0]:
        extracted = got[0]

    result = fetch_page_content_result(url, max_tokens=8000)
    kept = (result.content if result else "") or ""

    answer = case["answer"]
    # Run the real corpus builder so chunk selection is measured, not simulated.
    builder = WebCorpusBuilder()
    records = builder.build_records(
        [{
            "source_id": "S1",
            "query_id": "Q1",
            "title": case["task"],
            "url": url,
            "raw_content": kept,
            "snippet": "",
        }],
        fetch_missing=False,
        max_chunks_per_url=chunk_budget,
        max_records=chunk_budget,
        question=case["question"],
    )
    selected = [r.text for r in records]
    chunks = DocumentChunker().chunk(kept)
    return {
        "task": case["task"],
        "url": url,
        "html": len(html),
        "extracted": len(extracted),
        "kept": len(kept),
        "in_extracted": answer.casefold() in extracted.casefold(),
        "in_kept": answer.casefold() in kept.casefold(),
        "chunks_total": len(chunks),
        "chunks_selected": len(selected),
        "in_selected_chunks": any(answer.casefold() in c.casefold() for c in selected),
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write results here as a baseline")
    parser.add_argument("--compare", help="compare against a saved baseline")
    parser.add_argument("--chunk-budget", type=int, default=9,
                        help="chunks per page the corpus accepted in run 4")
    args = parser.parse_args()

    rows = [measure(c, args.chunk_budget) for c in CASES]

    print(f"{'task':5} {'html':>8} {'extract':>8} {'kept':>7} {'lost%':>6} "
          f"{'inExtr':>7} {'inKept':>7} {'chunks':>7} {'inSel':>6}  answer")
    print("-" * 96)
    for r in rows:
        if r.get("error"):
            print(f"{r['task']:5} ERROR {r['error'][:70]}")
            continue
        lost = 100 * (1 - r["kept"] / r["extracted"]) if r["extracted"] else 0
        print(f"{r['task']:5} {r['html']:>8} {r['extracted']:>8} {r['kept']:>7} "
              f"{lost:>5.0f}% {str(r['in_extracted']):>7} {str(r['in_kept']):>7} "
              f"{r['chunks_selected']:>3}/{r['chunks_total']:<3} "
              f"{str(r['in_selected_chunks']):>6}  {r['answer']}")

    ok = [r for r in rows if not r.get("error")]
    print()
    print(f"answer survives extraction : {sum(1 for r in ok if r['in_extracted'])}/{len(ok)}")
    print(f"answer survives section cap: {sum(1 for r in ok if r['in_kept'])}/{len(ok)}")
    print(f"answer reaches the corpus  : {sum(1 for r in ok if r['in_selected_chunks'])}/{len(ok)}")
    print(f"chunks entering corpus     : {sum(r['chunks_selected'] for r in ok)} (must not grow)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
        print(f"\nbaseline written to {args.json}")

    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            base = {r["task"]: r for r in json.load(handle)}
        print("\n=== vs baseline ===")
        regressed = False
        for r in ok:
            b = base.get(r["task"])
            if not b or b.get("error"):
                continue
            gained = r["in_selected_chunks"] and not b["in_selected_chunks"]
            lost_it = b["in_selected_chunks"] and not r["in_selected_chunks"]
            grew = r["chunks_selected"] > b["chunks_selected"]
            regressed = regressed or lost_it or grew
            mark = "+" if gained else ("-" if lost_it else " ")
            print(f"  [{mark}] {r['task']}: kept {b['kept']}->{r['kept']} "
                  f"answer_in_corpus {b['in_selected_chunks']}->{r['in_selected_chunks']} "
                  f"chunks {b['chunks_selected']}->{r['chunks_selected']}"
                  f"{'   *** CHUNKS GREW' if grew else ''}")
        print("REGRESSION" if regressed else "no regression: nothing lost, no chunk growth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
