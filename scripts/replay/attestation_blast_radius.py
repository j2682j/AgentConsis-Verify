"""What would a morphology-aware attestation count actually touch?

`_drop_unattested_candidates` reserves a candidate the fetched corpus never
states, once some rival clears three mentions. It counts exact surface forms
only, so task 034's `Rockhopper penguins` scored zero while the corpus states
`rockhopper penguin` -- the same species, singular.

Teaching the counter about inflection is a change to a gate that measures 8
helps against 2 hurts over five runs, so the set of candidates it can reach has
to be known before an A/B, not after. Two failures this makes visible that an
A/B alone does not:

  - a change that reaches nothing still reports "no regressions", which is how
    a requirement-gate repair passed five runs while never executing
  - a change that reaches candidates the corpus genuinely never states is
    unsafe regardless of what this run's winners happen to do

Counts are per unique task-candidate. Five runs replay the same 53 tasks, so
raw event counts multiply the same finding by five.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import json
import os
from typing import Any

from benchmark.gaia.answer_matcher import exact_match
from scripts.replay.corpus_attestation_diagnostics import (
    PRODUCTION_WINDOW_CHARS,
    classify,
    morphological_variants,
)
from scripts.replay.retrieval_document_reader import documents, mention_report

RUNS = (
    "level1_final_13",
    "level1_final_15",
    "level1_final_16",
    "level_1_final_20",
    "level1_final_21",
)


@dataclass
class AffectedCandidate:
    run: str
    task_id: str
    candidate: str
    production_mentions: int
    full_exact_mentions: int
    window_exact_mentions: int
    morph_mentions: int
    morph_document_count: int
    matched_surfaces: list[dict[str, Any]] = field(default_factory=list)
    classification: str = "UNCLASSIFIED"
    current_state: str = ""
    current_winner: str = ""
    gold: str = ""
    supporting_runs: int = 0
    winner_supporting_runs: int = 0
    task_currently_exact: bool = False

    @property
    def task_candidate_key(self) -> str:
        """Identity across runs: the same finding repeated is one finding."""

        return f"{self.task_id}|{self.candidate.casefold()}"

    @property
    def winner_sensitive(self) -> bool:
        """Could readmitting this candidate move the winner?

        A prediction, not a measurement: admitting it only matters if it can
        outvote the current winner. The A/B is what confirms it.
        """

        return (
            self.classification == "MORPHOLOGY_BLIND"
            and self.supporting_runs > self.winner_supporting_runs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "task_id": self.task_id,
            "candidate": self.candidate,
            "production_mentions": self.production_mentions,
            "morph_mentions": self.morph_mentions,
            "morph_document_count": self.morph_document_count,
            "matched_surfaces": self.matched_surfaces[:4],
            "classification": self.classification,
            "current_state": self.current_state,
            "current_winner": self.current_winner,
            "counterfactual_state": "active",
            "winner_sensitive": self.winner_sensitive,
            "gold": self.gold,
            "task_currently_exact": self.task_currently_exact,
        }


def _windowed_documents(docs):
    kept, total = [], 0
    for document in docs:
        if total >= PRODUCTION_WINDOW_CHARS:
            break
        kept.append(document)
        total += len(document.text)
    return kept


def scan_task(run: str, path: str) -> list[AffectedCandidate]:
    task = json.loads(open(path, encoding="utf-8").read())
    task_id = os.path.basename(path).split("_")[0]
    gold = str(task.get("expected") or "")
    meta = (task.get("network_summary") or {}).get("metadata") or {}
    trace = ((meta.get("winner_selection") or {}).get("selection_trace") or {})
    gates = trace.get("gate_trace") or []

    attestation = next(
        (g for g in gates if g.get("gate_name") == "corpus_attestation"), None
    )
    if not attestation:
        return []

    # Only where the gate actually counted. `_apply_corpus_attestation_gate`
    # returns early on several paths -- nothing countable, no rival clearing the
    # floor -- and those decisions carry an empty `details`. Reading a missing
    # `corpus_mentions` as zero turns "never measured" into "measured zero", and
    # it inflated this scan from a handful of candidates to 368 of them, almost
    # all from tasks whose attestation never ran a count at all.
    mentions: dict[str, int] = {}
    states: dict[str, str] = {}
    for decision in attestation.get("decisions") or []:
        details = decision.get("details") or {}
        if "corpus_mentions" not in details:
            continue
        key = str(decision.get("candidate_key"))
        mentions[key] = int(details.get("corpus_mentions") or 0)
        states[key] = str(decision.get("outcome") or "")
    if not mentions:
        return []

    zero_keys = [key for key, count in mentions.items() if count == 0]
    if not zero_keys:
        return []

    by_key = {str(c.get("candidate_key")): c for c in (trace.get("candidates") or [])}
    winner_answer = str(trace.get("selected_answer") or "")
    winner_runs = 0
    for candidate in trace.get("candidates") or []:
        if str(candidate.get("answer") or "") == winner_answer:
            winner_runs = int(candidate.get("supporting_run_count") or 0)
            break

    docs = documents(task)
    window_docs = _windowed_documents(docs)

    out: list[AffectedCandidate] = []
    for key in zero_keys:
        candidate = by_key.get(key) or {}
        answer = str(candidate.get("answer") or "")
        if not answer:
            continue
        full = mention_report(docs, answer)
        window = mention_report(window_docs, answer)
        morph_total = 0
        morph_docs: set[str] = set()
        surfaces: list[dict[str, Any]] = []
        for variant in morphological_variants(answer):
            report = mention_report(docs, variant)
            morph_total += report.occurrences
            for hit in report.hits:
                morph_docs.add(hit.url or hit.document_id)
                surfaces.append(hit.to_dict())

        row = AffectedCandidate(
            run=run,
            task_id=task_id,
            candidate=answer,
            production_mentions=mentions[key],
            full_exact_mentions=full.occurrences,
            window_exact_mentions=window.occurrences,
            morph_mentions=morph_total,
            morph_document_count=len(morph_docs),
            matched_surfaces=surfaces,
            current_state=states.get(key, ""),
            current_winner=winner_answer,
            gold=gold,
            supporting_runs=int(candidate.get("supporting_run_count") or 0),
            winner_supporting_runs=winner_runs,
            task_currently_exact=bool(task.get("exact_match")),
        )
        # `classify` reads the same names the diagnostics module uses.
        from scripts.replay.corpus_attestation_diagnostics import CandidateAttestation

        row.classification = classify(
            CandidateAttestation(
                candidate=answer,
                production_mentions=row.production_mentions,
                full_corpus_exact_mentions=row.full_exact_mentions,
                attestation_window_exact_mentions=row.window_exact_mentions,
                canonical_morph_mentions=row.morph_mentions,
            )
        )
        # Only candidates the intervention would actually move are in scope.
        if row.morph_mentions > 0 or row.classification != "TRULY_UNATTESTED":
            out.append(row)
        else:
            out.append(row)
    return out


def scan(runs: tuple[str, ...] = RUNS) -> list[AffectedCandidate]:
    rows: list[AffectedCandidate] = []
    for run in runs:
        for path in sorted(glob.glob(f"c:/SCP/outputs/{run}/tasks/*.json")):
            rows.extend(scan_task(run, path))
    return rows


def report(rows: list[AffectedCandidate]) -> None:
    moved = [row for row in rows if row.morph_mentions > 0]
    unique = {row.task_candidate_key for row in moved}
    classes: dict[str, set[str]] = {}
    for row in moved:
        classes.setdefault(row.classification, set()).add(row.task_candidate_key)

    print(f"attestation 判為 0 命中的候選（不重複 task-candidate）: "
          f"{len({r.task_candidate_key for r in rows})}")
    print(f"morphology 會改成非零的（不重複）: {len(unique)}\n")
    print("分類分佈（不重複 task-candidate）:")
    for name in sorted(classes):
        print(f"   {name:<28} {len(classes[name])}")
    unmoved = {r.task_candidate_key for r in rows} - unique
    print(f"   {'（不受影響，維持 0）':<28} {len(unmoved)}")

    sensitive = [row for row in moved if row.winner_sensitive]
    print(f"\nwinner_sensitive（票數足以翻動勝者）: "
          f"{len({r.task_candidate_key for r in sensitive})}")
    for row in sensitive:
        print(f"   {row.run[-6:]}/{row.task_id}  {row.candidate[:34]!r} "
              f"runs={row.supporting_runs} vs winner {row.winner_supporting_runs}")

    at_risk = [row for row in moved if row.task_currently_exact]
    print(f"\n碰到既有正確題目的候選: {len({r.task_candidate_key for r in at_risk})}")
    for row in at_risk:
        print(f"   {row.run[-6:]}/{row.task_id}  目前 ✓ winner={row.current_winner[:26]!r}"
              f"  受影響候選={row.candidate[:26]!r} runs={row.supporting_runs}"
              f" vs {row.winner_supporting_runs}")

    print("\n受影響候選明細:")
    seen: set[str] = set()
    for row in moved:
        if row.task_candidate_key in seen:
            continue
        seen.add(row.task_candidate_key)
        surface = row.matched_surfaces[0] if row.matched_surfaces else {}
        print(f"   {row.task_id} {row.candidate[:30]!r:<32} "
              f"prod={row.production_mentions} morph={row.morph_mentions}"
              f"/{row.morph_document_count}份  {row.classification}")
        if surface:
            print(f"        來源 {surface.get('surface')!r} "
                  f"doc={surface.get('document_id')} span={surface.get('character_span')}")


if __name__ == "__main__":
    report(scan())
