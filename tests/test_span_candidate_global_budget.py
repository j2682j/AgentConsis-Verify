"""Pin the batch-global span budgets, which per-document tests cannot reach.

`PassageEvidenceUnitBuilder.max_units` and
`IterativeRetrievalControl._span_role_candidates(max_total=...)` bound a whole
round, not a document. A single-document test never reaches either, which is how
a global budget of 10 survived while `max_units_per_document` was being raised
from 3 to 6: rounds carried 21.9 documents on average, so the global bound was
always the one that bit, and 1918 of 2535 documents contributed no candidate.

These tests therefore use many documents and assert on what the budget decides:
whether a document late in the ordering contributes at all, and whether the
second budget in the chain re-truncates the first.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.evidence.candidate_span_quality_gate import (
    CandidateSpanQualityGate,
)
from tools.search_result_builder.evidence.passage_evidence_unit_builder import (
    PassageEvidenceUnitBuilder,
)
from tools.search_result_builder.evidence.span_role_classifier import CandidateSpan
from tools.search_result_builder.retrieval_control import (
    IterativeRetrievalControl,
    RetrievalRoundTrace,
    RetrievedDocumentTrace,
)

DOCUMENT_COUNT = 10
ANSWER_DOCUMENT_INDEX = 8
ANSWER_FRAGMENT = "wojciech"


def _documents() -> list[dict[str, object]]:
    """Ten documents of two sentences each, the answer in the eighth.

    Without an embedder the builder keeps retrieval order, so the answer is
    reachable only if the global budget spans more than a handful of documents.
    """

    documents = []
    for index in range(1, DOCUMENT_COUNT + 1):
        if index == ANSWER_DOCUMENT_INDEX:
            body = (
                "The final round settled the contest. "
                "The award went to Wojciech that year."
            )
        else:
            body = (
                f"Report {index} opens with procedural notes. "
                f"Report {index} closes without naming a winner."
            )
        documents.append(
            {
                "id": f"doc-{index}",
                "title": f"Report {index}",
                "text": body,
                "record_type": "passage",
            }
        )
    return documents


class PassageEvidenceUnitBuilderGlobalBudgetTest(unittest.TestCase):
    def test_default_global_budget_is_forty(self) -> None:
        self.assertEqual(PassageEvidenceUnitBuilder().max_units, 40)

    def test_answer_in_a_late_document_survives_the_default_budget(self) -> None:
        builder = PassageEvidenceUnitBuilder()

        result = builder.build(question="Who won the award?", documents=_documents())

        texts = " ".join(unit.text for unit in result.units).casefold()
        self.assertIn(ANSWER_FRAGMENT, texts)

    def test_a_global_budget_of_ten_would_drop_the_answer(self) -> None:
        """Guard the regression direction: the old budget loses the answer."""

        builder = PassageEvidenceUnitBuilder(max_units=10)

        result = builder.build(question="Who won the award?", documents=_documents())

        texts = " ".join(unit.text for unit in result.units).casefold()
        self.assertNotIn(ANSWER_FRAGMENT, texts)

    def test_every_document_contributes_under_the_default_budget(self) -> None:
        builder = PassageEvidenceUnitBuilder()

        result = builder.build(question="Who won the award?", documents=_documents())

        contributing = {unit.document_index for unit in result.units}
        self.assertEqual(len(contributing), DOCUMENT_COUNT)


class SpanRoleCandidateGlobalBudgetTest(unittest.TestCase):
    """The second budget in the chain, inside RetrievalControl."""

    def _round(self) -> RetrievalRoundTrace:
        round_trace = RetrievalRoundTrace(round_index=0, query="Who won the award?")
        documents = []
        for index in range(1, DOCUMENT_COUNT + 1):
            trace = RetrievedDocumentTrace(
                document_id=f"doc-{index}",
                title=f"Report {index}",
                text=f"Report {index} body.",
                url=f"https://example/doc-{index}",
                retrieval_score=1.0 - index / 100,
            )
            trace.useful_spans = [
                f"Report {index} first relevant sentence",
                f"Report {index} second relevant sentence",
                f"Report {index} third relevant sentence",
            ]
            documents.append(trace)
        round_trace.documents = documents
        return round_trace

    def test_default_total_matches_the_builder_budget(self) -> None:
        control = IterativeRetrievalControl.__new__(IterativeRetrievalControl)

        candidates, _ = control._span_role_candidates(self._round())

        self.assertGreater(len(candidates), 15)
        self.assertEqual(len(candidates), 3 * DOCUMENT_COUNT)

    def test_a_total_of_fifteen_would_retruncate_the_wider_budget(self) -> None:
        """Guard the stacked-budget failure: raising only the first does nothing."""

        control = IterativeRetrievalControl.__new__(IterativeRetrievalControl)

        candidates, _ = control._span_role_candidates(self._round(), max_total=15)

        self.assertEqual(len(candidates), 15)


class CandidateSpanQualityGateBudgetTest(unittest.TestCase):
    """The fourth budget, and the one level1_final_07 truncated at.

    Raising the builder budget to 40 moved candidate units per round from 8.5 to
    30.2, and the spans reaching the classifier still stopped at exactly 15 --
    this gate's own cap, left behind.
    """

    def _candidates(self, count: int) -> list[CandidateSpan]:
        return [
            CandidateSpan(
                id=str(index),
                text=f"The committee recorded outcome number {index} that season.",
                local_context=f"Context for outcome number {index}.",
                source_title="Report",
            )
            for index in range(1, count + 1)
        ]

    def test_default_cap_matches_the_other_budgets(self) -> None:
        self.assertEqual(CandidateSpanQualityGate().max_candidates, 40)

    def test_gate_passes_more_than_fifteen_candidates(self) -> None:
        result = CandidateSpanQualityGate().filter_candidates(self._candidates(30))

        self.assertGreater(len(result.candidates), 15)

    def test_a_cap_of_fifteen_would_retruncate_the_wider_budget(self) -> None:
        """Guard the regression direction: the old cap is the binding one."""

        result = CandidateSpanQualityGate(max_candidates=15).filter_candidates(
            self._candidates(30)
        )

        self.assertEqual(len(result.candidates), 15)


if __name__ == "__main__":
    unittest.main()
