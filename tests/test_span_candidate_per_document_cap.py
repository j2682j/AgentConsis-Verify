"""Pin the per-document span cap that decides answer reachability.

On level1_final_06 the gold answer reached the extractor on only 3 of 20 lookup
tasks. The loss was not chunking (the answer survived segmentation on 8 of 8
answer-bearing documents) but two stacked per-document caps of 3:
PassageEvidenceUnitBuilder.max_units_per_document and
IterativeRetrievalControl._span_role_candidates(max_per_document). An
answer-bearing document routinely yields more than three task-relevant
sentences, and the answer often sat past the third. Raising both caps to 6 moved
end-to-end answer survival from 4 of 8 to 7 of 8; raising only one left it at 4
because the other cap re-truncated.

These tests hold both caps open so a single rich document can contribute more
than three units through both stages.
"""

from __future__ import annotations

import unittest

from tools.search_result_builder.evidence.passage_evidence_unit_builder import (
    PassageEvidenceUnitBuilder,
)
from tools.search_result_builder.retrieval_control import (
    IterativeRetrievalControl,
    RetrievalRoundTrace,
    RetrievedDocumentTrace,
)


# Six short, distinct, task-relevant sentences in one document. The sixth holds
# the answer, so any per-document cap below six drops it.
SIX_SENTENCE_DOCUMENT = (
    "The committee reviewed five nominations in March. "
    "The first candidate withdrew before the vote. "
    "The second candidate was ruled ineligible. "
    "The third candidate deferred to the next cycle. "
    "The fourth candidate lost on a tie-break. "
    "The award went to Wojciech in the final round."
)
ANSWER_SENTENCE_FRAGMENT = "wojciech"


def _document() -> dict[str, object]:
    return {
        "id": "doc-1",
        "title": "Award history",
        "text": SIX_SENTENCE_DOCUMENT,
        "record_type": "passage",
    }


class PassageEvidenceUnitBuilderCapTest(unittest.TestCase):
    def test_default_per_document_cap_is_six(self) -> None:
        self.assertEqual(PassageEvidenceUnitBuilder().max_units_per_document, 6)

    def test_builder_keeps_more_than_three_units_from_one_document(self) -> None:
        builder = PassageEvidenceUnitBuilder()

        result = builder.build(question="Who won the award?", documents=[_document()])

        from_doc = [unit for unit in result.units if unit.document_id == "doc-1"]
        self.assertGreater(len(from_doc), 3)

    def test_answer_sentence_survives_the_builder_cap(self) -> None:
        builder = PassageEvidenceUnitBuilder()

        result = builder.build(question="Who won the award?", documents=[_document()])

        texts = " ".join(unit.text for unit in result.units).casefold()
        self.assertIn(ANSWER_SENTENCE_FRAGMENT, texts)

    def test_a_cap_of_three_would_drop_the_answer(self) -> None:
        """Guard the regression direction: the old cap loses the answer here."""

        builder = PassageEvidenceUnitBuilder(max_units_per_document=3)

        result = builder.build(question="Who won the award?", documents=[_document()])

        texts = " ".join(unit.text for unit in result.units).casefold()
        self.assertNotIn(ANSWER_SENTENCE_FRAGMENT, texts)


class SpanRoleCandidateCapTest(unittest.TestCase):
    """The second stacked cap, inside RetrievalControl._span_role_candidates."""

    def _round_with_one_document(self, spans: list[str]) -> RetrievalRoundTrace:
        trace = RetrievedDocumentTrace(
            document_id="doc-1",
            title="Award history",
            text=SIX_SENTENCE_DOCUMENT,
            url="https://example/doc-1",
            retrieval_score=0.9,
        )
        trace.useful_spans = list(spans)
        round_trace = RetrievalRoundTrace(round_index=0, query="Who won the award?")
        round_trace.documents = [trace]
        return round_trace

    def test_span_role_candidates_pass_more_than_three_from_one_document(self) -> None:
        control = IterativeRetrievalControl.__new__(IterativeRetrievalControl)
        spans = [
            "The committee reviewed five nominations in March",
            "The first candidate withdrew before the vote",
            "The second candidate was ruled ineligible",
            "The third candidate deferred to the next cycle",
            "The fourth candidate lost on a tie-break",
            "The award went to Wojciech in the final round",
        ]
        round_trace = self._round_with_one_document(spans)

        candidates, _ = control._span_role_candidates(round_trace)

        self.assertGreater(len(candidates), 3)
        joined = " ".join(candidate.text for candidate in candidates).casefold()
        self.assertIn(ANSWER_SENTENCE_FRAGMENT, joined)


if __name__ == "__main__":
    unittest.main()
