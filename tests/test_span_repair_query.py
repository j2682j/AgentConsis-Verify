from __future__ import annotations

from types import SimpleNamespace
import unittest

from tools.search_result_builder.query.semantic_impact import TokenSalient
from tools.search_result_builder.query.span_repair import SalientSpan, SpanRepairer


class FakeScorer:
    def score_span_impacts(self, question, spans):
        return [0.123 + index * 0.01 for index, _span in enumerate(spans)]


def token(text: str, question: str, *, score: float = 0.1, index: int = 0) -> TokenSalient:
    start = question.index(text)
    end = start + len(text)
    return TokenSalient(
        token_index=index,
        token=text,
        text=text,
        start=start,
        end=end,
        embedding_similarity=0.9,
        embedding_delta=score,
        score=score,
        keep=True,
        reason="test",
    )


class SpanRepairerQueryTests(unittest.TestCase):
    def test_ner_repair_expands_salient_fragment_to_complete_entity(self):
        question = (
            "There is a fish popularized by Finding Nemo. According to the USGS, "
            "where was this fish found?"
        )
        start = question.index("Nemo")
        end = question.index("According") + len("According")
        repairer = SpanRepairer()
        repairer._spacy_doc = lambda _question: SimpleNamespace(
            ents=[
                SimpleNamespace(
                    start_char=question.index("Finding Nemo"),
                    end_char=question.index("Finding Nemo") + len("Finding Nemo"),
                    label_="WORK_OF_ART",
                ),
                SimpleNamespace(
                    start_char=question.index("USGS"),
                    end_char=question.index("USGS") + len("USGS"),
                    label_="ORG",
                ),
            ],
            noun_chunks=[],
        )
        span = SalientSpan(
            text="Nemo. According",
            start=start,
            end=end,
            score=0.5,
            tokens=["Nemo"],
            token_indices=[1],
        )

        repaired = repairer.repair_spans(question, [span], scorer=FakeScorer())

        self.assertEqual(repaired[0].text, "Finding Nemo")
        self.assertIn("ner_entity", repaired[0].repair_source)
        self.assertEqual(repaired[0].original_text, "Nemo. According")
        self.assertAlmostEqual(repaired[0].score, 0.123, places=3)

    def test_boundary_cleanup_does_not_keep_instruction_tail_across_sentence(self):
        question = "What are the EC numbers in the Pearl Of Africa from 2016? Return semicolon-separated numbers."
        start = question.index("2016")
        end = question.index("Return") + len("Return")
        repairer = SpanRepairer()
        repairer._spacy_doc = lambda _question: None
        span = SalientSpan(
            text="2016? Return",
            start=start,
            end=end,
            score=0.4,
            tokens=["2016"],
            token_indices=[2],
        )

        repaired = repairer.repair_spans(question, [span])

        self.assertEqual(repaired[0].text, "2016")
        self.assertEqual(repaired[0].repair_source, "boundary_cleanup")

    def test_build_spans_rescores_repaired_units(self):
        question = "Nature articles in 2020 had an average p-value of 0.04."
        repairer = SpanRepairer(max_salient_spans=3)
        repairer._spacy_doc = lambda _question: None

        spans = repairer.build_spans(
            question,
            [token("Nature", question, score=0.5), token("2020", question, score=0.4, index=1)],
            scorer=FakeScorer(),
        )

        self.assertTrue(spans)
        self.assertEqual(spans[0].rescore, spans[0].score)
        self.assertGreater(spans[0].score, 0)

    def test_domain_phrase_keeps_pdb_id_with_identifier(self):
        question = "Parse the PDB file of the protein identified by the PDB ID 5wb7 from RCSB."
        repairer = SpanRepairer(max_salient_spans=3)
        repairer._spacy_doc = lambda _question: None

        spans = repairer.build_spans(question, [token("5wb7", question, score=0.5)])

        self.assertTrue(any("5wb7" in span.text for span in spans))

    def test_repair_does_not_drop_numeric_or_dotted_identifiers(self):
        question = "Parse the PDB file identified by the PDB ID 5wb7 from RCSB."
        start = question.index("PDB ID")
        end = question.index("5wb7") + len("5wb7")
        repairer = SpanRepairer()
        repairer._spacy_doc = lambda _question: SimpleNamespace(
            ents=[],
            noun_chunks=[
                SimpleNamespace(
                    start_char=question.index("PDB ID"),
                    end_char=question.index("PDB ID") + len("PDB ID"),
                )
            ],
        )
        span = SalientSpan(
            text="PDB ID 5wb7",
            start=start,
            end=end,
            score=0.5,
            tokens=["PDB", "ID", "5wb7"],
            token_indices=[1, 2, 3],
        )

        repaired = repairer.repair_spans(question, [span])

        self.assertEqual(repaired[0].text, "PDB ID 5wb7")

    def test_repair_does_not_replace_museum_number_with_generic_chunk(self):
        question = "The museum number 2012,5015.17 is the shell of a mollusk species."
        start = question.index("2012,5015.17")
        end = start + len("2012,5015.17")
        repairer = SpanRepairer()
        repairer._spacy_doc = lambda _question: SimpleNamespace(
            ents=[],
            noun_chunks=[
                SimpleNamespace(
                    start_char=question.index("shell"),
                    end_char=question.index("shell") + len("shell"),
                )
            ],
        )
        span = SalientSpan(
            text="2012,5015.17",
            start=start,
            end=end,
            score=0.5,
            tokens=["2012,5015.17"],
            token_indices=[1],
        )

        repaired = repairer.repair_spans(question, [span])

        self.assertEqual(repaired[0].text, "2012,5015.17")


if __name__ == "__main__":
    unittest.main()
