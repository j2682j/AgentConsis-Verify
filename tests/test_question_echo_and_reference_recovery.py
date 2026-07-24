from __future__ import annotations

"""Regression tests for the question-echo guard and reference recovery.

Covers the two dominant level1 error patterns:
1. Question-echo answers (a quoted title, or a fragment of a reversed-text
   puzzle) must not be promoted to evidence support by finding themselves in
   retrieved text — while trusted deterministic handler finals stay exempt.
2. Best-effort references must keep collection tables whole (sibling rows
   merged, not capped per-domain) so count/aggregate questions see every row,
   and the versa gate must use corpus mentions as a final tie-break.
"""

import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network
from score.evidence_support_checker import EvidenceSupportChecker
from score.question_echo import is_question_echo
from tools.search_result_builder.evidence.best_effort_reference import (
    BestEffortReferenceSelector,
)


def run(agent_id: str, run_index: int, answer: str, reasoning: str) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=run_index,
        raw_reply="",
        reasoning=reasoning,
        final_answer=answer,
        parse_completed=True,
        tool_context="",
        schema_valid=True,
        eligible_for_winner=True,
    )


def summary(agent_id: str, answer: str, reasoning: str) -> AgentReasoningSummary:
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=[run(agent_id, 1, answer, reasoning)],
        compressed_answer=answer,
        compressed_reasoning=reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=1,
        eligible_run_count=1,
    )


class IsQuestionEchoTests(unittest.TestCase):
    def test_quoted_title_is_echo(self) -> None:
        question = (
            'Of the authors that worked on the paper "Pie Menus or Linear '
            'Menus, Which Is Better?" in 2015, what was the title of the '
            "first paper authored by the one that had authored prior papers?"
        )
        self.assertTrue(
            is_question_echo("Pie Menus or Linear Menus, Which Is Better?", question)
        )

    def test_reversed_text_fragment_is_echo(self) -> None:
        question = (
            '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw '
            ",ecnetnes siht dnatsrednu uoy fI"
        )
        self.assertTrue(is_question_echo("etisoppo eht etirw", question))

    def test_short_answers_are_exempt(self) -> None:
        question = "Write the opposite of the word left as the answer."
        self.assertFalse(is_question_echo("left", question))
        self.assertFalse(is_question_echo("3", "Is 3 the count of albums?"))

    def test_unrelated_answer_is_not_echo(self) -> None:
        self.assertFalse(
            is_question_echo("Mapping Human Oriented Information", "Which paper?")
        )


class EchoSupportGuardTests(unittest.TestCase):
    _QUESTION = (
        'Of the authors that worked on the paper "Pie Menus or Linear Menus, '
        'Which Is Better?" in 2015, what was the title of the first paper '
        "authored by the one that had authored prior papers?"
    )

    def _search_evidence(self, text: str) -> dict:
        return {
            "tool_usage": [
                {
                    "tool_name": "search",
                    "ok": True,
                    "evidence_valid": True,
                    "output_text": text,
                }
            ]
        }

    def test_echo_answer_is_not_promoted_by_search_text(self) -> None:
        answer = "Pie Menus or Linear Menus, Which Is Better?"
        target = summary("a1", answer, f"step 1. The paper is {answer}")

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, f"The paper is {answer}")],
            evidence=self._search_evidence(
                "Murano, P. published Pie Menus or Linear Menus, Which Is "
                "Better? in the Journal of Emerging Trends in Computing."
            ),
            question=self._QUESTION,
        )

        self.assertEqual(support.status, "no_support")
        self.assertTrue(support.metadata.get("question_echo"))

    def test_non_echo_answer_keeps_search_support(self) -> None:
        answer = "Paris"
        target = summary("a1", answer, f"step 1. The capital is {answer}")

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, f"The capital is {answer}")],
            evidence={
                "tool_usage": [
                    {
                        "tool_name": "search",
                        "ok": True,
                        "raw_result": {
                            "evidence_items": [
                                {
                                    "evidence_id": "E1",
                                    "title": "France",
                                    "text": "Paris is the capital of France.",
                                    "direct_contracts": [
                                        {
                                            "goal_id": "G1",
                                            "answer_span": "Paris",
                                            "context": "Paris is the capital of France.",
                                            "answer_requirement": "the capital of France",
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ]
            },
            question="What is the capital of France?",
        )

        self.assertEqual(support.status, "search_evidence_supported")

    def test_echo_answer_keeps_trusted_tool_final_support(self) -> None:
        question = (
            "Which of the following is equivalent: (A → B) ↔ (¬B → ¬A) or "
            "¬(A ∨ B) ↔ (¬A ∧ ¬B)?"
        )
        answer = "(A → B) ↔ (¬B → ¬A)"
        target = summary("a1", answer, f"step 1. The handler returned {answer}")

        support = EvidenceSupportChecker().check_agent(
            target=target,
            reasoning_steps=[(1, f"The handler returned {answer}")],
            evidence={
                "tool_usage": [
                    {
                        "tool_name": "deterministic_handler_router",
                        "ok": True,
                        "evidence_valid": True,
                        "trusted": True,
                        "output_type": "final_answer",
                        "output_text": answer,
                        "handler_trust": {"trusted": True},
                        "raw_result": {
                            "output_type": "final_answer",
                            "final_answer": answer,
                        },
                    }
                ]
            },
            question=question,
        )

        self.assertEqual(support.status, "tool_final_supported")


class CollectionRowMergeTests(unittest.TestCase):
    def _row(self, year: str, album: str, score: float) -> dict:
        return {
            "document_id": f"row-{year}",
            "title": year,
            "text": (
                f"Record Type: database_row Title: {year} Date: {year} "
                f"Source: Mercedes Sosa - Wikipedia Album details: {album}"
            ),
            "url": "https://en.wikipedia.org/wiki/Mercedes_Sosa",
            "record_type": "database_row",
            "record_fields": {"parent_url": "https://en.wikipedia.org/wiki/Mercedes_Sosa"},
            "retrieval_score": score,
        }

    def test_sibling_rows_merge_into_one_complete_reference(self) -> None:
        rows = [
            self._row("2000", "Misa Criolla", 0.86),
            self._row("2003", "Argentina Quiere Cantar", 0.85),
            self._row("2005", "Corazon Libre", 0.84),
            self._row("2009", "Cantora 1", 0.83),
            self._row("2011", "Deja La Vida Volar", 0.82),
        ]
        output = {"retrieval": {"rounds": [{"round_index": 1, "documents": rows}]}}

        references = BestEffortReferenceSelector().select(output)

        merged = [item for item in references if item.source_type == "collection_rows"]
        self.assertEqual(len(merged), 1)
        for year in ("2000", "2003", "2005", "2009", "2011"):
            self.assertIn(year, merged[0].text)

    def test_regular_passages_still_respect_domain_cap(self) -> None:
        passages = [
            {
                "document_id": f"D{index}",
                "title": f"Passage {index}",
                "text": (
                    f"Passage number {index} discussing a distinct topic in "
                    f"enough detail to pass the minimum length filter easily."
                ),
                "url": f"https://example.com/page-{index}",
                "record_type": "passage",
                "retrieval_score": 0.9 - index * 0.01,
            }
            for index in range(1, 5)
        ]
        output = {"retrieval": {"rounds": [{"round_index": 1, "documents": passages}]}}

        references = BestEffortReferenceSelector(max_items_per_domain=2).select(output)

        self.assertEqual(len(references), 2)


class CorpusAttestationTests(unittest.TestCase):
    """An answer the fetched pages never state loses to one they do.

    Ranking candidates by mention count does not work — the topic of the
    question is repeated on every relevant page and outnumbers the answer.
    Absence is the usable signal, and it is checked before the vote-based
    gates because agents sharing a context agree on invented answers often
    enough that consensus would otherwise confirm them.
    """

    def _evidence(self, text: str) -> dict:
        return {
            "routing": {"primary_route": "factual_search"},
            "tool_usage": [
                {
                    "tool_name": "search",
                    "raw_result": {
                        "retrieval": {
                            "rounds": [
                                {"round_index": 1, "documents": [{"text": text}]}
                            ]
                        }
                    },
                }
            ],
        }

    def _select(self, network, results, evidence):
        return network.final_winner_selector.select(
            stage1_results=results,
            candidates=network.answer_candidate_clusterer.cluster(results),
            verifier_results=[],
            evidence=evidence,
        )

    def test_unattested_answer_loses_to_an_attested_one(self) -> None:
        # The invented answer has the stronger vote: two agents against one.
        results = [
            summary("a1", "FunkMonk", "step 1. The nominator was FunkMonk."),
            summary("a2", "Ian Rose", "step 1. The nominator was Ian Rose."),
            summary("a3", "Ian Rose", "step 1. The nominator was Ian Rose."),
        ]
        network = Network(
            "Who nominated the featured article promoted in November 2016?",
            [AgentConfig(agent_id=f"a{i}", model_name="test-model") for i in (1, 2, 3)],
        )
        evidence = self._evidence(
            "The nomination was made by FunkMonk. FunkMonk opened the review, "
            "and FunkMonk answered the comments during the FunkMonk nomination."
        )

        selection = self._select(network, results, evidence)

        self.assertIsNotNone(selection.winner)
        self.assertEqual(selection.winner.compressed_answer, "FunkMonk")
        gate = next(
            item for item in selection.gate_trace
            if item.gate_name == "corpus_attestation"
        )
        self.assertEqual([item.answer for item in gate.survivors], ["FunkMonk"])

    def test_both_attested_leaves_the_decision_to_later_gates(self) -> None:
        """The gate must not rank by frequency: a topic word beats the answer."""
        results = [
            summary("a1", "THE CASTLE", "step 1. The location is the castle."),
            summary("a2", "Heaven Sent", "step 1. The location is Heaven Sent."),
        ]
        network = Network(
            "In Series 9 Episode 11 of Doctor Who, what is the location called?",
            [AgentConfig(agent_id="a1", model_name="test-model"),
             AgentConfig(agent_id="a2", model_name="test-model")],
        )
        evidence = self._evidence(
            "Heaven Sent is the episode. Heaven Sent aired in 2015. In Heaven "
            "Sent the Doctor is trapped. THE CASTLE is the setting."
        )

        selection = self._select(network, results, evidence)

        gate = next(
            item for item in selection.gate_trace
            if item.gate_name == "corpus_attestation"
        )
        self.assertEqual(len(gate.survivors), 2)

    def test_single_character_answers_are_not_counted(self) -> None:
        """A one-character answer matches too much text to judge by absence."""
        results = [
            summary("a1", "3", "step 1. The count is 3."),
            summary("a2", "9", "step 1. The count is 9."),
        ]
        network = Network("How many studio albums between 2000 and 2009?",
                          [AgentConfig(agent_id="a1", model_name="test-model"),
                           AgentConfig(agent_id="a2", model_name="test-model")])
        evidence = self._evidence("She released 3 studio albums between 2000 and 2009.")

        selection = self._select(network, results, evidence)

        gate = next(
            item for item in selection.gate_trace
            if item.gate_name == "corpus_attestation"
        )
        self.assertEqual(len(gate.survivors), 2)


if __name__ == "__main__":
    unittest.main()
