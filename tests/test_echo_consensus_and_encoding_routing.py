from __future__ import annotations

"""Regression tests for the three fixes targeting tasks like L1 #7/#9.

1. Cross-agent consensus gate: a question-echo candidate must not win the
   head-count against a non-echo candidate; trusted tool finals stay exempt,
   and an all-echo field keeps the original behavior.
2. Reversed-text routing: a fully reversed question routes deterministic-
   first with search forbidden, and EvidenceRunner supplies the decoded text
   as trusted intermediate context. Ordinary questions never trigger.
3. Document-type directives: an "official script"-style question adds one
   targeted query variant with a pdf_text preference; questions without a
   directive keep their query plan byte-identical.
"""

import unittest

from core.config import (
    AgentConfig,
    AgentReasoningSummary,
    EachAgentReply,
    VerifierScoreByReasoning,
)
from core.network import Network
from tools.search_result_builder.query.document_type_directive import (
    detect_document_type_directive,
)
from tools.system_routing_contract import SystemRoutingContract

REVERSED_QUESTION = (
    '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw '
    ",ecnetnes siht dnatsrednu uoy fI"
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


def summary(
    agent_id: str,
    answer: str,
    *,
    runs_count: int = 1,
) -> AgentReasoningSummary:
    reasoning = f"step 1. The answer is {answer}."
    return AgentReasoningSummary(
        agent_id=agent_id,
        model_name="test-model",
        runs=[run(agent_id, index, answer, reasoning) for index in range(1, runs_count + 1)],
        compressed_answer=answer,
        compressed_reasoning=reasoning,
        confidence_score=1.0,
        active=True,
        valid_run_count=runs_count,
        eligible_run_count=runs_count,
    )


def verifier(agent_id: str, candidate_key: str, status: str = "no_support"):
    return VerifierScoreByReasoning(
        verifier_id="versa_prm",
        target_agent_id=agent_id,
        verifier_score=0.9,
        metadata={
            "candidate_key": candidate_key,
            "target_run_index": 1,
            "evidence_support": {"status": status, "priority": 1},
            "process_verification": {
                "critical_step_floor": 0.9,
                "critical_step_geometric_mean": 0.9,
                "average_probability": 0.9,
            },
        },
    )


class EchoConsensusGateTests(unittest.TestCase):
    def test_non_echo_minority_beats_echo_majority(self) -> None:
        configs = [
            AgentConfig(agent_id=f"a{index}", model_name="test-model")
            for index in (1, 2, 3)
        ]
        junk = "etisoppo eht etirw"
        results = [
            summary("a1", junk),
            summary("a2", junk),
            summary("a3", "right"),
        ]
        network = Network(REVERSED_QUESTION, configs)
        candidates = network.answer_candidate_clusterer.cluster(results)
        verifier_results = [
            verifier("a1", junk),
            verifier("a2", junk),
            verifier("a3", "right"),
        ]

        selection = network.final_winner_selector.select(
            stage1_results=results,
            candidates=candidates,
            verifier_results=verifier_results,
            evidence={},
        )

        self.assertIsNotNone(selection.winner)
        self.assertEqual(selection.winner.compressed_answer, "right")
        consensus_gate = next(
            item
            for item in selection.gate_trace
            if item.gate_name == "cross_agent_consensus"
        )
        self.assertIn(
            "etisoppo eht etirw",
            [key.lower() for key in consensus_gate.metadata.get("echo_candidates_deferred", [])],
        )

    def test_all_echo_field_keeps_head_count_behavior(self) -> None:
        configs = [
            AgentConfig(agent_id=f"a{index}", model_name="test-model")
            for index in (1, 2, 3)
        ]
        junk_major = "etisoppo eht etirw"
        junk_minor = "drow eht fo etisoppo"
        results = [
            summary("a1", junk_major),
            summary("a2", junk_major),
            summary("a3", junk_minor),
        ]
        network = Network(REVERSED_QUESTION, configs)
        candidates = network.answer_candidate_clusterer.cluster(results)
        verifier_results = [
            verifier("a1", junk_major),
            verifier("a2", junk_major),
            verifier("a3", junk_minor),
        ]

        selection = network.final_winner_selector.select(
            stage1_results=results,
            candidates=candidates,
            verifier_results=verifier_results,
            evidence={},
        )

        self.assertIsNotNone(selection.winner)
        self.assertEqual(selection.winner.compressed_answer, junk_major)


class ReversedTextRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = SystemRoutingContract()

    def test_reversed_question_routes_deterministic_and_forbids_search(self) -> None:
        decision = self.contract.route(
            question=REVERSED_QUESTION,
            stage="stage1_round0",
        )

        self.assertEqual(decision.question_encoding, "reversed")
        self.assertEqual(decision.initial_route, "deterministic_first")
        self.assertEqual(decision.search_policy, "forbidden")
        self.assertIn("reversed_text", decision.trigger_terms)

    def test_ordinary_question_does_not_trigger(self) -> None:
        decision = self.contract.route(
            question=(
                "How many studio albums were published by Mercedes Sosa "
                "between 2000 and 2009?"
            ),
            stage="stage1_round0",
        )

        self.assertEqual(decision.question_encoding, "")

    def test_attachment_question_never_triggers(self) -> None:
        decision = self.contract.route(
            question=REVERSED_QUESTION,
            stage="stage1_round0",
            has_attachment=True,
        )

        self.assertEqual(decision.question_encoding, "")

    def test_evidence_runner_decodes_reversed_question(self) -> None:
        from core.evidence_runner import EvidenceRunner

        runner = EvidenceRunner(question=REVERSED_QUESTION)
        context, usage = runner._decode_reversed_question()

        self.assertIn("write the opposite of the word", context)
        self.assertEqual(usage[0]["tool_name"], "reversed_text_decoder")
        self.assertEqual(usage[0]["output_type"], "intermediate_value")
        self.assertTrue(usage[0]["trusted"])


class DocumentTypeDirectiveTests(unittest.TestCase):
    def test_official_script_maps_to_pdf_preference(self) -> None:
        directive = detect_document_type_directive(
            "What is this location called in the official script for the "
            "episode? Give the setting exactly as it appears in the first "
            "scene heading."
        )

        self.assertIsNotNone(directive)
        self.assertEqual(directive.required_content, "pdf_text")
        self.assertIn("script", directive.type_terms)

    def test_bare_pdf_mention_does_not_trigger(self) -> None:
        self.assertIsNone(
            detect_document_type_directive(
                "How many applicants for the job in the PDF are only missing "
                "a single qualification?"
            )
        )

    def test_plain_question_does_not_trigger(self) -> None:
        self.assertIsNone(
            detect_document_type_directive("What is the capital of France?")
        )


if __name__ == "__main__":
    unittest.main()
