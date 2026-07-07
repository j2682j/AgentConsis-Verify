from __future__ import annotations

import unittest

from core.config import AgentConfig, EachAgentReply
from core.network import Network
from score.stage1_aggregator import Stage1Aggregator


def reply(
    *,
    agent_id: str,
    answer: str,
    eligible: bool,
    labels: list[str] | None = None,
    schema_valid: bool = True,
    run_index: int = 1,
) -> EachAgentReply:
    return EachAgentReply(
        agent_id=agent_id,
        model_name="test-model",
        run_index=run_index,
        raw_reply=answer,
        reasoning="step 1. test",
        final_answer=answer,
        parse_completed=eligible,
        tool_context="",
        schema_valid=schema_valid,
        eligible_for_winner=eligible,
        validity_labels=list(labels or []),
    )


class AbstentionWinnerSelectionTests(unittest.TestCase):
    def test_all_unknown_runs_are_not_active(self):
        config = AgentConfig(agent_id="a1", model_name="test-model")
        summary = Stage1Aggregator().summarize(
            config,
            [
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=1),
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=2),
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=3),
            ],
        )

        self.assertFalse(summary.active)
        self.assertFalse(summary.winner_selection_eligible)
        self.assertEqual(summary.winner_selection_status, "all_runs_abstained")
        self.assertEqual(summary.abstention_run_count, 3)

    def test_single_valid_answer_can_win_over_abstention(self):
        aggregator = Stage1Aggregator()
        abstain_config = AgentConfig(agent_id="a1", model_name="test-model")
        answer_config = AgentConfig(agent_id="a2", model_name="test-model")
        abstain = aggregator.summarize(
            abstain_config,
            [
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=1),
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=2),
                reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=3),
            ],
        )
        answer = aggregator.summarize(
            answer_config,
            [
                reply(agent_id="a2", answer="42", eligible=True, run_index=1),
                reply(agent_id="a2", answer="", eligible=False, labels=["empty_final_answer"], run_index=2),
                reply(agent_id="a2", answer="", eligible=False, labels=["empty_final_answer"], run_index=3),
            ],
        )
        abstain_config.total_score = 10.0
        answer_config.total_score = 0.33

        network = Network("test question", [abstain_config, answer_config])
        winner = network._select_winner([abstain, answer])
        metadata = network._winner_selection_metadata([abstain, answer], winner)

        self.assertIsNotNone(winner)
        self.assertEqual(winner.agent_id, "a2")
        self.assertEqual(answer.winner_selection_status, "mixed_low_coverage")
        self.assertEqual(metadata["answerable_agent_count"], 1)
        self.assertEqual(metadata["abstained_agent_count"], 1)

    def test_direct_consensus_ignores_abstention(self):
        aggregator = Stage1Aggregator()
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        summaries = [
            aggregator.summarize(
                configs[0],
                [
                    reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=1),
                    reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=2),
                    reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=3),
                ],
            ),
            aggregator.summarize(
                configs[1],
                [
                    reply(agent_id="a2", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=1),
                    reply(agent_id="a2", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=2),
                    reply(agent_id="a2", answer="unknown", eligible=False, labels=["refusal_like_final_answer"], run_index=3),
                ],
            ),
        ]

        winner, support = Network("test question", configs)._confidence_one_answer_consensus(summaries)

        self.assertIsNone(winner)
        self.assertEqual(support, [])

    def test_all_agents_abstained_or_invalid_metadata(self):
        aggregator = Stage1Aggregator()
        configs = [
            AgentConfig(agent_id="a1", model_name="test-model"),
            AgentConfig(agent_id="a2", model_name="test-model"),
        ]
        summaries = [
            aggregator.summarize(
                configs[0],
                [reply(agent_id="a1", answer="unknown", eligible=False, labels=["refusal_like_final_answer"])],
            ),
            aggregator.summarize(
                configs[1],
                [reply(agent_id="a2", answer="", eligible=False, labels=["invalid_final_answer"], schema_valid=False)],
            ),
        ]
        network = Network("test question", configs)

        winner = network._select_winner(summaries)
        metadata = network._winner_selection_metadata(summaries, winner)

        self.assertIsNone(winner)
        self.assertEqual(metadata["status"], "all_agents_abstained_or_invalid")
        self.assertEqual(metadata["answerable_agent_count"], 0)


if __name__ == "__main__":
    unittest.main()
