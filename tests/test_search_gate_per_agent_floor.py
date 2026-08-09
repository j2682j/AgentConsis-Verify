"""Pin that every Agent gets a refinement search before the shared budget bites.

The refinement budget is task-scoped but requested by three Agents across three
runs each, so a purely shared pool is decided by who asks first. On
level1_final_08 it was exhausted on 24 of 53 tasks and gemma was refused 93 times
while being granted none -- and because each refusal costs one of a run's few
tool turns, losing them all is what pushed gemma into the tool-less repair prompt
on 100% of its runs.

A floor grant is reserved per Agent and does not draw on the shared pool, so the
task ceiling becomes `floor * agents + budget` rather than `budget`. Setting the
floor to 0 restores the previous purely shared behaviour.
"""

from __future__ import annotations

import unittest

from core.stage1_search_gate import Stage1SearchAccessState


def _state(*, budget: int = 2, floor: int = 1) -> Stage1SearchAccessState:
    state = Stage1SearchAccessState(
        prepared_status="prepared_usable",
        prepared_evidence_available=True,
        refinement_budget=budget,
        per_agent_refinement_floor=floor,
    )
    state._prepared_query_keys = set()
    return state


def _ask(state, agent_id: str, query: str):
    return state.authorize(
        query=query, missing_information="one specific fact", agent_id=agent_id
    )


class PerAgentRefinementFloorTest(unittest.TestCase):
    def test_each_agent_gets_its_first_search_even_after_others_asked(self) -> None:
        state = _state()
        for index in range(4):
            _ask(state, "gemma", f"gemma-{index}")

        for agent in ("qwen", "nemotron"):
            decision = _ask(state, agent, f"{agent}-first")
            self.assertTrue(decision.allowed, agent)
            self.assertEqual(decision.reason, "refinement_allowed_by_floor", agent)

    def test_a_floor_grant_does_not_consume_the_shared_pool(self) -> None:
        state = _state()

        _ask(state, "gemma", "g1")
        _ask(state, "qwen", "q1")
        _ask(state, "nemotron", "n1")

        self.assertEqual(state.shared_refinement_used, 0)
        self.assertEqual(state.refinement_used, 3)

    def test_the_task_ceiling_is_floor_times_agents_plus_budget(self) -> None:
        state = _state(budget=2, floor=1)
        granted = 0
        for index in range(4):
            for agent in ("gemma", "qwen", "nemotron"):
                if _ask(state, agent, f"{agent}-{index}").allowed:
                    granted += 1

        self.assertEqual(granted, 1 * 3 + 2)

    def test_beyond_floor_and_budget_the_block_reason_is_unchanged(self) -> None:
        state = _state(budget=0, floor=1)

        _ask(state, "gemma", "g1")
        decision = _ask(state, "gemma", "g2")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "refinement_budget_exhausted")

    def test_floor_zero_restores_a_purely_shared_budget(self) -> None:
        state = _state(budget=2, floor=0)
        granted = sum(
            1
            for index, agent in enumerate(("gemma", "qwen", "nemotron", "gemma"))
            if _ask(state, agent, f"q{index}").allowed
        )

        self.assertEqual(granted, 2)

    def test_an_unidentified_caller_draws_only_on_the_shared_pool(self) -> None:
        """No agent id means no reserved floor to claim."""

        state = _state(budget=1, floor=1)

        self.assertTrue(_ask(state, "", "a").allowed)
        self.assertFalse(_ask(state, "", "b").allowed)

    def test_snapshot_reports_the_split(self) -> None:
        state = _state()
        _ask(state, "gemma", "g1")
        _ask(state, "gemma", "g2")

        snapshot = state.snapshot()

        self.assertEqual(snapshot["per_agent_refinement_floor"], 1)
        self.assertEqual(snapshot["shared_refinement_used"], 1)
        self.assertEqual(snapshot["agent_refinement_used"], {"gemma": 2})
        self.assertEqual(snapshot["refinement_remaining"], 1)


if __name__ == "__main__":
    unittest.main()
