"""Pin that evidence-only resolution cannot outvote a cross-agent majority.

When every candidate is unsupported, the selector may answer straight from the
fact store. A completed relation plan makes that legitimate -- a multi-hop
target is often a value no Agent nominated -- but the presence of a relation
goal says nothing about whether the fact filling it is right.

level1_final_07 task 9d191bce is the failure: all three Agents answered
'Extremely' and the candidate cleared the validity gate, while retrieval
produced one relation-bound fact reading (Teal'c, response, "Isn't that hot?"),
which is the question quoted back rather than the reply. Evidence-only
resolution replaced the unanimous answer with it. That is the same answer the
attestation majority guard already protects, lost through a second door, so the
same rule applies here: a candidate held by enough runs across enough agents may
be confirmed by evidence but not replaced by it.

The single-run case has to keep working, since that is where deriving an
unnominated multi-hop answer is the whole point.
"""

from __future__ import annotations

from core.config import (
    AgentReasoningSummary,
    AnswerCandidate,
    CandidatePathEvaluation,
    CandidatePathIdentity,
    CandidateRun,
    EachAgentReply,
)
from score.evidence_answer_resolver import EvidenceAnswerResolver
from score.final_winner_selector import FinalWinnerSelector
from tools.evidence.fact_extraction import EvidenceFact, TaskFactStore
from tools.search_result_builder.query import RelationPlan

ANSWER = "Extremely"
EVIDENCE_ANSWER = "Isn't that hot?"
AGENTS = ("nemotron", "qwen", "gemma")


def _plan() -> RelationPlan:
    return RelationPlan.from_dict(
        {
            "goals": [
                {
                    "goal_id": "G1",
                    "subject": "Teal'c",
                    "relation": "response",
                    "target": "reply",
                    "state": "resolved",
                    "resolved_values": [EVIDENCE_ANSWER],
                }
            ],
            "active_goal_id": "",
        }
    )


def _fact() -> EvidenceFact:
    return EvidenceFact(
        fact_id="F-response",
        subject="Teal'c",
        relation="response",
        object=EVIDENCE_ANSWER,
        goal_id="G1",
        role="ANSWER_SUPPORT",
        grounding_status="grounded",
        qualifiers={"answer_binding": "direct"},
        evidence_spans=[f"Teal'c response {EVIDENCE_ANSWER}."],
        context=f"Teal'c response {EVIDENCE_ANSWER}.",
    )


def _evidence() -> dict:
    store = TaskFactStore()
    store.extend([_fact()])
    return {
        "relation_plan": _plan().to_dict(),
        "required_relation": "response",
        "required_relation_goal_id": "G1",
        "answer_role": "reply",
        "fact_store": store.to_dict(),
        "routing": {"use_search": True},
    }


def _stage1(agents: tuple[str, ...], runs_per_agent: int):
    """Build one unsupported candidate held by the given agents and runs."""

    summaries = []
    members = []
    paths = []
    for agent_id in agents:
        replies = []
        for run_index in range(1, runs_per_agent + 1):
            reply = EachAgentReply(
                agent_id=agent_id,
                model_name=f"{agent_id}:test",
                run_index=run_index,
                raw_reply="",
                reasoning=f"step 1. The clip has Teal'c reply {ANSWER}.",
                final_answer=ANSWER,
                parse_completed=True,
                tool_context="",
                schema_valid=True,
                eligible_for_winner=True,
            )
            replies.append(reply)
            members.append(
                CandidateRun(
                    agent_id=agent_id,
                    model_name=f"{agent_id}:test",
                    run_index=run_index,
                    answer=ANSWER,
                    normalized_answer=ANSWER.casefold(),
                    reasoning=reply.reasoning,
                )
            )
            paths.append(
                CandidatePathEvaluation(
                    identity=CandidatePathIdentity(
                        ANSWER.casefold(), agent_id, run_index
                    ),
                    answer=ANSWER,
                    valid=True,
                    eligible_for_winner=True,
                    schema_valid=True,
                    parse_completed=True,
                    reasoning=reply.reasoning,
                    evidence_support_status="no_support",
                    evidence_support_level="unsupported",
                    agent_answer_frequency=runs_per_agent,
                    eligible_run_count=runs_per_agent,
                    agent_confidence=1.0,
                )
            )
        summaries.append(
            AgentReasoningSummary(
                agent_id=agent_id,
                model_name=f"{agent_id}:test",
                runs=replies,
                compressed_answer=ANSWER,
                compressed_reasoning=replies[0].reasoning,
                confidence_score=1.0,
                active=True,
                valid_run_count=runs_per_agent,
                eligible_run_count=runs_per_agent,
            )
        )
    candidate = AnswerCandidate(
        candidate_key=ANSWER.casefold(),
        representative_answer=ANSWER,
        members=members,
    )
    return summaries, [candidate], paths


def _select(selector: FinalWinnerSelector, agents, runs_per_agent):
    stage1, candidates, paths = _stage1(agents, runs_per_agent)
    return selector.select(
        stage1_results=stage1,
        candidates=candidates,
        path_evaluations=paths,
        evidence=_evidence(),
    )


def test_the_fact_alone_would_resolve_to_the_wrong_answer() -> None:
    """Without the guard the resolver is happy to nominate the question text."""

    resolution = EvidenceAnswerResolver().resolve(_evidence())

    assert resolution.resolved
    assert resolution.answer == EVIDENCE_ANSWER


def test_unanimous_candidate_is_not_replaced_by_evidence() -> None:
    selection = _select(FinalWinnerSelector(), AGENTS, 3)

    assert selection.resolved_answer != EVIDENCE_ANSWER
    assert selection.selection_origin != "evidence_only_resolution"
    assert selection.to_dict()["selected_answer"] == ANSWER


def test_a_single_run_candidate_still_yields_to_evidence() -> None:
    """Guard the other direction: multi-hop derivation must keep working."""

    selection = _select(FinalWinnerSelector(), ("qwen",), 1)

    assert selection.resolved_answer == EVIDENCE_ANSWER
    assert selection.selection_origin == "evidence_only_resolution"


def test_guard_can_be_disabled() -> None:
    selection = _select(
        FinalWinnerSelector(evidence_resolution_override_min_runs=0),
        AGENTS,
        3,
    )

    assert selection.resolved_answer == EVIDENCE_ANSWER


def test_one_agent_repeating_itself_is_not_a_majority() -> None:
    """Three runs from a single agent must not count as cross-agent breadth."""

    selection = _select(FinalWinnerSelector(), ("qwen",), 3)

    assert selection.resolved_answer == EVIDENCE_ANSWER
