from __future__ import annotations

from tools.evidence.fact_extraction import (
    EvidenceFact,
    SemanticExtractionResult,
    TaskFactStore,
)
from tools.search_result_builder.candidate_verification import (
    CandidateVerificationSearcher,
)
from core.config import AgentConfig, CandidatePathEvaluation, CandidatePathIdentity
from core.network import Network


class FakeFactExtractor:
    max_units_per_call = 8

    def extract_batch(self, *, units, **kwargs):
        facts = []
        for unit in units:
            candidate = str(unit.metadata.get("candidate_answer") or "")
            if candidate != "Paris":
                continue
            facts.append(
                EvidenceFact(
                    fact_id=f"fact-{unit.unit_id}",
                    subject="France",
                    relation="capital",
                    object="Paris",
                    qualifiers={"answer_binding": "direct"},
                    role="ANSWER_SUPPORT",
                    evidence_spans=["Paris is the capital of France."],
                    context=unit.text,
                    source_id=unit.source_id,
                    source_type=unit.source_type,
                    grounding_status="grounded",
                    extraction_method="semantic_model",
                )
            )
        return SemanticExtractionResult(
            facts=facts,
            diagnostics={"success": True, "fact_count": len(facts)},
        )


def fake_search(query: str, max_results: int):
    answer = "Paris" if '"Paris"' in query else "Lyon"
    return {
        "raw_result": {
            "results": [
                {
                    "title": f"Reference for {answer}",
                    "url": f"https://example.org/{answer.lower()}",
                    "content": (
                        "Paris is the capital of France according to the "
                        "official national reference page."
                        if answer == "Paris"
                        else "Lyon is a major city in France with a long history."
                    ),
                }
            ][:max_results]
        }
    }


def test_candidate_recovery_uses_grounded_fact_verification() -> None:
    store = TaskFactStore()
    searcher = CandidateVerificationSearcher(
        search_executor=fake_search,
        semantic_fact_extractor=FakeFactExtractor(),
        max_candidates=5,
        max_results_per_candidate=3,
        max_workers=2,
    )

    result = searcher.verify(
        question="What is the capital of France?",
        candidate_answers=["Lyon", "Paris"],
        fact_store=store,
        answer_requirement="What is the capital of France?",
        answer_role="place",
    )

    by_answer = {trace.candidate_answer: trace for trace in result.traces}
    assert result.attempted
    assert result.added_fact_count == 1
    assert by_answer["Paris"].status == "supported"
    assert by_answer["Paris"].supporting_fact_ids
    assert by_answer["Lyon"].status == "unresolved"


def test_candidate_recovery_is_bounded_and_deduplicated() -> None:
    calls = []

    def recording_search(query: str, max_results: int):
        calls.append(query)
        return {"raw_result": {"results": []}}

    result = CandidateVerificationSearcher(
        search_executor=recording_search,
        semantic_fact_extractor=FakeFactExtractor(),
        max_candidates=2,
    ).verify(
        question="Which answer is correct?",
        candidate_answers=["A", "A", "B", "C"],
        fact_store=TaskFactStore(),
    )

    assert result.attempted
    assert len(result.traces) == 2
    assert len(calls) == 2


def test_recovery_does_not_trigger_when_any_candidate_has_support() -> None:
    network = Network(
        "What is the capital of France?",
        [AgentConfig(agent_id="a1", model_name="test")],
        tool_manager=object(),
    )
    supported = CandidatePathEvaluation(
        identity=CandidatePathIdentity("paris", "a1", 1, 0),
        answer="Paris",
        valid=True,
        eligible_for_winner=True,
        evidence_support_level="direct_evidence",
    )
    unsupported = CandidatePathEvaluation(
        identity=CandidatePathIdentity("lyon", "a1", 2, 0),
        answer="Lyon",
        valid=True,
        eligible_for_winner=True,
        evidence_support_level="unsupported",
    )

    assert not network._should_run_candidate_verification(
        [supported, unsupported],
        evidence={"routing": {"primary_route": "factual_search"}},
    )


def test_recovery_does_not_trigger_for_closed_world_task() -> None:
    network = Network(
        "Compute 2 + 2.",
        [AgentConfig(agent_id="a1", model_name="test")],
        tool_manager=object(),
    )
    unsupported = CandidatePathEvaluation(
        identity=CandidatePathIdentity("4", "a1", 1, 0),
        answer="4",
        valid=True,
        eligible_for_winner=True,
        evidence_support_level="unsupported",
    )

    assert not network._should_run_candidate_verification(
        [unsupported],
        evidence={"routing": {"primary_route": "deterministic"}},
    )
