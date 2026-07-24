from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from benchmark.gaia.gaia_runner import extract_search_summary
from tools.search_result_builder.corpus import (
    DocumentChunker,
    PassagePageIndex,
    WebCorpusBuilder,
    canonicalize_page_url,
)
from tools.search_result_builder.query import RelationPlan
from tools.search_result_builder.retrieval_control import (
    IterativeRetrievalControl,
    RetrievalRoundTrace,
    RetrievedDocumentTrace,
)


def _controller() -> IterativeRetrievalControl:
    return IterativeRetrievalControl(
        retriever=SimpleNamespace(passage_map={}),
        bypass_labeler=True,
        max_iter=1,
    )


def test_corpus_records_preserve_stable_page_metadata() -> None:
    builder = WebCorpusBuilder(
        chunker=DocumentChunker(
            max_chars=120,
            overlap_chars=0,
            min_chars=20,
        )
    )
    records = builder.build_records(
        [
            {
                "title": "Example page",
                "url": (
                    "https://Example.com/article/?utm_source=test"
                    "&edition=full#section"
                ),
                "raw_content": (
                    "First paragraph contains enough useful words to become a "
                    "retrieval passage for this test. "
                    "Second paragraph contains another complete statement for "
                    "the same source page and retrieval task."
                ),
                "content_complete": True,
            }
        ],
        fetch_missing=False,
        max_chunks_per_url=10,
    )

    assert len(records) >= 2
    assert len({record.page_id for record in records}) == 1
    assert [record.passage_index for record in records] == sorted(
        record.passage_index for record in records
    )
    assert all(record.source_url.startswith("https://Example.com") for record in records)
    assert all(
        record.canonical_url == "https://example.com/article?edition=full"
        for record in records
    )
    assert all(record.content_type == "text" for record in records)


def test_page_index_groups_pages_and_sections_in_passage_order() -> None:
    index = PassagePageIndex.build(
        [
            {
                "id": "D2",
                "page_id": "P1",
                "passage_index": 2,
                "section_index": 1,
            },
            {
                "id": "D1",
                "page_id": "P1",
                "passage_index": 1,
                "section_index": 1,
            },
            {
                "id": "D3",
                "url": "https://example.com/other",
                "passage_index": 0,
                "section_index": 0,
            },
        ]
    )

    assert index.passage_ids_by_page["P1"] == ["D1", "D2"]
    assert index.passage_ids_by_section[("P1", 1)] == ["D1", "D2"]
    assert index.page_id_by_passage["D1"] == "P1"
    assert index.page_id_by_passage["D3"].startswith("page-")


def test_uncontracted_or_mismatched_bridge_cannot_trigger_next_hop() -> None:
    controller = _controller()
    trace = RetrievedDocumentTrace(
        document_id="D1",
        title="Page",
        text="KGOT studios are in the Dimond Center.",
        url="https://example.com/page",
        retrieval_score=0.9,
        bridge_spans=["Dimond Center"],
        valid_for_next_hop=True,
    )
    assert controller._grounded_bridge_contracts(trace) == []

    trace.bridge_contracts = [
        {
            "goal_id": "G1",
            "bridge_span": "Dimond Center",
            "context": "KGOT studios are in the Dimond Center.",
            "document_id": "OTHER",
        }
    ]
    assert controller._grounded_bridge_contracts(trace) == []

    trace.bridge_contracts[0]["document_id"] = "D1"
    assert controller._grounded_bridge_contracts(trace) == trace.bridge_contracts


def test_page_trace_keeps_direct_and_bridge_authority_separate() -> None:
    controller = _controller()
    direct = RetrievedDocumentTrace(
        document_id="D1",
        title="Answer page",
        text="The capacity is 0.1777 m3.",
        url="https://example.com/answer",
        canonical_url="https://example.com/answer",
        page_id="P1",
        retrieval_score=0.9,
        direct_contracts=[
            {
                "fact_id": "F1",
                "answer_span": "0.1777 m3",
                "document_id": "D1",
            }
        ],
    )
    bridge = RetrievedDocumentTrace(
        document_id="D2",
        title="Bridge page",
        text="KGOT studios are in the Dimond Center.",
        url="https://example.com/bridge",
        canonical_url="https://example.com/bridge",
        page_id="P2",
        retrieval_score=0.8,
        bridge_contracts=[
            {
                "goal_id": "G1",
                "bridge_span": "Dimond Center",
                "context": "KGOT studios are in the Dimond Center.",
                "document_id": "D2",
            }
        ],
    )
    round_trace = RetrievalRoundTrace(
        round_index=1,
        query="question",
        documents=[direct, bridge],
    )

    controller._refresh_page_traces(round_trace)

    pages = {page.page_id: page for page in round_trace.pages}
    assert pages["P1"].status == "direct_found"
    assert pages["P1"].direct_fact_ids == ["F1"]
    assert pages["P2"].status == "bridge_found"
    assert pages["P2"].bridge_goal_ids == ["G1"]


def test_relation_hop_requires_evidence_id_from_resolved_goal() -> None:
    controller = _controller()
    plan = RelationPlan.from_dict(
        {
            "goals": [
                {
                    "goal_id": "G1",
                    "subject": "KGOT",
                    "relation": "located in",
                    "target": "shopping mall",
                    "state": "resolved",
                    "resolved_values": ["Dimond Center"],
                    "evidence_ids": ["D1"],
                },
                {
                    "goal_id": "G2",
                    "subject": "",
                    "relation": "has size",
                    "target": "area",
                    "state": "active",
                },
            ],
            "active_goal_id": "G2",
        }
    )
    document = RetrievedDocumentTrace(
        document_id="D1",
        title="Page",
        text="KGOT studios are in the Dimond Center.",
        url="",
        retrieval_score=0.9,
    )

    selected = controller._documents_for_resolved_goals(
        plan=plan,
        resolved_goal_ids=["G1"],
        documents=[document],
    )
    assert selected == [document]
    assert (
        controller._documents_for_resolved_goals(
            plan=plan,
            resolved_goal_ids=[],
            documents=[document],
        )
        == []
    )


def test_search_summary_exports_page_and_next_hop_trace() -> None:
    round_payload = {
        "round_index": 1,
        "query": "KGOT studios",
        "pages": [{"page_id": "P1", "status": "bridge_found"}],
        "next_hop_decision": {
            "required": True,
            "decision_reason": "grounded_bridge_next_hop",
            "generated_query": "How large is Dimond Center?",
        },
    }
    summary = extract_search_summary(
        {
            "metadata": {
                "tool_usage": [
                    {
                        "tool_name": "search",
                        "raw_result": {
                            "web_searches": [],
                            "retrieval": {
                                "rounds": [round_payload],
                                "stop_reason": "goal_completion_sufficient",
                            },
                            "diagnostics": {},
                        },
                    }
                ]
            }
        }
    )

    assert summary["page_traces"] == round_payload["pages"]
    assert summary["next_hop_decisions"] == [
        round_payload["next_hop_decision"]
    ]


def test_next_hop_trace_is_dataclass_serializable() -> None:
    controller = _controller()
    trace = RetrievedDocumentTrace(
        document_id="D1",
        title="Page",
        text="KGOT studios are in the Dimond Center.",
        url="",
        retrieval_score=0.9,
        bridge_contracts=[
            {
                "goal_id": "G1",
                "bridge_span": "Dimond Center",
                "context": "KGOT studios are in the Dimond Center.",
                "document_id": "D1",
            }
        ],
    )
    round_trace = RetrievalRoundTrace(round_index=1, query="KGOT")
    controller._record_next_hop_decision(
        round_trace,
        required=True,
        reason="grounded_bridge_next_hop",
        documents=[trace],
        generated_query="How large is Dimond Center?",
    )

    payload = asdict(round_trace)
    assert payload["next_hop_decision"]["required"] is True
    assert payload["next_hop_decision"]["bridge_spans"] == ["Dimond Center"]
    assert canonicalize_page_url("https://example.com/a#x") == "https://example.com/a"
