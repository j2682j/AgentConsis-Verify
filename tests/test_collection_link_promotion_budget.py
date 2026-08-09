"""Pin how many stub links get opened per task.

A structured corpus record carries a title and a `content_url` but no page body,
so the page behind it is only read if `max_collection_links_to_fetch` reaches it.
On level1_final_06 that budget was 3 against 567 candidates (7.4%), and six tasks
promoted nothing. Replaying the run's candidates in rank order and fetching them,
the link holding the answer sat at ranks 1, 2, 5, 8 and 19 across the five tasks
where such a link existed -- so 3 opened two of them and 10 opens four.

These tests drive the selection with more stubs than the budget, which is the
shape production has and the shape a single-stub test cannot reach.
"""

from __future__ import annotations

import unittest
from typing import Any

from tools.search_result_builder.retrieval_control import WebRetrievalControl

STUB_COUNT = 30


def _stub(index: int) -> dict[str, Any]:
    return {
        "id": f"record-{index:03d}",
        "record_id": f"record-{index:03d}",
        "record_type": "article",
        "title": f"Entry {index}",
        "text": f"Record Type: article\nTitle: Entry {index}",
        "content_url": f"https://example.org/entry/{index}",
        "parent_url": "https://example.org/index",
    }


class _Retriever:
    """Rank stubs in id order, which is all the selection needs to be tested."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.passage_map = {doc["id"]: doc for doc in documents}
        self.embedder = None

    def search(self, question: str, top_k: int) -> tuple[list[dict[str, Any]]]:
        return (list(self.passage_map.values())[:top_k],)


class _Session:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add_records(self, records: list[Any]) -> list[Any]:
        self.added.extend(records)
        return list(records)


class _Control(WebRetrievalControl):
    """Only the promotion path is under test, so record what it would fetch."""

    def __init__(self, budget: int) -> None:
        self.max_collection_links_to_fetch = budget
        self.collection_link_fetch_tokens = 5000
        self.fetched: list[str] = []
        outer = self

        class _Builder:
            def build_enriched_records(self, document, **kwargs):
                outer.fetched.append(str(document.get("content_url") or ""))
                return [f"passage-for-{document.get('record_id')}"]

        self.corpus_builder = _Builder()


def _promote(budget: int) -> list[str]:
    control = _Control(budget)
    control._enrich_collection_links(
        question="which entry answers the question?",
        retriever=_Retriever([_stub(index) for index in range(1, STUB_COUNT + 1)]),
        corpus_session=_Session(),
    )
    return control.fetched


class CollectionLinkPromotionBudgetTest(unittest.TestCase):
    def test_default_budget_is_ten(self) -> None:
        default = WebRetrievalControl.__init__.__kwdefaults__[
            "max_collection_links_to_fetch"
        ]

        self.assertEqual(default, 10)

    def test_promotion_opens_links_up_to_the_budget(self) -> None:
        fetched = _promote(10)

        self.assertEqual(len(fetched), 10)
        self.assertEqual(len(set(fetched)), 10)

    def test_a_budget_of_three_stops_before_the_deeper_ranks(self) -> None:
        """Guard the regression direction: the old budget missed ranks 5 and 8."""

        fetched = _promote(3)

        self.assertEqual(len(fetched), 3)
        self.assertNotIn("https://example.org/entry/5", fetched)
        self.assertNotIn("https://example.org/entry/8", fetched)

    def test_default_budget_reaches_the_ranks_the_replay_found_answers_at(self) -> None:
        fetched = _promote(
            WebRetrievalControl.__init__.__kwdefaults__[
                "max_collection_links_to_fetch"
            ]
        )

        for rank in (1, 2, 5, 8):
            self.assertIn(f"https://example.org/entry/{rank}", fetched)

    def test_a_stub_pointing_at_its_own_page_is_not_promoted(self) -> None:
        control = _Control(10)
        same = _stub(1)
        same["content_url"] = same["parent_url"]

        control._enrich_collection_links(
            question="which entry answers the question?",
            retriever=_Retriever([same, _stub(2)]),
            corpus_session=_Session(),
        )

        self.assertEqual(control.fetched, ["https://example.org/entry/2"])


if __name__ == "__main__":
    unittest.main()
