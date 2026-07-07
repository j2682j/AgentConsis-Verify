import unittest

from tools.search_result_builder.next_hop_query.query_guard import NextHopQueryGuard
from tools.search_result_builder.query.search_intent_planner import SearchIntentPlan


class NextHopQueryGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = NextHopQueryGuard()

    def test_accepts_query_covering_intent(self) -> None:
        plan = SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target="Doctor Who Series 9 Episode 11 official script maze location",
            must_include=["Doctor Who", "Series 9", "Episode 11", "maze location"],
            avoid_terms=[],
        )
        result = self.guard.validate(
            original_question="What is the maze location in the official script?",
            current_query="Doctor Who Series 9 Episode 11 script",
            proposed_next_query="Doctor Who Series 9 Episode 11 official script maze location",
            intent_plan=plan,
            useful_spans=[],
            seen_query_keys=set(),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")

    def test_rejects_noise_dominated_query(self) -> None:
        plan = SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target="Doctor Who Series 9 Episode 11 official script maze location",
            must_include=["Doctor Who", "Series 9", "Episode 11", "maze location"],
            avoid_terms=[],
        )
        result = self.guard.validate(
            original_question="What is the maze location in the official script?",
            current_query="Doctor Who Series 9 Episode 11 script",
            proposed_next_query="COUGHING POWERING WHEEZING WHIRRING Advantage",
            intent_plan=plan,
            useful_spans=[],
            seen_query_keys=set(),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "low_must_include_coverage")
        self.assertIn("Doctor Who", result.query)
        self.assertIn("maze location", result.query)

    def test_keeps_preferred_domain_in_fallback(self) -> None:
        plan = SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target="Bielefeld University Library BASE DDC 633 unknown language article",
            must_include=["Bielefeld University Library", "BASE", "DDC 633", "unknown language article"],
            avoid_terms=[],
            preferred_domain="base.bielefeld.de",
            missing_terms=["unknown language article"],
        )
        result = self.guard.validate(
            original_question="Under DDC 633 on BASE, what country was the unknown language article from?",
            current_query="site:base.bielefeld.de Bielefeld University Library BASE DDC 633",
            proposed_next_query="2020 DNB Deutsch outside article",
            intent_plan=plan,
            useful_spans=["flag unique country"],
            seen_query_keys=set(),
        )
        self.assertFalse(result.accepted)
        self.assertIn("site:base.bielefeld.de", result.query)
        self.assertIn("unknown language article", result.query)

    def test_fallback_uses_missing_terms_and_useful_spans(self) -> None:
        plan = SearchIntentPlan(
            search_needed=True,
            intent="paper",
            target="Pie Menus or Linear Menus, Which Is Better?",
            must_include=["Pie Menus or Linear Menus, Which Is Better?", "2015", "authors"],
            avoid_terms=[],
            missing_terms=["authors"],
        )
        result = self.guard.validate(
            original_question="Find the first paper title authored by the prior-paper author.",
            current_query="Pie Menus or Linear Menus Which Is Better 2015 authors",
            proposed_next_query="Presentation Synchronization 203 204 237",
            intent_plan=plan,
            useful_spans=["Iram Khan"],
            seen_query_keys=set(),
        )
        self.assertFalse(result.accepted)
        self.assertIn("authors", result.query)
        self.assertIn("Iram Khan", result.query)

    def test_duplicate_fallback_returns_empty_query(self) -> None:
        plan = SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target="Mercedes Sosa studio albums 2000 2009",
            must_include=["Mercedes Sosa", "studio albums", "2000", "2009"],
            avoid_terms=[],
        )
        duplicate = "Mercedes Sosa studio albums 2000 2009"
        result = self.guard.validate(
            original_question="How many albums were published?",
            current_query=duplicate,
            proposed_next_query="album list",
            intent_plan=plan,
            useful_spans=[],
            seen_query_keys={duplicate.casefold()},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.query, "")
        self.assertIn("fallback_duplicate", result.reason)


if __name__ == "__main__":
    unittest.main()
