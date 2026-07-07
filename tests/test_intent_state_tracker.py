from __future__ import annotations

import unittest

from tools.search_result_builder.next_hop_query import SearchIntentStateTracker
from tools.search_result_builder.query import SearchIntentPlan


class SearchIntentStateTrackerTests(unittest.TestCase):
    def tracker(self) -> SearchIntentStateTracker:
        return SearchIntentStateTracker()

    def test_paper_multi_hop_first_hop_needs_next_hop(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="paper",
            target="Find the authors of the paper.",
            must_include=["Pie Menus or Linear Menus, Which Is Better?", "2015", "authors"],
            avoid_terms=[],
        )
        updated = self.tracker().update(
            plan=plan,
            question='What was the title of the first paper authored by the one that had authored prior papers after "Pie Menus or Linear Menus, Which Is Better?" in 2015?',
            documents=[
                {
                    "title": "PDF Pie Menus or Linear Menus, Which Is Better?",
                    "text": "Pie Menus or Linear Menus, Which Is Better? 2015 authors Pietro Murano and Iram Khan.",
                    "url": "https://example.org/paper.pdf",
                }
            ],
        )

        self.assertEqual(updated.state, "needs_next_hop")
        self.assertIn("authors", updated.completed_terms)

    def test_paper_multi_hop_second_hop_can_be_sufficient(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="paper",
            target="Find the authors of the paper.",
            must_include=["Pie Menus or Linear Menus, Which Is Better?", "2015", "authors"],
            avoid_terms=[],
            state="needs_next_hop",
        )
        updated = self.tracker().update(
            plan=plan,
            question='What was the title of the first paper authored by the one that had authored prior papers after "Pie Menus or Linear Menus, Which Is Better?" in 2015?',
            documents=[
                {
                    "title": "Pietro Murano Publications",
                    "text": "Pietro Murano publications include Mapping Human Oriented Information to Software Agents for Online Systems Usage as an earlier publication.",
                    "url": "https://example.org/publications",
                }
            ],
        )

        self.assertEqual(updated.state, "sufficient")

    def test_official_page_without_answer_candidate_is_first_hop(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="official_page",
            target="Find official page.",
            must_include=["Merriam-Webster", "Word of the Day", "June 27 2022"],
            avoid_terms=[],
            preferred_domain="merriam-webster.com",
        )
        updated = self.tracker().update(
            plan=plan,
            question="What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?",
            documents=[
                {
                    "title": "Word of the Day: Jingoism",
                    "text": "June 27 2022 extreme patriotism or nationalism.",
                    "url": "https://www.merriam-webster.com/word-of-the-day/jingoism-2022-06-27",
                }
            ],
        )

        self.assertEqual(updated.state, "first_hop_satisfied")
        self.assertIn("answer_candidate:person", updated.missing_terms)

    def test_official_page_with_answer_candidate_is_sufficient(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="official_page",
            target="Find official page.",
            must_include=["Merriam-Webster", "Word of the Day", "June 27 2022"],
            avoid_terms=[],
            preferred_domain="merriam-webster.com",
        )
        updated = self.tracker().update(
            plan=plan,
            question="What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?",
            documents=[
                {
                    "title": "Word of the Day: Jingoism",
                    "text": "June 27 2022 quotation from Annie Levin.",
                    "url": "https://www.merriam-webster.com/word-of-the-day/jingoism-2022-06-27",
                }
            ],
        )

        self.assertEqual(updated.state, "sufficient")

    def test_definition_rule_found_is_sufficient(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="definition",
            target="Find botanical classification rules.",
            must_include=["botanical", "fruit", "vegetable"],
            avoid_terms=[],
        )
        updated = self.tracker().update(
            plan=plan,
            question="Classify a list using botanical fruit and vegetable rules.",
            documents=[
                {
                    "title": "Fruit vs Vegetable",
                    "text": "A botanical fruit develops from a flower ovary, while vegetable classification in cooking differs.",
                    "url": "https://example.org/fruit-vegetable",
                }
            ],
        )

        self.assertEqual(updated.state, "sufficient")

    def test_fact_missing_must_include_stays_pending(self):
        plan = SearchIntentPlan(
            search_needed=True,
            intent="fact",
            target="Find a fact.",
            must_include=["Mercedes Sosa", "2000", "2009"],
            avoid_terms=[],
        )
        updated = self.tracker().update(
            plan=plan,
            question="How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
            documents=[
                {
                    "title": "Albums",
                    "text": "A list of albums from 1990.",
                    "url": "https://example.org",
                }
            ],
        )

        self.assertEqual(updated.state, "pending")
        self.assertIn("Mercedes Sosa", updated.missing_terms)


if __name__ == "__main__":
    unittest.main()
