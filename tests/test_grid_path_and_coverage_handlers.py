import unittest
from pathlib import Path

from tools.deterministic_handlers.base import HandlerInput
from tools.deterministic_handlers.handlers.color_grid_path import (
    ColorGridPathRouterHandler,
)
from tools.deterministic_handlers.handlers.road_interval_coverage import (
    RoadIntervalCoverageRouterHandler,
)


GAIA_VALIDATION = Path(__file__).resolve().parents[1] / "data" / "gaia" / "2023" / "validation"
GRID_XLSX = GAIA_VALIDATION / "65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx"
ROAD_TXT = GAIA_VALIDATION / "389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt"

GRID_QUESTION = (
    "You are given this Excel file as a map. You start on the START cell and "
    "move toward the END cell. You are allowed to move two cells per turn, and "
    "you may move up, down, left, or right. You may not move fewer than two "
    "cells, and you may not move backward. You must avoid moving onto any blue "
    "cells. On the eleventh turn, what is the 6-digit hex code (without prefix) "
    "of the color of the cell where you land after moving?"
)
ROAD_QUESTION = (
    "You are a telecommunications engineer who wants to build cell phone towers "
    "on a stretch of road. In the reference file is a layout of the road and "
    "nearby houses. Each dash, \"-\", is a marker indicating a mile. Each "
    "capital H indicates a house located next to a mile marker, appearing above "
    "or below the stretch of road. Each cell phone tower can cover houses "
    "located next to the road within a 4-mile radius. Find the minimum number "
    "of cell phone towers needed to cover all houses next to the road. Your "
    "answer should be a positive numerical integer value."
)


class ColorGridPathHandlerTests(unittest.TestCase):
    def test_question_parsing(self) -> None:
        handler = ColorGridPathRouterHandler()
        question = GRID_QUESTION.casefold()
        self.assertEqual(handler._avoid_color(question), "blue")
        self.assertEqual(handler._turn_number(question), 11)
        self.assertEqual(handler._cells_per_turn(question), 2)

    def test_match_requires_grid_inputs(self) -> None:
        handler = ColorGridPathRouterHandler()
        match = handler.match_input(
            HandlerInput(
                question=GRID_QUESTION,
                attachment={"file_path": str(GRID_XLSX)},
            )
        )
        self.assertTrue(match.matched)
        no_match = handler.match_input(
            HandlerInput(question="What is the capital of France?", attachment={})
        )
        self.assertFalse(no_match.matched)

    @unittest.skipUnless(GRID_XLSX.is_file(), "GAIA validation attachment not present")
    def test_gaia_grid_landing_color(self) -> None:
        handler = ColorGridPathRouterHandler()
        result = handler.run(
            handler.build_input(
                HandlerInput(
                    question=GRID_QUESTION,
                    attachment={"file_path": str(GRID_XLSX)},
                )
            )
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.answer, "F478A7")


class RoadIntervalCoverageHandlerTests(unittest.TestCase):
    def test_radius_parsing(self) -> None:
        handler = RoadIntervalCoverageRouterHandler()
        self.assertEqual(handler._coverage_radius(ROAD_QUESTION.casefold()), 4)

    def test_greedy_positions(self) -> None:
        handler = RoadIntervalCoverageRouterHandler()
        text = "  H   H\n----------\n H       H"
        self.assertEqual(handler._marker_positions(text), [1, 2, 6, 9])

    @unittest.skipUnless(ROAD_TXT.is_file(), "GAIA validation attachment not present")
    def test_gaia_minimum_towers(self) -> None:
        handler = RoadIntervalCoverageRouterHandler()
        result = handler.run(
            handler.build_input(
                HandlerInput(
                    question=ROAD_QUESTION,
                    attachment={"file_path": str(ROAD_TXT)},
                )
            )
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.answer, "3")


if __name__ == "__main__":
    unittest.main()
