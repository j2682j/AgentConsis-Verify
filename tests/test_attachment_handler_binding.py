"""Every attachment-parsing handler must be reachable by the router.

A handler with no input adapter reports ``attachment_bound=False``, which
removes it from the eligible-capability set before the strategy planner ever
sees it. Nothing errors — the handler is simply never invoked. Two handlers
were dead this way, so the binding is asserted here rather than maintained by
hand.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.deterministic_handlers.base import HandlerInput
from tools.deterministic_handlers.input_adapters import HandlerInputAdapterRegistry
from tools.deterministic_handlers.input_adapters.attachment_file_adapter import (
    specialized_attachment_handler_names,
)
from tools.deterministic_handlers.registry import default_deterministic_registry
from tools.deterministic_handlers.router import DeterministicHandlerRouter


GAIA = Path(__file__).resolve().parents[1] / "data" / "gaia" / "2023" / "validation"
ROAD_TXT = GAIA / "389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt"
GRID_XLSX = GAIA / "65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx"

ROAD_QUESTION = (
    "You are a telecommunications engineer who wants to build cell phone towers "
    "on a stretch of road. In the reference file is a layout of the road and "
    "nearby houses. Each dash is a mile marker and each capital H is a house. "
    "Each cell phone tower can cover houses within a 4-mile radius. Find the "
    "minimum number of cell phone towers needed to cover all houses next to the "
    "road. Your answer should be a positive numerical integer value."
)
GRID_QUESTION = (
    "You are given this Excel file as a map. You start on the START cell and "
    "move toward the END cell. You are allowed to move two cells per turn, and "
    "you may move up, down, left, or right. You may not move fewer than two "
    "cells, and you may not move backward. You must avoid moving onto any blue "
    "cells. On the eleventh turn, what is the 6-digit hex code (without prefix) "
    "of the color of the cell where you land after moving?"
)


class AttachmentHandlerBindingTests(unittest.TestCase):
    def test_every_specialized_handler_has_an_adapter(self) -> None:
        registry = default_deterministic_registry()
        adapters = HandlerInputAdapterRegistry()
        declared = {
            handler.name
            for handler in registry.list_handlers()
            if getattr(handler, "uses_specialized_attachment_parser", False)
        }
        self.assertTrue(declared, "expected at least one specialized attachment handler")
        for name in declared:
            self.assertTrue(
                adapters.has_adapter(name),
                f"handler {name!r} parses the attachment itself but has no input "
                f"adapter, so the router would silently never invoke it",
            )

    def test_derived_names_match_handler_declarations(self) -> None:
        registry = default_deterministic_registry()
        declared = {
            handler.name
            for handler in registry.list_handlers()
            if getattr(handler, "uses_specialized_attachment_parser", False)
        }
        self.assertEqual(set(specialized_attachment_handler_names()), declared)

    @unittest.skipUnless(ROAD_TXT.is_file(), "GAIA validation attachment not present")
    def test_road_coverage_handler_is_offered_to_the_planner(self) -> None:
        router = DeterministicHandlerRouter()
        capabilities, _ = router.eligible_capabilities(
            question=ROAD_QUESTION,
            attachment={"file_path": str(ROAD_TXT), "extension": ".txt"},
            metadata={
                "attachment_profile": {"parse_status": "success"},
                "require_attachment_provenance": True,
            },
        )
        self.assertIn(
            "road_interval_coverage",
            [item.handler_name for item in capabilities],
        )

    @unittest.skipUnless(GRID_XLSX.is_file(), "GAIA validation attachment not present")
    def test_grid_path_handler_is_offered_and_uniquely_matches(self) -> None:
        router = DeterministicHandlerRouter()
        capabilities, _ = router.eligible_capabilities(
            question=GRID_QUESTION,
            attachment={"file_path": str(GRID_XLSX), "extension": ".xlsx"},
            metadata={
                "attachment_profile": {"parse_status": "success"},
                "require_attachment_provenance": True,
            },
        )
        offered = [item.handler_name for item in capabilities]
        self.assertIn("color_grid_path", offered)

        # Both grid handlers are offered, but only the navigation one claims
        # this question — the cycle handler must not match a routing question.
        registry = default_deterministic_registry()
        handler_input = HandlerInput(
            question=GRID_QUESTION, attachment={"file_path": str(GRID_XLSX)}
        )
        matched = [
            name
            for name in offered
            if registry.get(name) is not None
            and registry.get(name).match_input(handler_input).matched
        ]
        self.assertEqual(matched, ["color_grid_path"])


if __name__ == "__main__":
    unittest.main()
