from __future__ import annotations

import unittest
from pathlib import Path

from tools.attachment_reader.specialized import ChessBoardArtifact, ChessBoardExtractor, ChessPieceLocation
from tools.deterministic_handlers.handlers.chess_image_solver import ChessImageSolverRouterHandler


class _FixtureExtractor:
    def extract(self, file_path: str, *, side_to_move: str = "black") -> ChessBoardArtifact:
        del file_path, side_to_move
        return ChessBoardArtifact(
            fen="3r2k1/pp3pp1/4b2p/7Q/3n4/PqBBR2P/5PP1/6K1 b - - 0 1",
            side_to_move="black", pieces=[ChessPieceLocation("white", "king", "g1")], valid=True,
        )


class ChessImageSolverHandlerTests(unittest.TestCase):
    def test_standard_diagram_is_reconstructed_as_expected_fen(self) -> None:
        image_path = (
            Path(__file__).resolve().parents[1]
            / "data/gaia/2023/validation/cca530fc-4052-43b2-b130-b30968d8aa44.png"
        )
        artifact = ChessBoardExtractor()._extract_standard_diagram(
            image_path, side_to_move="black"
        )

        self.assertTrue(artifact.valid, artifact.errors)
        self.assertEqual(
            artifact.fen,
            "3r2k1/pp3pp1/4b2p/7Q/3n4/PqBBR2P/5PP1/6K1 b - - 0 1",
        )

    def test_stockfish_returns_expected_best_move(self) -> None:
        result = ChessImageSolverRouterHandler(extractor=_FixtureExtractor(), depth=16).run(
            {"file_path": "board.png", "side_to_move": "black"}
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.answer, "Rd5")
        self.assertEqual(result.semantic_role, "chess_best_move_san")


if __name__ == "__main__":
    unittest.main()
