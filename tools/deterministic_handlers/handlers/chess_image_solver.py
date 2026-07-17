from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
from typing import Any

from tools.attachment_reader.specialized import ChessBoardExtractor

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class ChessImageSolverRouterHandler:
    name = "chess_image_solver"
    uses_specialized_attachment_parser = True
    handler_role = "chess_tactics"
    capability_description = (
        "Transcribe a chessboard image into a validated FEN position and compute the best "
        "move with a deterministic chess engine."
    )
    supported_attachment_types = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    supported_task_roles = {"chess_tactics"}
    supported_answer_roles = {"move"}
    input_schema = io_contract(
        name,
        [
            input_field("file_path", "str", True, "Chessboard image path.", "attachment"),
            input_field("side_to_move", "str", True, "Side whose best move is requested.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def __init__(self, extractor: ChessBoardExtractor | None = None, *, depth: int = 18) -> None:
        self.extractor = extractor or ChessBoardExtractor()
        self.depth = max(10, int(depth))

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        path = self._file_path(handler_input)
        lowered = handler_input.question.lower()
        ready = bool(
            path and Path(path).suffix.lower() in self.supported_attachment_types
            and ("chess" in lowered or "algebraic notation" in lowered)
            and ("move" in lowered or "turn" in lowered)
        )
        return HandlerMatch(
            handler_name=self.name, matched=ready, confidence=0.99 if ready else 0.0,
            reason="chess_image_and_best_move_request", handler_role=self.handler_role,
            missing_inputs=[] if ready else ["chessboard_image_or_best_move_request"],
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        lowered = handler_input.question.lower()
        side = "white" if "white's turn" in lowered or "white to move" in lowered else "black"
        return {"question": handler_input.question, "file_path": self._file_path(handler_input), "side_to_move": side}

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        file_path = str(inputs.get("file_path") or "").strip()
        side = str(inputs.get("side_to_move") or "black").strip().lower()
        artifact = self.extractor.extract(file_path, side_to_move=side)
        if not artifact.valid:
            return HandlerResult.missing(
                handler_name=self.name, missing_inputs=["validated_chess_position"],
                structured_result={"file_path": file_path, "extraction_errors": artifact.errors, "piece_count": len(artifact.pieces)},
                next_action_hint="Re-extract a complete board position before attempting chess analysis.",
            )
        engine_path = self._stockfish_path()
        if not engine_path:
            return HandlerResult.missing(
                handler_name=self.name, missing_inputs=["stockfish_executable"],
                structured_result={"fen": artifact.fen}, next_action_hint="Install Stockfish or set STOCKFISH_PATH.",
            )
        try:
            answer, detail = self._best_move(artifact.fen, engine_path)
        except Exception as exc:
            return HandlerResult.error_result(handler_name=self.name, error=f"chess engine failed: {exc}")
        return HandlerResult(
            handler_name=self.name, status="ok", answer=answer, confidence=0.99,
            output_type="final_answer", semantic_role="chess_best_move_san",
            supporting_inputs=[artifact.fen, file_path],
            structured_result={
                "task_type": "chess_best_move", "operation": "engine_best_move",
                "fen": artifact.fen, "piece_count": len(artifact.pieces),
                "orientation": artifact.orientation, **detail,
                "input_provenance": {
                    "source": "specialized_attachment_input",
                    "file_path": file_path,
                    "parse_status": "success",
                },
            },
        )

    def _best_move(self, fen: str, engine_path: str) -> tuple[str, dict[str, Any]]:
        import chess  # type: ignore
        import chess.engine  # type: ignore

        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=2)
        rows = info if isinstance(info, list) else [info]
        if not rows or not rows[0].get("pv"):
            raise ValueError("engine returned no principal variation")
        move = rows[0]["pv"][0]
        alternatives = [
            {"move": board.san(row["pv"][0]), "score": str(row.get("score", "")), "depth": int(row.get("depth") or 0)}
            for row in rows if row.get("pv")
        ]
        return board.san(move), {"engine": "Stockfish", "engine_path": engine_path, "alternatives": alternatives}

    @staticmethod
    def _file_path(handler_input: HandlerInput) -> str:
        adapted = handler_input.adapted_inputs()
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        return str(adapted.get("file_path") or attachment.get("file_path") or attachment.get("path") or "").strip()

    @staticmethod
    def _stockfish_path() -> str:
        candidates = [str(os.getenv("STOCKFISH_PATH") or "").strip(), str(shutil.which("stockfish") or "")]
        candidates.extend(glob.glob(str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_*" / "stockfish/stockfish*.exe")))
        return next((value for value in candidates if value and Path(value).is_file()), "")


__all__ = ["ChessImageSolverRouterHandler"]
