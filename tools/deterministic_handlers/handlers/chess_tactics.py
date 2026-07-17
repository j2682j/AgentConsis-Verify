from __future__ import annotations

import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract


class ChessTacticsRouterHandler:
    name = "chess_tactics"
    handler_role = "chess_tactics"
    capability_description = (
        "Validate chess positions from FEN and answer simple legal-move or checkmate-status tasks. "
        "Requires python-chess for execution."
    )
    supported_attachment_types: set[str] = {".txt", ".json", ".pgn"}
    supported_task_roles: set[str] = {"chess_tactics"}
    supported_answer_roles: set[str] = {"move", "boolean", "yes_no"}
    input_schema = io_contract(
        name,
        [
            input_field("fen", "str", True, "Chess FEN board position.", "question|attachment"),
            input_field("operation", "str", True, "legal_moves, checkmate_status, or best_move_validation.", "question"),
        ],
        default_outputs(),
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        fen = self._extract_fen(handler_input.combined_text())
        operation = self._operation(handler_input.question)
        missing = []
        if not fen:
            missing.append("fen")
        if not operation:
            missing.append("chess_operation")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.88 if not missing else 0.2,
            reason="chess_fen_and_operation_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        return {
            "question": handler_input.question,
            "fen": self._extract_fen(handler_input.combined_text()),
            "operation": self._operation(handler_input.question),
            "candidate_move": self._candidate_move(handler_input.question),
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        try:
            import chess  # type: ignore
        except Exception:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["python_chess"],
                next_action_hint="Install python-chess or provide a chess-specific handler runtime.",
                structured_result={"operation": inputs.get("operation"), "fen": inputs.get("fen")},
            )

        fen = str(inputs.get("fen") or "").strip()
        operation = str(inputs.get("operation") or "").strip()
        if not fen or not operation:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=[
                    item for item, value in (("fen", fen), ("chess_operation", operation)) if not value
                ],
            )
        try:
            board = chess.Board(fen)
        except Exception as exc:
            return HandlerResult.error_result(handler_name=self.name, error=f"invalid FEN: {exc}")

        if operation == "checkmate_status":
            answer = "yes" if board.is_checkmate() else "no"
            detail = {"is_checkmate": board.is_checkmate(), "is_check": board.is_check()}
        elif operation == "legal_moves":
            moves = sorted(board.san(move) for move in board.legal_moves)
            answer = ", ".join(moves)
            detail = {"legal_moves": moves, "legal_move_count": len(moves)}
        elif operation == "move_legal":
            candidate = str(inputs.get("candidate_move") or "").strip()
            if not candidate:
                return HandlerResult.missing(
                    handler_name=self.name,
                    missing_inputs=["candidate_move"],
                    next_action_hint="Provide a candidate SAN or UCI move to validate.",
                )
            legal = self._is_legal_move(board, candidate, chess)
            answer = "yes" if legal else "no"
            detail = {"candidate_move": candidate, "legal": legal}
        else:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["chess_operation"],
            )

        structured = {
            "task_type": f"chess_{operation}",
            "fen": fen,
            "operation": operation,
            **detail,
        }
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"FEN: {fen}\n"
                f"Operation: {operation}\n"
                f"Answer: {answer}\n"
                "Instruction: use this result only for the stated chess-position task."
            ),
            structured_result=structured,
            confidence=0.9,
            output_type="final_answer",
            semantic_role=f"chess_{operation}_answer",
            supporting_inputs=[fen],
        )

    def _extract_fen(self, text: str) -> str:
        match = re.search(
            r"([rnbqkpRNBQKP1-8/]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+)",
            text or "",
        )
        return match.group(1).strip() if match else ""

    def _operation(self, question: str) -> str:
        lowered = str(question or "").lower()
        if "checkmate" in lowered or "mate" in lowered:
            return "checkmate_status"
        if "legal move" in lowered or "all legal" in lowered:
            return "legal_moves"
        if "is" in lowered and re.search(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", lowered):
            return "move_legal"
        return ""

    def _candidate_move(self, question: str) -> str:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', question or "")
        for match in quoted:
            value = next((part for part in match if part), "").strip()
            if value:
                return value
        move = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", question or "", re.IGNORECASE)
        return move.group(1) if move else ""

    def _is_legal_move(self, board: Any, candidate: str, chess_module: Any) -> bool:
        try:
            move = board.parse_san(candidate)
            return move in board.legal_moves
        except Exception:
            pass
        try:
            move = chess_module.Move.from_uci(candidate.lower())
            return move in board.legal_moves
        except Exception:
            return False


__all__ = ["ChessTacticsRouterHandler"]
