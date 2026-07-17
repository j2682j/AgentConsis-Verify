from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any


@dataclass(frozen=True)
class ChessPieceLocation:
    color: str
    piece: str
    square: str


@dataclass
class ChessBoardArtifact:
    fen: str = ""
    side_to_move: str = ""
    pieces: list[ChessPieceLocation] = field(default_factory=list)
    orientation: dict[str, Any] = field(default_factory=dict)
    uncertain_squares: list[str] = field(default_factory=list)
    valid: bool = False
    errors: list[str] = field(default_factory=list)


class ChessBoardExtractor:
    """Transcribe a board image and validate the result as a chess position."""

    _PIECE_SYMBOLS = {
        ("white", "king"): "K", ("white", "queen"): "Q",
        ("white", "rook"): "R", ("white", "bishop"): "B",
        ("white", "knight"): "N", ("white", "pawn"): "P",
        ("black", "king"): "k", ("black", "queen"): "q",
        ("black", "rook"): "r", ("black", "bishop"): "b",
        ("black", "knight"): "n", ("black", "pawn"): "p",
    }

    def __init__(self, *, model: str | None = None, timeout: int = 240, endpoint: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b")
        self.timeout = max(30, int(timeout))
        self.endpoint = endpoint or self._ollama_endpoint()

    def extract(self, file_path: str | Path, *, side_to_move: str = "black") -> ChessBoardArtifact:
        path = Path(file_path)
        if not path.is_file():
            return ChessBoardArtifact(errors=["image_file_not_found"])
        diagram_artifact = self._extract_standard_diagram(path, side_to_move=side_to_move)
        if diagram_artifact.valid:
            return diagram_artifact
        try:
            payload = self._request_transcription(path, side_to_move=side_to_move)
        except Exception as exc:
            return ChessBoardArtifact(
                errors=[
                    *diagram_artifact.errors,
                    f"vision_transcription_failed:{type(exc).__name__}:{exc}",
                ]
            )
        return self.from_payload(payload, side_to_move=side_to_move)

    def _extract_standard_diagram(
        self,
        path: Path,
        *,
        side_to_move: str,
    ) -> ChessBoardArtifact:
        try:
            import cairosvg
            import chess  # type: ignore
            import chess.svg  # type: ignore
            import numpy as np
            from PIL import Image
        except Exception as exc:
            return ChessBoardArtifact(errors=[f"diagram_runtime_unavailable:{exc}"])
        try:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            if min(width, height) < 320 or abs(width - height) / max(width, height) > 0.08:
                raise ValueError("image is not a near-square board diagram")
            image = image.resize((768, 768))
            templates: dict[str, list[Any]] = {}
            for symbol in "kqrbnp":
                templates[symbol] = []
                for size in range(72, 97, 3):
                    svg = chess.svg.piece(chess.Piece.from_symbol(symbol), size=size)
                    png = cairosvg.svg2png(
                        bytestring=svg.encode("utf-8"), output_width=size, output_height=size
                    )
                    rendered = Image.open(BytesIO(png)).convert("RGBA")
                    canvas = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
                    offset = (96 - size) // 2
                    canvas.alpha_composite(rendered, (offset, offset))
                    template = np.asarray(canvas)[:, :, 3] > 20
                    self._clear_mask_border(template)
                    templates[symbol].append(template)

            black_orientation = self._black_orientation(image, np)
            files = "hgfedcba" if black_orientation else "abcdefgh"
            pieces: list[dict[str, str]] = []
            parity_backgrounds: dict[int, list[Any]] = {0: [], 1: []}
            for row in range(8):
                for column in range(8):
                    cell = np.asarray(
                        image.crop((column * 96, row * 96, (column + 1) * 96, (row + 1) * 96))
                    )
                    corners = np.concatenate(
                        [
                            cell[:8, :8].reshape(-1, 3), cell[:8, -8:].reshape(-1, 3),
                            cell[-8:, :8].reshape(-1, 3), cell[-8:, -8:].reshape(-1, 3),
                        ]
                    )
                    background = np.median(corners, axis=0)
                    parity_backgrounds[(row + column) % 2].append(background)
                    foreground = np.linalg.norm(cell.astype(float) - background, axis=2) > 35
                    self._clear_mask_border(foreground)
                    if int(foreground.sum()) < 1500:
                        continue
                    ranked = []
                    for symbol, symbol_templates in templates.items():
                        symbol_score = 0.0
                        for template in symbol_templates:
                            denominator = int(foreground.sum()) + int(template.sum())
                            overlap = int(np.logical_and(foreground, template).sum())
                            symbol_score = max(symbol_score, 2 * overlap / max(1, denominator))
                        ranked.append((symbol_score, symbol))
                    ranked.sort(reverse=True)
                    similarity, piece_symbol = ranked[0]
                    if {ranked[0][1], ranked[1][1]} == {"b", "r"} and ranked[0][0] - ranked[1][0] < 0.03:
                        piece_symbol = self._bishop_or_rook(foreground, np)
                    if similarity < 0.62:
                        raise ValueError(
                            f"unrecognized piece at screen cell {row},{column}: {similarity:.3f}"
                        )
                    luminance = cell[foreground].mean(axis=1)
                    color = "white" if float(np.median(luminance)) > 130 else "black"
                    rank = row + 1 if black_orientation else 8 - row
                    pieces.append(
                        {
                            "color": color,
                            "piece": {
                                "k": "king", "q": "queen", "r": "rook",
                                "b": "bishop", "n": "knight", "p": "pawn",
                            }[piece_symbol],
                            "square": f"{files[column]}{rank}",
                        }
                    )
            first_background = np.median(np.asarray(parity_backgrounds[0]), axis=0)
            second_background = np.median(np.asarray(parity_backgrounds[1]), axis=0)
            if float(np.linalg.norm(first_background - second_background)) < 20:
                raise ValueError("checkerboard background could not be verified")
            payload = {
                "side_to_move": side_to_move,
                "orientation": {
                    "top_rank": 1 if black_orientation else 8,
                    "bottom_rank": 8 if black_orientation else 1,
                    "left_file": files[0],
                    "right_file": files[-1],
                    "method": "checkerboard_svg_shape_matching",
                },
                "pieces": pieces,
                "uncertain_squares": [],
            }
            artifact = self.from_payload(payload, side_to_move=side_to_move)
            if not artifact.valid:
                artifact.errors.insert(0, "diagram_position_validation_failed")
            return artifact
        except Exception as exc:
            return ChessBoardArtifact(errors=[f"diagram_extraction_failed:{type(exc).__name__}:{exc}"])

    @staticmethod
    def _black_orientation(image: Any, np: Any) -> bool:
        counts: list[int] = []
        for row in (0, 7):
            crop = np.asarray(image.crop((4, row * 96 + 4, 26, row * 96 + 30)))
            background = np.median(crop[-5:, -5:].reshape(-1, 3), axis=0)
            foreground = np.linalg.norm(crop.astype(float) - background, axis=2) > 30
            counts.append(int(foreground.sum()))
        if min(counts) <= 0 or max(counts) < min(counts) * 1.25:
            raise ValueError("rank-label orientation is ambiguous")
        # The glyph '8' has substantially more foreground than the glyph '1'.
        return counts[0] < counts[1]

    @staticmethod
    def _clear_mask_border(mask: Any) -> None:
        mask[:5] = False
        mask[-5:] = False
        mask[:, :5] = False
        mask[:, -5:] = False

    @staticmethod
    def _bishop_or_rook(mask: Any, np: Any) -> str:
        y_values, x_values = np.where(mask)
        if not len(y_values):
            return "r"
        x_min, x_max = int(x_values.min()), int(x_values.max())
        y_min, y_max = int(y_values.min()), int(y_values.max())
        cropped = mask[y_min:y_max + 1, x_min:x_max + 1]
        sample_row = cropped[min(cropped.shape[0] - 1, int(cropped.shape[0] * 0.15))]
        upper_width_ratio = float(sample_row.sum()) / max(1, cropped.shape[1])
        return "b" if upper_width_ratio < 0.45 else "r"

    def from_payload(self, payload: dict[str, Any], *, side_to_move: str = "black") -> ChessBoardArtifact:
        requested_side = str(payload.get("side_to_move") or side_to_move).strip().lower()
        if requested_side not in {"white", "black"}:
            requested_side = side_to_move
        errors: list[str] = []
        pieces: list[ChessPieceLocation] = []
        occupied: set[str] = set()
        for item in list(payload.get("pieces") or []):
            if not isinstance(item, dict):
                errors.append("invalid_piece_record")
                continue
            color = str(item.get("color") or "").strip().lower()
            piece = str(item.get("piece") or "").strip().lower()
            square = str(item.get("square") or "").strip().lower()
            if (color, piece) not in self._PIECE_SYMBOLS or not re.fullmatch(r"[a-h][1-8]", square):
                errors.append(f"invalid_piece:{color}:{piece}:{square}")
                continue
            if square in occupied:
                errors.append(f"duplicate_square:{square}")
                continue
            occupied.add(square)
            pieces.append(ChessPieceLocation(color=color, piece=piece, square=square))
        for color in ("white", "black"):
            kings = sum(1 for item in pieces if item.color == color and item.piece == "king")
            if kings != 1:
                errors.append(f"{color}_king_count:{kings}")
        uncertain = [
            str(value).strip().lower()
            for value in list(payload.get("uncertain_squares") or [])
            if re.fullmatch(r"[a-h][1-8]", str(value).strip().lower())
        ]
        if uncertain:
            errors.append("uncertain_piece_locations")
        fen = ""
        if not errors:
            try:
                fen = self._to_fen(pieces, requested_side)
                self._validate_fen(fen)
            except Exception as exc:
                errors.append(f"invalid_chess_position:{exc}")
                fen = ""
        return ChessBoardArtifact(
            fen=fen,
            side_to_move=requested_side,
            pieces=pieces,
            orientation=dict(payload.get("orientation") or {}),
            uncertain_squares=uncertain,
            valid=bool(fen and not errors),
            errors=errors,
        )

    def _request_transcription(self, path: Path, *, side_to_move: str) -> dict[str, Any]:
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "Transcribe the chessboard attachment into one JSON object. "
            "Read coordinate labels printed on the image; never assume screen orientation. "
            "List every visible piece exactly once. Do not solve or explain. "
            "Use piece names king, queen, rook, bishop, knight, pawn. "
            f"The stated side to move is {side_to_move}. "
            "Schema: {\"side_to_move\":\"black\",\"orientation\":{},"
            "\"pieces\":[{\"color\":\"white\",\"piece\":\"king\",\"square\":\"g1\"}],"
            "\"uncertain_squares\":[]}."
        )
        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False, "think": False, "keep_alive": 0, "format": "json",
            "options": {"temperature": 0, "num_predict": 4096},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama vision HTTP {exc.code}: {detail}") from exc
        message = raw.get("message") if isinstance(raw, dict) else None
        content = str((message or {}).get("content") or "").strip()
        if not content:
            raise ValueError("vision model returned no non-reasoning JSON content")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("vision transcription is not a JSON object")
        return value

    def _to_fen(self, pieces: list[ChessPieceLocation], side_to_move: str) -> str:
        board = {item.square: self._PIECE_SYMBOLS[(item.color, item.piece)] for item in pieces}
        ranks: list[str] = []
        for rank in range(8, 0, -1):
            empty = 0
            row = ""
            for file_name in "abcdefgh":
                symbol = board.get(f"{file_name}{rank}")
                if symbol is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                row += symbol
            if empty:
                row += str(empty)
            ranks.append(row)
        turn = "w" if side_to_move == "white" else "b"
        return f"{'/'.join(ranks)} {turn} - - 0 1"

    @staticmethod
    def _validate_fen(fen: str) -> None:
        import chess  # type: ignore

        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("python-chess rejected the reconstructed board")

    @staticmethod
    def _ollama_endpoint() -> str:
        base_url = (
            os.getenv("OLLAMA_NATIVE_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        ).strip()
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url.rstrip("/") + "/api/chat"


__all__ = ["ChessBoardArtifact", "ChessBoardExtractor", "ChessPieceLocation"]
