from __future__ import annotations

import json
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, lower_text


class GridWordHandler:
    """Solve Boggle-style word search with eight-direction DFS."""

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def solve(self, question: str, **_: Any) -> DeterministicSolverResult:
        lowered = lower_text(question)
        if not any(term in lowered for term in ("boggle", "letter grid", "word search")):
            return DeterministicSolverResult.miss("grid_word")
        payload = self._json_payload(question)
        grid = self._normalize_grid(payload.get("grid"))
        words = [
            clean_text(word).upper()
            for word in payload.get("words", [])
            if clean_text(word)
        ]
        if not grid or not words:
            return DeterministicSolverResult.miss(
                "grid_word",
                "provide JSON with grid and words",
            )

        found = [word for word in words if self._exists(grid, word)]
        if len(words) == 1 and any(term in lowered for term in ("can", "exists", "formed")):
            answer = "yes" if found else "no"
        else:
            answer = ", ".join(found)
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type="boggle_dfs",
            answer=answer,
            answer_text=answer,
            confidence=0.98,
            evidence={"grid": grid, "words": words, "found": found},
        )

    def _exists(self, grid: list[list[str]], word: str) -> bool:
        rows = len(grid)
        cols = len(grid[0])

        def dfs(row: int, col: int, index: int, visited: set[tuple[int, int]]) -> bool:
            if grid[row][col] != word[index]:
                return False
            if index == len(word) - 1:
                return True
            visited.add((row, col))
            for row_delta, col_delta in self.DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if (next_row, next_col) in visited:
                    continue
                if dfs(next_row, next_col, index + 1, visited):
                    visited.remove((row, col))
                    return True
            visited.remove((row, col))
            return False

        return any(
            dfs(row, col, 0, set())
            for row in range(rows)
            for col in range(cols)
        )

    def _normalize_grid(self, value: Any) -> list[list[str]]:
        if not isinstance(value, list) or not value:
            return []
        grid: list[list[str]] = []
        for row in value:
            cells = list(row) if isinstance(row, str) else row
            if not isinstance(cells, list) or not cells:
                return []
            grid.append([clean_text(cell).upper() for cell in cells])
        width = len(grid[0])
        return grid if width and all(len(row) == width for row in grid) else []

    def _json_payload(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


__all__ = ["GridWordHandler"]
